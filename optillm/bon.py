import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import optillm
from optillm import conversation_logger
from optillm.config_loader import load_complex_profile

logger = logging.getLogger(__name__)

# BON requires six role entries per configuration:
#   - First four drive candidate generation
#   - QUALITY_RATER scores the candidates
#   - FINAL_SYNTHESIZER optionally combines top results
BON_REQUIRED_ROLE_NAMES = [
    "CREATIVE",
    "ANALYST",
    "CRITIC",
    "SYNTHESIZER",
    "QUALITY_RATER",
    "FINAL_SYNTHESIZER",
]
BON_GENERATION_ROLE_COUNT = 4
DEFAULT_REASONING_EFFORT = "high"

# Lazy-loaded LiteLLM client for OpenRouter models
_litellm_client = None


def _validate_bon_profile(config: List[Dict[str, Any]]) -> bool:
    """Ensure a profile defines all required BON roles."""

    if len(config) < len(BON_REQUIRED_ROLE_NAMES):
        logger.error(
            "BON profile has %s roles but requires %s (%s)",
            len(config),
            len(BON_REQUIRED_ROLE_NAMES),
            ", ".join(BON_REQUIRED_ROLE_NAMES),
        )
        return False

    defined_names = {role.get("name") for role in config if isinstance(role, dict)}
    missing = [name for name in BON_REQUIRED_ROLE_NAMES if name not in defined_names]
    if missing:
        logger.error(
            "BON profile is missing required roles: %s",
            ", ".join(missing),
        )
        return False

    return True


def _get_litellm_client():
    """Get or create the LiteLLM client for OpenRouter models (thread-safe singleton)."""
    global _litellm_client
    if _litellm_client is None:
        from optillm.litellm_wrapper import LiteLLMWrapper

        _litellm_client = LiteLLMWrapper()
        logger.info("BON: Created LiteLLM client for OpenRouter models")
    return _litellm_client


def _get_role_by_name(
    config: List[Dict[str, Any]],
    role_name: str,
    default_model: str = "openrouter/google/gemini-3-flash-preview",
) -> Dict[str, Any]:
    """Find a role configuration by name from the config list.

    Args:
        config: Full configuration list
        role_name: Name of the role to find (e.g., 'QUALITY_RATER', 'SYNTHESIZER')
        default_model: Default model if role not found

    Returns:
        Role configuration dict with model, reasoning_effort, etc.
    """
    for role in config:
        if role.get("name") == role_name:
            return role
    # Fallback to default model
    logger.warning(
        f"Role {role_name} not found in config, using default model {default_model}"
    )
    return {"name": role_name, "model": default_model, "reasoning_effort": "high"}


def _generate_bon_candidate(
    client, default_model, system_prompt, initial_query, max_tokens, candidate, index
):
    """Generate a single BON candidate (thread-safe) with multi-model support."""
    candidate_name = candidate["name"]
    candidate_model = candidate.get("model", default_model)
    candidate_effort = candidate.get("reasoning_effort", "high")

    # Route to appropriate client based on model prefix
    # OpenRouter models go through LiteLLM, others use the default client (e.g., Z.ai)
    if candidate_model.startswith("openrouter/"):
        agent_client = _get_litellm_client()
        provider = "OpenRouter (via LiteLLM)"
    else:
        agent_client = client
        provider = f"Direct ({type(client).__name__})"

    # Log the candidate configuration
    logger.info(
        f"{candidate_name}: Using model='{candidate_model}' via {provider}, "
        f"reasoning_effort={candidate_effort}, max_tokens={max_tokens}"
    )

    provider_request = {
        "model": candidate_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_query},
        ],
        "max_tokens": max_tokens,
        "reasoning_effort": candidate_effort,
    }

    response = optillm.safe_completions_create(agent_client, provider_request)

    return {
        "index": index,
        "name": candidate_name,
        "model": candidate_model,
        "effort": candidate_effort,
        "content": response.choices[0].message.content,
        "tokens": response.usage.completion_tokens,
        "response_dict": (
            response.model_dump() if hasattr(response, "model_dump") else response
        ),
        "request": provider_request,
    }


def _determine_default_model(preferred_model: Optional[str]) -> str:
    """Return the model identifier to use when a role leaves it unspecified."""

    if (
        preferred_model
        and preferred_model not in {"auto", "none"}
        and not preferred_model.startswith("pre-")
    ):
        return preferred_model
    return os.environ.get("OPTILLM_MODEL", "gpt-4o-mini")


def _build_homogeneous_bon_config(model: str) -> List[Dict[str, Any]]:
    """Create a BON configuration where every role shares the same model."""

    return [
        {
            "name": role_name,
            "model": model,
            "reasoning_effort": DEFAULT_REASONING_EFFORT,
        }
        for role_name in BON_REQUIRED_ROLE_NAMES
    ]


def _resolve_bon_candidates(
    model: Optional[str], n: int
) -> Tuple[List[Dict[str, Any]], str]:
    """Resolve the BON configuration to use, falling back to homogeneous roles.

    Returns the full role list (always 6 entries) and the default model that
    should be used if a specific role omits a model.
    """

    config_name = None
    resolved_config = None

    if model:
        force_predefined = model.startswith("pre-")
        candidate_key = model[4:] if force_predefined else model
        resolved_config = load_complex_profile("bon", candidate_key)
        if resolved_config and not _validate_bon_profile(resolved_config):
            resolved_config = None

        if resolved_config:
            config_name = candidate_key
            logger.info("BON: Using complex profile '%s'", config_name)
        elif force_predefined:
            logger.warning(
                "BON: Requested complex profile '%s' not found; falling back to default roles",
                candidate_key,
            )

    if resolved_config:
        default_model = _determine_default_model(None)
        return resolved_config, default_model

    default_model = _determine_default_model(model)
    if n < BON_GENERATION_ROLE_COUNT:
        logger.warning(
            "BON: Requested n=%s is less than required %s generation roles; "
            "results will reuse role definitions.",
            n,
            BON_GENERATION_ROLE_COUNT,
        )

    homogeneous = _build_homogeneous_bon_config(default_model)
    if not model or model in {"auto", "none"}:
        logger.info(
            "BON: No model specified, using default OPTILLM_MODEL: %s",
            default_model,
        )
    else:
        logger.info(
            "BON: Using model '%s' for all %d BON roles",
            default_model,
            len(homogeneous),
        )
    return homogeneous, default_model


def best_of_n_sampling(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    n: int = 4,
    request_config: dict = None,
    request_id: str = None,
    use_synthesis: bool = False,
) -> str:
    """Best-of-N sampling with diverse temperatures and batch rating.

    Args:
        system_prompt: System prompt for the model
        initial_query: User query to process
        client: LLM client instance
        model: Model name
        n: Number of candidates to generate (default 4)
        request_config: Optional request configuration
        request_id: Optional request ID for logging
        use_synthesis: If True, synthesize top-2 instead of returning just #1

    Returns:
        Tuple of (best_response, total_tokens_used)
    """
    logger.info(f"Starting BON sampling with n={n}, model={model}")
    bon_completion_tokens = 0

    # Extract max_tokens from request_config with default
    max_tokens = 4096
    if request_config:
        max_tokens = request_config.get("max_tokens", max_tokens)

    bon_config, default_model = _resolve_bon_candidates(model, n)

    # Extract first 4 entries for candidate generation (config always has 6 total)
    candidates_for_generation = bon_config[:BON_GENERATION_ROLE_COUNT]

    completions = []
    candidate_names = []

    logger.debug(f"Generating {n} candidates with diverse models via OpenRouter")

    # Generate candidates with diverse models in parallel
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {}
        for i in range(n):
            candidate = candidates_for_generation[i % len(candidates_for_generation)]
            future = executor.submit(
                _generate_bon_candidate,
                client,
                default_model,
                system_prompt,
                initial_query,
                max_tokens,
                candidate,
                i,
            )
            futures[future] = (i, candidate["name"])

        for future in as_completed(futures):
            try:
                result = future.result()
                completions.append(result["content"])
                candidate_names.append(result["name"])

                if request_id:
                    conversation_logger.log_provider_call(
                        request_id, result["request"], result["response_dict"]
                    )

                bon_completion_tokens += result["tokens"]
                logger.info(
                    f"{result['name']}: Generated candidate {result['index'] + 1}/{n} (effort={result['effort']}). Tokens: {result['tokens']}"
                )

            except Exception as e:
                idx, name = futures[future]
                logger.error(
                    f"{name}: Error generating candidate {idx + 1}/{n}: {str(e)}"
                )
                continue

    if not completions:
        logger.error("Failed to generate any candidates")
        return "Error: Could not generate any candidates", 0

    if len(completions) < n:
        logger.warning(f"Only generated {len(completions)}/{n} candidates")

    # Batch rate all candidates using configurable QUALITY_RATER role
    logger.debug("Batch rating candidates")

    # Get QUALITY_RATER configuration
    quality_rater = _get_role_by_name(bon_config, "QUALITY_RATER", default_model)
    rating_model = quality_rater.get(
        "model", "openrouter/google/gemini-3-flash-preview"
    )
    rating_effort = quality_rater.get("reasoning_effort", "high")

    # Build the rating prompt with all candidates
    candidates_text = ""
    for i, (completion, name) in enumerate(zip(completions, candidate_names), 1):
        candidates_text += f"""
### Candidate {i} ({name}):
{completion}

"""

    rating_prompt = f"""You are a QUALITY RATER for a Best-of-N sampling system. Your task is to evaluate and rank candidate responses.

## Original Query
{initial_query}

## Candidates to Rate
{candidates_text}

## Rating Criteria

Rate each candidate on these dimensions (1-5 scale):
1. **Relevance**: How directly does it address the query?
2. **Coherence**: Is the reasoning logical and well-structured?
3. **Completeness**: Does it cover all aspects of the question?
4. **Clarity**: Is it easy to understand?

## Your Output Format

For each candidate, provide:
```
Candidate X: [SUMMARY SCORE]/20
  Relevance: [1-5] - [brief justification]
  Coherence: [1-5] - [brief justification]
  Completeness: [1-5] - [brief justification]
  Clarity: [1-5] - [brief justification]
```

Then provide your final ranking:
```
RANKING: [Candidate numbers in order, best first]
```

Be thorough and objective in your assessment."""

    # Route to appropriate client based on model prefix
    if rating_model.startswith("openrouter/"):
        rating_client = _get_litellm_client()
        provider = "OpenRouter (via LiteLLM)"
    else:
        rating_client = client
        provider = f"Direct ({type(client).__name__})"

    provider_request = {
        "model": rating_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rating_prompt},
        ],
        "max_tokens": 2048,
        "reasoning_effort": rating_effort,
    }

    logger.info(
        f"QUALITY_RATER: Using model='{rating_model}' via {provider}, "
        f"max_tokens=2048, reasoning_effort={rating_effort}"
    )

    rating_response = optillm.safe_completions_create(rating_client, provider_request)

    # Log provider call
    if request_id:
        response_dict = (
            rating_response.model_dump()
            if hasattr(rating_response, "model_dump")
            else rating_response
        )
        conversation_logger.log_provider_call(
            request_id, provider_request, response_dict
        )

    bon_completion_tokens += rating_response.usage.completion_tokens

    # Parse the rating response
    if (
        rating_response is None
        or not rating_response.choices
        or rating_response.choices[0].message.content is None
        or rating_response.choices[0].finish_reason == "length"
    ):
        logger.warning("Rating response truncated, using first candidate as default")
        best_index = 0
        rating_text = "Rating unavailable - using first candidate"
    else:
        rating_text = rating_response.choices[0].message.content
        best_index = _parse_best_candidate(rating_text, len(completions))

    logger.info(f"Batch rating complete. Selected candidate {best_index + 1}")
    logger.debug(f"Rating analysis:\n{rating_text}")

    # Return best candidate or synthesize top-K
    if use_synthesis and len(completions) >= 2:
        # Get indices of top 2 candidates
        top_indices = _parse_top_k_candidates(rating_text, len(completions), k=2)
        if len(top_indices) < 2:
            top_indices = [0, 1] if len(completions) >= 2 else [0]

        logger.debug(f"Synthesizing top {len(top_indices)} candidates")

        # Get FINAL_SYNTHESIZER configuration
        final_synthesizer = _get_role_by_name(
            bon_config, "FINAL_SYNTHESIZER", default_model
        )
        synthesis_model = final_synthesizer.get(
            "model", "openrouter/google/gemini-3-flash-preview"
        )
        synthesis_effort = final_synthesizer.get("reasoning_effort", "high")

        # Build synthesis prompt
        top_candidates_text = ""
        for rank, idx in enumerate(top_indices, 1):
            top_candidates_text += f"""
### Top Candidate {rank} ({candidate_names[idx]}, original rank {idx + 1}):
{completions[idx]}

"""

        synthesis_prompt = f"""You are a FINAL SYNTHESIZER for a Best-of-N sampling system. Your task is to create the optimal response by combining insights from the highest-rated candidates.

## Original Query
{initial_query}

## Top-Rated Candidates
{top_candidates_text}

## Rating Analysis
{rating_text}

## Your Task

Create a final response that:
1. **Incorporates the best elements** from each top candidate
2. **Addresses any weaknesses** identified in the ratings
3. **Maintains the strengths** that made these candidates top-rated
4. **Provides a clear, coherent answer** superior to any single candidate

Your synthesis should be better than the individual candidates - that's the advantage of Best-of-N with intelligent selection."""

        # Route to appropriate client based on model prefix
        if synthesis_model.startswith("openrouter/"):
            synthesis_client = _get_litellm_client()
            provider = "OpenRouter (via LiteLLM)"
        else:
            synthesis_client = client
            provider = f"Direct ({type(client).__name__})"

        provider_request = {
            "model": synthesis_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": synthesis_prompt},
            ],
            "max_tokens": max_tokens,
            "reasoning_effort": synthesis_effort,
        }

        logger.info(
            f"FINAL_SYNTHESIZER: Using model='{synthesis_model}' via {provider}, "
            f"max_tokens={max_tokens}, reasoning_effort={synthesis_effort}"
        )

        synthesis_response = optillm.safe_completions_create(
            synthesis_client, provider_request
        )

        # Log provider call
        if request_id:
            response_dict = (
                synthesis_response.model_dump()
                if hasattr(synthesis_response, "model_dump")
                else synthesis_response
            )
            conversation_logger.log_provider_call(
                request_id, provider_request, response_dict
            )

        bon_completion_tokens += synthesis_response.usage.completion_tokens

        if (
            synthesis_response is None
            or not synthesis_response.choices
            or synthesis_response.choices[0].message.content is None
            or synthesis_response.choices[0].finish_reason == "length"
        ):
            logger.warning("Synthesis failed, returning best candidate")
            result = completions[best_index]
        else:
            result = synthesis_response.choices[0].message.content
            logger.info("Used synthesis of top candidates")

    else:
        # Return the best candidate directly
        result = completions[best_index]

    logger.info(
        f"BON complete: {len(completions)} candidates, {bon_completion_tokens} total tokens"
    )
    return result, bon_completion_tokens


def _parse_best_candidate(rating_text: str, num_candidates: int) -> int:
    """Extract the best candidate index from rating response.

    Looks for patterns like:
    - "RANKING: 3, 1, 2, 4"
    - "Best: Candidate 2"
    - "Candidate 1: 18/20" (highest score)
    """
    import re

    # Try to find explicit ranking line
    ranking_match = re.search(r"RANKING:\s*([\d,\s]+)", rating_text, re.IGNORECASE)
    if ranking_match:
        ranks = [
            int(x.strip())
            for x in ranking_match.group(1).split(",")
            if x.strip().isdigit()
        ]
        if ranks:
            best = ranks[0] - 1  # Convert to 0-indexed
            if 0 <= best < num_candidates:
                return best

    # Try to find "Best: Candidate X"
    best_match = re.search(r"Best.*?Candidate\s+(\d+)", rating_text, re.IGNORECASE)
    if best_match:
        best = int(best_match.group(1)) - 1
        if 0 <= best < num_candidates:
            return best

    # Try to find highest score: "Candidate X: NN/20"
    scores = []
    pattern = r"Candidate\s+(\d+):[^0-9]*(\d+)\s*/\s*20"
    for match in re.finditer(pattern, rating_text, re.IGNORECASE):
        idx = int(match.group(1)) - 1
        score = int(match.group(2))
        if 0 <= idx < num_candidates:
            scores.append((score, idx))

    if scores:
        scores.sort(reverse=True)
        return scores[0][1]

    # Default to first candidate if parsing fails
    return 0


def _parse_top_k_candidates(rating_text: str, num_candidates: int, k: int = 2) -> list:
    """Extract top K candidate indices from rating response."""
    import re

    indices = []

    # Try explicit ranking
    ranking_match = re.search(r"RANKING:\s*([\d,\s]+)", rating_text, re.IGNORECASE)
    if ranking_match:
        ranks = [
            int(x.strip()) - 1
            for x in ranking_match.group(1).split(",")
            if x.strip().isdigit()
        ]
        for r in ranks:
            if 0 <= r < num_candidates and r not in indices:
                indices.append(r)
            if len(indices) >= k:
                return indices

    # Try to find scores and pick top K
    scores = []
    pattern = r"Candidate\s+(\d+):[^0-9]*(\d+)\s*/\s*20"
    for match in re.finditer(pattern, rating_text, re.IGNORECASE):
        idx = int(match.group(1)) - 1
        score = int(match.group(2))
        if 0 <= idx < num_candidates:
            scores.append((score, idx))

    if scores:
        scores.sort(reverse=True)
        for _, idx in scores:
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= k:
                return indices

    # Fallback to sequential candidates
    return list(range(min(k, num_candidates)))
