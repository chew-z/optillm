import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

import optillm
from optillm import conversation_logger

logger = logging.getLogger(__name__)

# Lazy-loaded LiteLLM client for OpenRouter models
_litellm_client = None


def _get_litellm_client():
    """Get or create the LiteLLM client for OpenRouter models (thread-safe singleton)."""
    global _litellm_client
    if _litellm_client is None:
        from optillm.litellm_wrapper import LiteLLMWrapper

        _litellm_client = LiteLLMWrapper()
        logger.info("BON: Created LiteLLM client for OpenRouter models")
    return _litellm_client


def _load_config_from_env(
    approach: str, config_name: str
) -> Optional[List[Dict[str, Any]]]:
    """Load model configuration from environment variable.

    Args:
        approach: Either 'bon' or 'moa'
        config_name: Configuration name (e.g., 'rapid', 'deep', 'coding')

    Returns:
        List of candidate/agent configurations, or None if not found
    """
    env_var = f"OPTILLM_{approach.upper()}_CONFIG_{config_name.upper()}"
    config_json = os.environ.get(env_var)

    if not config_json:
        return None

    try:
        return json.loads(config_json)
    except json.JSONDecodeError:
        logger.error(f"Failed to parse {env_var} as JSON")
        return None


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


# Fallback candidates for BON when no configuration is provided
# Uses the "deep" configuration with diverse models and explanatory role names
BON_CANDIDATES = [
    {
        "name": "CREATIVE",
        "model": "openrouter/google/gemini-3-flash-preview",
        "reasoning_effort": "high",
    },
    {
        "name": "ANALYST",
        "model": "openrouter/openai/gpt-5.2-chat",
        "reasoning_effort": "high",
    },
    {
        "name": "CRITIC",
        "model": "openrouter/anthropic/claude-haiku-4.5",
        "reasoning_effort": "high",
    },
    {
        "name": "SYNTHESIZER",
        "model": "openrouter/google/gemini-3-flash-preview",
        "reasoning_effort": "high",
    },
    {
        "name": "QUALITY_RATER",
        "model": "openrouter/google/gemini-3-flash-preview",
        "reasoning_effort": "high",
    },
    {
        "name": "FINAL_SYNTHESIZER",
        "model": "openrouter/google/gemini-3-flash-preview",
        "reasoning_effort": "high",
    },
]


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

    # Determine configuration from model parameter
    # Note: parse_combined_approach() already stripped the "bon-" prefix
    # So model="rapid" for "bon-rapid", or model="pre-rapid" for "bon-pre-rapid"
    config_name = None

    # Check for "pre-" prefix indicating predefined config
    if model.startswith("pre-"):
        config_name = model[4:]  # Extract config name after "pre-"
    else:
        # Check if model name itself is a known config name
        candidates = _load_config_from_env("bon", model)
        if candidates:
            config_name = model

    # Use configuration if found, otherwise use single model for all candidates
    if config_name:
        BON_CANDIDATES_LOCAL = _load_config_from_env("bon", config_name)
        logger.info(f"BON: Using predefined config '{config_name}'")
    else:
        # Use the specified model, or fall back to default if not provided/invalid
        if not model or model in ["auto", "none"]:
            model = os.environ.get("OPTILLM_MODEL", "gpt-4o-mini")
            logger.info(
                f"BON: No model specified, using default OPTILLM_MODEL: {model}"
            )

        # Create homogeneous candidates using the specified model
        BON_CANDIDATES_LOCAL = [
            {
                "name": f"CANDIDATE_{i}",
                "model": model,
                "reasoning_effort": "high",
            }
            for i in range(n)
        ]
        logger.info(f"BON: Using model '{model}' for all {n} candidates")

    # Extract first 4 entries for candidate generation (config may have 6 total)
    candidates_for_generation = BON_CANDIDATES_LOCAL[:4]

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
                model,
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
                    f'{result["name"]}: Generated candidate {result["index"] + 1}/{n} (effort={result["effort"]}). Tokens: {result["tokens"]}'
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
    quality_rater = _get_role_by_name(BON_CANDIDATES_LOCAL, "QUALITY_RATER")
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
        final_synthesizer = _get_role_by_name(BON_CANDIDATES_LOCAL, "FINAL_SYNTHESIZER")
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
