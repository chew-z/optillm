import logging
import optillm
from optillm import conversation_logger

logger = logging.getLogger(__name__)

# Diverse sampling strategies for BON - different temperatures for variety
BON_TEMPERATURES = [
    1.2,   # Highly exploratory - more creative, diverse outputs
    0.9,   # Balanced creative
    0.6,   # Balanced - good tradeoff
    0.3,   # Precise - focused, deterministic
]


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

    completions = []
    temperatures_used = []

    logger.debug(f"Generating {n} candidates with diverse temperatures")

    # Generate candidates with diverse temperatures
    for i in range(n):
        # Cycle through temperatures if n > predefined list
        temp = BON_TEMPERATURES[i % len(BON_TEMPERATURES)]
        temperatures_used.append(temp)

        try:
            provider_request = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_query},
                ],
                "max_tokens": max_tokens,
                "temperature": temp,
            }

            response = optillm.safe_completions_create(client, provider_request)

            # Log provider call
            if request_id:
                response_dict = (
                    response.model_dump() if hasattr(response, "model_dump") else response
                )
                conversation_logger.log_provider_call(
                    request_id, provider_request, response_dict
                )

            # Check for valid response
            if (
                response is None
                or not response.choices
                or response.choices[0].message.content is None
            ):
                logger.warning(f"Candidate {i + 1}/{n} (temp={temp}): Empty response, skipping")
                continue

            content = response.choices[0].message.content
            completions.append(content)
            bon_completion_tokens += response.usage.completion_tokens
            logger.info(
                f"Candidate {i + 1}/{n} (temp={temp}): Generated. Tokens: {response.usage.completion_tokens}"
            )

        except Exception as e:
            logger.error(f"Candidate {i + 1}/{n} (temp={temp}): Error - {str(e)}")
            continue

    if not completions:
        logger.error("Failed to generate any candidates")
        return "Error: Could not generate any candidates", 0

    if len(completions) < n:
        logger.warning(f"Only generated {len(completions)}/{n} candidates")

    # Batch rate all candidates in a single LLM call
    logger.debug("Batch rating candidates")

    # Build the rating prompt with all candidates
    candidates_text = ""
    for i, completion in enumerate(completions, 1):
        candidates_text += f"""
### Candidate {i} (temperature={temperatures_used[i-1]}):
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

    provider_request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": rating_prompt},
        ],
        "max_tokens": 2048,
        "n": 1,
        "temperature": 0.2,
    }

    rating_response = optillm.safe_completions_create(client, provider_request)

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

        # Build synthesis prompt
        top_candidates_text = ""
        for rank, idx in enumerate(top_indices, 1):
            top_candidates_text += f"""
### Top Candidate {rank} (temperature={temperatures_used[idx]}, original rank {idx + 1}):
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

        provider_request = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": synthesis_prompt},
            ],
            "max_tokens": max_tokens,
            "n": 1,
            "temperature": 0.5,
        }

        synthesis_response = optillm.safe_completions_create(client, provider_request)

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
    ranking_match = re.search(r'RANKING:\s*([\d,\s]+)', rating_text, re.IGNORECASE)
    if ranking_match:
        ranks = [int(x.strip()) for x in ranking_match.group(1).split(',') if x.strip().isdigit()]
        if ranks:
            best = ranks[0] - 1  # Convert to 0-indexed
            if 0 <= best < num_candidates:
                return best

    # Try to find "Best: Candidate X"
    best_match = re.search(r'Best.*?Candidate\s+(\d+)', rating_text, re.IGNORECASE)
    if best_match:
        best = int(best_match.group(1)) - 1
        if 0 <= best < num_candidates:
            return best

    # Try to find highest score: "Candidate X: NN/20"
    scores = []
    pattern = r'Candidate\s+(\d+):[^0-9]*(\d+)\s*/\s*20'
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
    ranking_match = re.search(r'RANKING:\s*([\d,\s]+)', rating_text, re.IGNORECASE)
    if ranking_match:
        ranks = [int(x.strip()) - 1 for x in ranking_match.group(1).split(',') if x.strip().isdigit()]
        for r in ranks:
            if 0 <= r < num_candidates and r not in indices:
                indices.append(r)
            if len(indices) >= k:
                return indices

    # Try to find scores and pick top K
    scores = []
    pattern = r'Candidate\s+(\d+):[^0-9]*(\d+)\s*/\s*20'
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
