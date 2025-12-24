import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        logger.info("MOA: Created LiteLLM client for OpenRouter models")
    return _litellm_client


# Agent personalities for MOA - distinct perspectives that catch different mistakes
MOA_AGENTS = [
    {
        "name": "explorer",
        "title": "EXPLORER",
        "system_suffix": (
            " You are an EXPLORER agent in a Mixture of Agents system.\n"
            "Your role: Think creatively, consider unconventional approaches, and generate diverse solutions.\n"
            "Embrace brainstorming and explore multiple angles even if they seem unusual at first."
        ),
        "temperature": 1.0,
        "model": "openrouter/google/gemini-3-flash-preview",
    },
    {
        "name": "analyst",
        "title": "ANALYST",
        "system_suffix": (
            " You are an ANALYST agent in a Mixture of Agents system.\n"
            "Your role: Be methodical, thorough, and systematic. Break down the problem step by step.\n"
            "Focus on logical structure, clear reasoning, and comprehensive coverage of the problem space."
        ),
        "temperature": 0.3,
        "model": "openrouter/openai/gpt-5.2-chat",
    },
    {
        "name": "critic",
        "title": "CRITIC",
        "system_suffix": (
            " You are a CRITIC agent in a Mixture of Agents system.\n"
            "Your role: Be skeptical and identify potential weaknesses, edge cases, and failure modes.\n"
            "Question assumptions and highlight what might go wrong or what's being overlooked."
        ),
        "temperature": 0.5,
        "model": "openrouter/anthropic/claude-haiku-4.5",
    },
]


def _generate_agent_response(client, default_model, system_prompt, initial_query, agent, max_tokens):
    """Generate response for a single MOA agent (thread-safe)."""
    agent_name = agent["title"]
    agent_system = system_prompt + agent["system_suffix"]

    # Use agent's specific model or default
    agent_model = agent.get("model", default_model)

    # Route to appropriate client based on model prefix
    # OpenRouter models go through LiteLLM, others use the default client (e.g., Z.ai)
    if agent_model.startswith("openrouter/"):
        agent_client = _get_litellm_client()
        provider = "OpenRouter (via LiteLLM)"
    else:
        agent_client = client
        provider = f"Direct ({type(client).__name__})"

    # Log the agent configuration with model and parameters
    logger.info(
        f"{agent_name}: Using model='{agent_model}' via {provider}, "
        f"temperature={agent['temperature']}, max_tokens={max_tokens}"
    )

    provider_request = {
        "model": agent_model,
        "messages": [
            {"role": "system", "content": agent_system},
            {"role": "user", "content": initial_query},
        ],
        "max_tokens": max_tokens,
        "temperature": agent["temperature"],
    }

    # Add reasoning effort for Gemini models via OpenRouter
    # LiteLLM translates reasoning_effort kwarg to OpenRouter's {"reasoning": {"effort": "high"}} format
    if agent_model.startswith("openrouter/google/") or agent_model.startswith(
        "openrouter/gemini-"
    ):
        provider_request["reasoning_effort"] = "high"
        logger.info(f"{agent_name}: Enabled reasoning_effort=high for Gemini model")

    response = optillm.safe_completions_create(agent_client, provider_request)

    return {
        "name": agent_name,
        "content": response.choices[0].message.content,
        "tokens": response.usage.completion_tokens,
        "response_dict": response.model_dump() if hasattr(response, "model_dump") else response,
        "request": provider_request,
        "temp": agent["temperature"],
    }


def mixture_of_agents(
    system_prompt: str,
    initial_query: str,
    client,
    model: str = None,
    request_config: dict = None,
    request_id: str = None,
) -> str:
    logger.info(f"Starting mixture_of_agents function with model: {model}")
    moa_completion_tokens = 0

    # Extract max_tokens from request_config with default
    # Increased default for comprehensive agent responses and synthesis
    max_tokens = 8192
    if request_config:
        max_tokens = request_config.get("max_tokens", max_tokens)

    completions = []
    agent_names = []

    logger.debug("Generating agent responses with distinct personalities")

    # Generate responses from each distinct agent in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_generate_agent_response, client, model, system_prompt,
                            initial_query, agent, max_tokens): agent
            for agent in MOA_AGENTS
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                agent_names.append(result["name"])
                completions.append(result["content"])

                if request_id:
                    conversation_logger.log_provider_call(
                        request_id, result["request"], result["response_dict"]
                    )

                moa_completion_tokens += result["tokens"]
                logger.info(f'{result["name"]}: Generated response (temp={result["temp"]}). Tokens: {result["tokens"]}')

            except Exception as e:
                agent = futures[future]
                logger.error(f"{agent['title']}: Error generating response: {str(e)}")
                continue

    if not completions:
        logger.error("Failed to generate any agent responses")
        return "Error: Could not generate any agent responses", 0

    if len(completions) < len(MOA_AGENTS):
        logger.warning(
            f"Only generated {len(completions)}/{len(MOA_AGENTS)} agent responses"
        )

    # Build critique prompt with agent labels
    critique_sections = []
    for i, (content, agent_name) in enumerate(zip(completions, agent_names)):
        critique_sections.append(
            f"""### {agent_name}'s Response:
{content}"""
        )

    critique_prompt = f"""You are a SYNTHESIZER agent in a Mixture of Agents system. Your task is to analyze and critique the responses from multiple specialized agents.

Original Query: {initial_query}

{''.join(critique_sections)}

## Your Analysis

For each agent's response above, provide:
1. **Strengths**: What aspects of this response are valuable, insightful, or well-reasoned?
2. **Weaknesses**: What is missing, flawed, or could be improved?
3. **Unique Contribution**: What distinct perspective or insight does this agent bring?

Then, provide a **Synthesis Assessment**:
- Which elements from each response should be incorporated into a final answer?
- What gaps remain that need to be addressed?
- What would the ideal combined response look like?

Be thorough and specific in your critique."""

    logger.debug("Generating agent critique")

    # Use LiteLLM client for SYNTHESIZER with Gemini via OpenRouter
    synthesizer_client = _get_litellm_client()
    synthesizer_model = "openrouter/google/gemini-3-flash-preview"

    provider_request = {
        "model": synthesizer_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": critique_prompt},
        ],
        "max_tokens": 8192,  # Increased for comprehensive critique
        "temperature": 0.3,
    }

    logger.info(
        f"SYNTHESIZER (critique): Using model='{synthesizer_model}' via OpenRouter (via LiteLLM), "
        f"max_tokens=8192, temperature=0.3"
    )

    critique_response = optillm.safe_completions_create(synthesizer_client, provider_request)

    # Convert response to dict for logging
    response_dict = (
        critique_response.model_dump()
        if hasattr(critique_response, "model_dump")
        else critique_response
    )

    # Log provider call if conversation logging is enabled
    if request_id:
        conversation_logger.log_provider_call(
            request_id, provider_request, response_dict
        )

    # Check for valid response
    if (
        critique_response is None
        or not critique_response.choices
        or critique_response.choices[0].message.content is None
        or critique_response.choices[0].finish_reason == "length"
    ):
        logger.warning("Critique response truncated or empty, using generic critique")
        critiques = f"Each of the {len(completions)} agent responses has been analyzed."
    else:
        critiques = critique_response.choices[0].message.content

    moa_completion_tokens += critique_response.usage.completion_tokens
    logger.info(
        f"Generated critiques. Tokens used: {critique_response.usage.completion_tokens}"
    )

    # Build final synthesis prompt
    final_sections = []
    for i, (content, agent_name) in enumerate(zip(completions, agent_names)):
        final_sections.append(f"**{agent_name}:**\n{content}")

    final_prompt = f"""You are the FINAL SYNTHESIZER in a Mixture of Agents system. Your task is to create the best possible response to the original query by synthesizing insights from multiple specialized agents.

## Original Query
{initial_query}

## Agent Responses
{''.join(final_sections)}

## Critique and Synthesis Guidance
{critiques}

## Your Task

Create a final response that:
1. **Synthesizes the best elements** from each agent's perspective
2. **Addresses the critiques** - incorporate the improvements identified
3. **Maintains the strengths** each agent brought (creativity, thoroughness, critical thinking)
4. **Fills any gaps** the critique identified
5. **Provides a clear, coherent answer** to the original query

Your response should be better than any single agent's response - that's the power of the Mixture of Agents approach."""

    logger.debug("Generating final synthesized response")

    # Use LiteLLM client for SYNTHESIZER with Gemini via OpenRouter
    # Reuse the same client and model from critique stage
    provider_request = {
        "model": synthesizer_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.5,
    }

    logger.info(
        f"SYNTHESIZER (final): Using model='{synthesizer_model}' via OpenRouter (via LiteLLM), "
        f"max_tokens={max_tokens}, temperature=0.5"
    )

    final_response = optillm.safe_completions_create(synthesizer_client, provider_request)

    # Convert response to dict for logging
    response_dict = (
        final_response.model_dump()
        if hasattr(final_response, "model_dump")
        else final_response
    )

    # Log provider call if conversation logging is enabled
    if request_id:
        conversation_logger.log_provider_call(
            request_id, provider_request, response_dict
        )

    moa_completion_tokens += final_response.usage.completion_tokens
    logger.info(
        f"Generated final response. Tokens used: {final_response.usage.completion_tokens}"
    )

    # Check for valid response
    if (
        final_response is None
        or not final_response.choices
        or final_response.choices[0].message.content is None
        or final_response.choices[0].finish_reason == "length"
    ):
        logger.error(
            "Final response truncated or empty. Consider increasing max_tokens."
        )
        # Return best completion if final response failed
        result = (
            completions[0]
            if completions
            else "Error: Response was truncated due to token limit. Please increase max_tokens or max_completion_tokens."
        )
    else:
        result = final_response.choices[0].message.content

    # Log snippet of the final result
    result_preview = result[:200].replace('\n', ' ') + '...' if len(result) > 200 else result.replace('\n', ' ')
    logger.info(f"Final result snippet: {result_preview}")

    logger.info(
        f"MOA complete: {len(completions)} agents, {moa_completion_tokens} total tokens"
    )
    return result, moa_completion_tokens
