import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import optillm
from optillm import conversation_logger
from optillm.config_loader import load_complex_profile

logger = logging.getLogger(__name__)

# MOA requires five role entries per configuration:
#   - First three generate candidate reasoning traces
#   - SYNTHESIZER critiques the agents
#   - FINAL_SYNTHESIZER produces the final response
MOA_REQUIRED_ROLE_NAMES = [
    "CREATIVE",
    "ANALYST",
    "CRITIC",
    "SYNTHESIZER",
    "FINAL_SYNTHESIZER",
]
MOA_GENERATION_ROLE_COUNT = 3
DEFAULT_REASONING_EFFORT = "high"

_ROLE_TEMPLATES = [
    {
        "name": "creative",
        "title": "CREATIVE",
        "system_suffix": (
            " You are a CREATIVE agent in a Mixture of Agents system.\n"
            "Your role: Think creatively, consider unconventional approaches, and generate diverse solutions.\n"
            "Embrace brainstorming and explore multiple angles even if they seem unusual at first."
        ),
    },
    {
        "name": "analyst",
        "title": "ANALYST",
        "system_suffix": (
            " You are an ANALYST agent in a Mixture of Agents system.\n"
            "Your role: Be methodical, thorough, and systematic. Break down the problem step by step.\n"
            "Focus on logical structure, clear reasoning, and comprehensive coverage of the problem space."
        ),
    },
    {
        "name": "critic",
        "title": "CRITIC",
        "system_suffix": (
            " You are a CRITIC agent in a Mixture of Agents system.\n"
            "Your role: Be skeptical and identify potential weaknesses, edge cases, and failure modes.\n"
            "Question assumptions and highlight what might go wrong or what's being overlooked."
        ),
    },
    {
        "name": "synthesizer",
        "title": "SYNTHESIZER",
        "system_suffix": (
            " You are a SYNTHESIZER agent in a Mixture of Agents system.\n"
            "Your role: Analyze and critique the responses from multiple specialized agents.\n"
            "Provide strengths, weaknesses, and synthesis guidance."
        ),
    },
    {
        "name": "final_synthesizer",
        "title": "FINAL_SYNTHESIZER",
        "system_suffix": (
            " You are the FINAL_SYNTHESIZER in a Mixture of Agents system.\n"
            "Your role: Create the best possible response by synthesizing insights from multiple specialized agents.\n"
            "Combine the best elements and address the critiques."
        ),
    },
]

_litellm_client = None


def _get_litellm_client():
    """Get or create the LiteLLM client for OpenRouter models (thread-safe singleton)."""
    global _litellm_client
    if _litellm_client is None:
        from optillm.litellm_wrapper import LiteLLMWrapper

        _litellm_client = LiteLLMWrapper()
        logger.info("MOA: Created LiteLLM client for OpenRouter models")
    return _litellm_client


def _validate_moa_profile(config: List[Dict[str, Any]]) -> bool:
    """Ensure a MOA profile defines all required roles."""

    if len(config) < len(MOA_REQUIRED_ROLE_NAMES):
        logger.error(
            "MOA profile has %s roles but requires %s (%s)",
            len(config),
            len(MOA_REQUIRED_ROLE_NAMES),
            ", ".join(MOA_REQUIRED_ROLE_NAMES),
        )
        return False

    defined_titles = {role.get("title") for role in config if isinstance(role, dict)}
    missing = [name for name in MOA_REQUIRED_ROLE_NAMES if name not in defined_titles]
    if missing:
        logger.error(
            "MOA profile is missing required roles: %s",
            ", ".join(missing),
        )
        return False

    return True


def _get_role_by_name(
    config: List[Dict[str, Any]],
    role_name: str,
    default_model: str = "openrouter/google/gemini-3-flash-preview",
) -> Dict[str, Any]:
    """Find a role configuration by name from the config list.

    Args:
        config: Full configuration list
        role_name: Name/title of the role to find (e.g., 'SYNTHESIZER', 'FINAL_SYNTHESIZER')
        default_model: Default model if role not found

    Returns:
        Role configuration dict with model, reasoning_effort, etc.
    """
    for role in config:
        # Check both 'name' and 'title' fields for flexibility
        if role.get("name") == role_name or role.get("title") == role_name:
            return role
    # Fallback to default model
    logger.warning(
        f"Role {role_name} not found in config, using default model {default_model}"
    )
    return {
        "name": role_name.lower(),
        "title": role_name,
        "model": default_model,
        "reasoning_effort": "high",
        "system_suffix": "",
    }


def _determine_default_model(preferred_model: Optional[str]) -> str:
    if (
        preferred_model
        and preferred_model not in {"auto", "none"}
        and not preferred_model.startswith("pre-")
    ):
        return preferred_model
    return os.environ.get("OPTILLM_MODEL", "gpt-4o-mini")


def _build_homogeneous_moa_config(model: str) -> List[Dict[str, Any]]:
    config: List[Dict[str, Any]] = []
    for template in _ROLE_TEMPLATES:
        config.append(
            {
                **template,
                "model": model,
                "reasoning_effort": template.get(
                    "reasoning_effort", DEFAULT_REASONING_EFFORT
                ),
            }
        )
    return config


def _resolve_moa_config(
    model: Optional[str],
) -> Tuple[List[Dict[str, Any]], str]:
    """Return the MOA role configuration and fallback model."""

    resolved_config = None

    if model:
        force_predefined = model.startswith("pre-")
        candidate_key = model[4:] if force_predefined else model
        resolved_config = load_complex_profile("moa", candidate_key)
        if resolved_config and not _validate_moa_profile(resolved_config):
            resolved_config = None

        if resolved_config:
            logger.info("MOA: Using complex profile '%s'", candidate_key)
        elif force_predefined:
            logger.warning(
                "MOA: Requested complex profile '%s' not found; falling back to default roles",
                candidate_key,
            )

    if resolved_config:
        default_model = _determine_default_model(None)
        return resolved_config, default_model

    default_model = _determine_default_model(model)
    if not model or model in {"auto", "none"}:
        logger.info(
            "MOA: No model specified, using default OPTILLM_MODEL: %s",
            default_model,
        )
    else:
        logger.info(
            "MOA: Using model '%s' for all MOA roles",
            default_model,
        )

    return _build_homogeneous_moa_config(default_model), default_model


def _generate_agent_response(
    client, default_model, system_prompt, initial_query, agent, max_tokens
):
    """Generate response for a single MOA agent (thread-safe)."""
    agent_name = agent["title"]
    agent_system = system_prompt + agent["system_suffix"]

    # Use agent's specific model or default
    agent_model = agent.get("model", default_model)
    agent_effort = agent.get("reasoning_effort", "high")

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
        f"reasoning_effort={agent_effort}, max_tokens={max_tokens}"
    )

    provider_request = {
        "model": agent_model,
        "messages": [
            {"role": "system", "content": agent_system},
            {"role": "user", "content": initial_query},
        ],
        "max_tokens": max_tokens,
        "reasoning_effort": agent_effort,
    }

    response = optillm.safe_completions_create(agent_client, provider_request)

    return {
        "name": agent_name,
        "content": response.choices[0].message.content,
        "tokens": response.usage.completion_tokens,
        "response_dict": (
            response.model_dump() if hasattr(response, "model_dump") else response
        ),
        "request": provider_request,
        "effort": agent_effort,
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

    moa_config, default_model = _resolve_moa_config(model)

    # Extract first 3 entries for agent generation (config always has 5 total)
    agents_for_generation = moa_config[:MOA_GENERATION_ROLE_COUNT]

    completions = []
    agent_names = []

    logger.debug("Generating agent responses with distinct personalities")

    # Generate responses from each distinct agent in parallel
    with ThreadPoolExecutor(max_workers=MOA_GENERATION_ROLE_COUNT) as executor:
        futures = {
            executor.submit(
                _generate_agent_response,
                client,
                default_model,
                system_prompt,
                initial_query,
                agent,
                max_tokens,
            ): agent
            for agent in agents_for_generation
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
                logger.info(
                    f"{result['name']}: Generated response (effort={result['effort']}). Tokens: {result['tokens']}"
                )

            except Exception as e:
                agent = futures[future]
                logger.error(f"{agent['title']}: Error generating response: {str(e)}")
                continue

    if not completions:
        logger.error("Failed to generate any agent responses")
        return "Error: Could not generate any agent responses", 0

    if len(completions) < len(agents_for_generation):
        logger.warning(
            f"Only generated {len(completions)}/{len(agents_for_generation)} agent responses"
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

{"".join(critique_sections)}

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

    # Get SYNTHESIZER configuration
    synthesizer = _get_role_by_name(moa_config, "SYNTHESIZER", default_model)
    synthesizer_model = synthesizer.get(
        "model", "openrouter/google/gemini-3-flash-preview"
    )
    synthesizer_effort = synthesizer.get("reasoning_effort", "high")
    synthesizer_suffix = synthesizer.get("system_suffix", "")

    # Add system suffix if configured
    if synthesizer_suffix:
        system_prompt_with_suffix = system_prompt + synthesizer_suffix
    else:
        system_prompt_with_suffix = system_prompt

    # Route to appropriate client based on model prefix
    if synthesizer_model.startswith("openrouter/"):
        synthesizer_client = _get_litellm_client()
        provider = "OpenRouter (via LiteLLM)"
    else:
        synthesizer_client = client
        provider = f"Direct ({type(client).__name__})"

    provider_request = {
        "model": synthesizer_model,
        "messages": [
            {"role": "system", "content": system_prompt_with_suffix},
            {"role": "user", "content": critique_prompt},
        ],
        "max_tokens": 8192,  # Increased for comprehensive critique
        "reasoning_effort": synthesizer_effort,
    }

    logger.info(
        f"SYNTHESIZER (critique): Using model='{synthesizer_model}' via {provider}, "
        f"max_tokens=8192, reasoning_effort={synthesizer_effort}"
    )

    critique_response = optillm.safe_completions_create(
        synthesizer_client, provider_request
    )

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
{"".join(final_sections)}

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

    # Get FINAL_SYNTHESIZER configuration
    final_synthesizer = _get_role_by_name(
        moa_config, "FINAL_SYNTHESIZER", default_model
    )
    final_model = final_synthesizer.get(
        "model", "openrouter/google/gemini-3-flash-preview"
    )
    final_effort = final_synthesizer.get("reasoning_effort", "high")
    final_suffix = final_synthesizer.get("system_suffix", "")

    # Add system suffix if configured
    if final_suffix:
        system_prompt_final = system_prompt + final_suffix
    else:
        system_prompt_final = system_prompt

    # Route to appropriate client based on model prefix
    if final_model.startswith("openrouter/"):
        final_client = _get_litellm_client()
        provider = "OpenRouter (via LiteLLM)"
    else:
        final_client = client
        provider = f"Direct ({type(client).__name__})"

    provider_request = {
        "model": final_model,
        "messages": [
            {"role": "system", "content": system_prompt_final},
            {"role": "user", "content": final_prompt},
        ],
        "max_tokens": max_tokens,
        "reasoning_effort": final_effort,
    }

    logger.info(
        f"FINAL_SYNTHESIZER: Using model='{final_model}' via {provider}, "
        f"max_tokens={max_tokens}, reasoning_effort={final_effort}"
    )

    final_response = optillm.safe_completions_create(final_client, provider_request)

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
    result_preview = (
        result[:200].replace("\n", " ") + "..."
        if len(result) > 200
        else result.replace("\n", " ")
    )
    logger.info(f"Final result snippet: {result_preview}")

    logger.info(
        f"MOA complete: {len(completions)} agents, {moa_completion_tokens} total tokens"
    )
    return result, moa_completion_tokens
