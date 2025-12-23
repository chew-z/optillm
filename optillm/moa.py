import logging
import optillm
from optillm import conversation_logger

logger = logging.getLogger(__name__)

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
    },
]


def mixture_of_agents(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    request_config: dict = None,
    request_id: str = None,
) -> str:
    logger.info(f"Starting mixture_of_agents function with model: {model}")
    moa_completion_tokens = 0

    # Extract max_tokens from request_config with default
    max_tokens = 4096
    if request_config:
        max_tokens = request_config.get("max_tokens", max_tokens)

    completions = []
    agent_names = []

    logger.debug("Generating agent responses with distinct personalities")

    # Generate responses from each distinct agent
    for agent in MOA_AGENTS:
        agent_name = agent["title"]
        agent_names.append(agent_name)
        agent_system = system_prompt + agent["system_suffix"]

        try:
            provider_request = {
                "model": model,
                "messages": [
                    {"role": "system", "content": agent_system},
                    {"role": "user", "content": initial_query},
                ],
                "max_tokens": max_tokens,
                "temperature": agent["temperature"],
            }

            response = optillm.safe_completions_create(client, provider_request)

            # Convert response to dict for logging
            response_dict = (
                response.model_dump() if hasattr(response, "model_dump") else response
            )

            # Log provider call if conversation logging is enabled
            if request_id:
                conversation_logger.log_provider_call(
                    request_id, provider_request, response_dict
                )

            # Check for valid response
            if (
                response is None
                or not response.choices
                or response.choices[0].message.content is None
            ):
                logger.warning(
                    f"{agent_name}: Response was empty or invalid, skipping"
                )
                continue

            content = response.choices[0].message.content
            completions.append(content)
            moa_completion_tokens += response.usage.completion_tokens
            logger.info(
                f"{agent_name}: Generated response (temp={agent['temperature']}). Tokens: {response.usage.completion_tokens}"
            )

        except Exception as e:
            logger.error(f"{agent_name}: Error generating response: {str(e)}")
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

    provider_request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": critique_prompt},
        ],
        "max_tokens": 2048,  # Increased for meaningful critique
        "n": 1,
        "temperature": 0.3,
    }

    critique_response = optillm.safe_completions_create(client, provider_request)

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

    provider_request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_prompt},
        ],
        "max_tokens": max_tokens,
        "n": 1,
        "temperature": 0.5,
    }

    final_response = optillm.safe_completions_create(client, provider_request)

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

    logger.info(
        f"MOA complete: {len(completions)} agents, {moa_completion_tokens} total tokens"
    )
    return result, moa_completion_tokens
