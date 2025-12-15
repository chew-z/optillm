import logging
import optillm

logger = logging.getLogger(__name__)


def re2_approach(
    system_prompt,
    initial_query,
    client,
    model,
    n=1,
    request_config: dict = None,
    request_id: str = None,
):
    """
    Implement the RE2 (Re-Reading) approach for improved reasoning in LLMs.

    Args:
    system_prompt (str): The system prompt to be used.
    initial_query (str): The initial user query.
    client: The OpenAI client object.
    model (str): The name of the model to use.
    n (int): Number of completions to generate.
    request_config (dict): Optional configuration including max_tokens.

    Returns:
    str or list: The generated response(s) from the model.
    """
    logger.info("Using RE2 approach for query processing")
    re2_completion_tokens = 0

    # Extract max_tokens from request_config if provided
    max_tokens = None
    if request_config:
        max_tokens = request_config.get("max_tokens")

    # Construct the RE2 prompt
    re2_prompt = f"{initial_query}\nRead the question again: {initial_query}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": re2_prompt},
    ]

    try:
        provider_request_base = {"model": model, "messages": messages}
        if max_tokens is not None:
            provider_request_base["max_tokens"] = max_tokens

        client_type = str(type(client)).lower()
        is_zai = "zai" in client_type

        if is_zai:
            # Z.ai doesn't support 'n' — generate n completions one-by-one and aggregate
            responses = []
            for i in range(max(1, n)):
                resp = client.chat.completions.create(**provider_request_base)
                responses.append(resp)

                # Log each provider call if conversation logging is enabled
                if (
                    hasattr(optillm, "conversation_logger")
                    and optillm.conversation_logger
                    and request_id
                ):
                    response_dict = (
                        resp.model_dump() if hasattr(resp, "model_dump") else resp
                    )
                    optillm.conversation_logger.log_provider_call(
                        request_id, provider_request_base, response_dict
                    )

            # Aggregate tokens and contents
            for resp in responses:
                if hasattr(resp, "usage") and hasattr(resp.usage, "completion_tokens"):
                    re2_completion_tokens += resp.usage.completion_tokens

            contents = [
                choice.message.content.strip()
                for resp in responses
                for choice in getattr(resp, "choices", [])
                if choice.message.content is not None
            ]

            if n == 1:
                return (contents[0] if contents else ""), re2_completion_tokens
            return contents, re2_completion_tokens
        else:
            provider_request = {**provider_request_base, "n": n}
            response = client.chat.completions.create(**provider_request)

            # Log provider call
            if (
                hasattr(optillm, "conversation_logger")
                and optillm.conversation_logger
                and request_id
            ):
                response_dict = (
                    response.model_dump()
                    if hasattr(response, "model_dump")
                    else response
                )
                optillm.conversation_logger.log_provider_call(
                    request_id, provider_request, response_dict
                )

            re2_completion_tokens += response.usage.completion_tokens
            if n == 1:
                return (
                    response.choices[0].message.content.strip(),
                    re2_completion_tokens,
                )
            else:
                return [
                    choice.message.content.strip() for choice in response.choices
                ], re2_completion_tokens

    except Exception as e:
        logger.error(f"Error in RE2 approach: {str(e)}")
        raise
