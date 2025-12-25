# Version information
__version__ = "0.3.12"

# Import from server module
from .server import (
    main,
    server_config,
    app,
    known_approaches,
    plugin_approaches,
    parse_combined_approach,
    parse_conversation,
    extract_optillm_approach,
    get_config,
    load_plugins,
    count_reasoning_tokens,
    parse_args,
    execute_single_approach,
    execute_combined_approaches,
    execute_parallel_approaches,
    generate_streaming_response,
)

# List of exported symbols
__all__ = [
    "main",
    "server_config",
    "app",
    "known_approaches",
    "plugin_approaches",
    "parse_combined_approach",
    "parse_conversation",
    "extract_optillm_approach",
    "get_config",
    "load_plugins",
    "count_reasoning_tokens",
    "parse_args",
    "execute_single_approach",
    "execute_combined_approaches",
    "execute_parallel_approaches",
    "generate_streaming_response",
    "strip_unsupported_params",
    "safe_completions_create",
]


import logging
import os
import json

logger = logging.getLogger(__name__)


def strip_unsupported_params(client, provider_request: dict) -> dict:
    """Return a sanitized copy of provider_request keeping only supported parameters.

    Uses allowlists to filter parameters based on provider capabilities:
    - Z.ai GLM-4.7: Limited set from Z.ai API (temperature, top_p, max_tokens, etc.)
    - Cerebras: Limited set from Cerebras SDK
    - OpenAI/Azure: Standard OpenAI API parameters
    - LiteLLM: Handles filtering automatically, pass everything through
    - InferenceClient: Local inference with extended parameters
    """
    client_type = str(type(client)).lower()

    # Standard OpenAI Chat Completions API parameters
    # Reference: https://platform.openai.com/docs/api-reference/chat/create
    OPENAI_STANDARD_PARAMS = {
        "model",
        "messages",
        "temperature",
        "top_p",
        "n",
        "stream",
        "stop",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "user",
        "seed",
        "tools",
        "tool_choice",
        "response_format",
        "logprobs",
        "top_logprobs",
    }

    # Z.ai GLM-4.7 supported parameters
    # Reference: https://docs.z.ai/guides/overview/migrate-to-glm-new.md
    ZAI_GLM_PARAMS = {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "stream",
        "tools",
        "tool_choice",
        "thinking",
        "tool_stream",
        "stop",
        "user",
        "extra_body",  # extra_body for 'n' parameter
    }

    # Cerebras Cloud SDK supported parameters
    # Reference: https://inference-docs.cerebras.ai/api-reference/chat-completions
    CEREBRAS_PARAMS = {
        "model",
        "messages",
        "temperature",
        "top_p",
        "max_tokens",
        "stream",
        "stop",
        "seed",
        "response_format",
    }

    # LiteLLM wrapper - passes through all parameters, handles filtering internally
    if "litellm" in client_type:
        logger.debug("LiteLLM client detected - passing all parameters through")
        return dict(provider_request)

    # InferenceClient (local) - supports extended parameters for decoding
    elif "inferenceclient" in client_type:
        logger.debug("InferenceClient detected - passing all parameters through")
        return dict(provider_request)

    # Z.ai client - use GLM-4.7 allowlist
    elif "zai" in client_type:
        sanitized = {}
        removed_params = []

        for key, value in provider_request.items():
            if key in ZAI_GLM_PARAMS:
                sanitized[key] = value
            else:
                removed_params.append(key)

        if removed_params:
            logger.debug(f"Z.ai: Filtered out unsupported parameters: {removed_params}")

        return sanitized

    # Cerebras client - use Cerebras allowlist
    elif "cerebras" in client_type:
        sanitized = {}
        removed_params = []

        for key, value in provider_request.items():
            if key in CEREBRAS_PARAMS:
                sanitized[key] = value
            else:
                removed_params.append(key)

        if removed_params:
            logger.debug(
                f"Cerebras: Filtered out unsupported parameters: {removed_params}"
            )

        return sanitized

    # OpenAI/Azure - use standard OpenAI allowlist
    elif "openai" in client_type or "azure" in client_type:
        sanitized = {}
        removed_params = []

        for key, value in provider_request.items():
            if key in OPENAI_STANDARD_PARAMS:
                sanitized[key] = value
            else:
                removed_params.append(key)

        if removed_params:
            logger.debug(
                f"OpenAI/Azure: Filtered out non-standard parameters: {removed_params}"
            )

        return sanitized

    # Unknown client - use conservative OpenAI standard params
    else:
        logger.warning(
            f"Unknown client type: {client_type}. Using OpenAI standard parameters."
        )
        sanitized = {}

        for key, value in provider_request.items():
            if key in OPENAI_STANDARD_PARAMS:
                sanitized[key] = value

        return sanitized


def _load_model_aliases(env_var_names=("OPTILLM_MODEL_ALIASES", "ZAI_MODEL_ALIASES")):
    """Load optional model alias mappings from environment.

    Supports either JSON (e.g., '{"zai/glm-4.6":"glm-4-9b-chat"}') or
    comma-separated pairs of the form 'from=to,from2=to2'.
    """
    mapping = {}
    for env_name in env_var_names:
        raw = os.environ.get(env_name)
        if not raw:
            continue
        try:
            # Try JSON first
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                mapping.update({str(k): str(v) for k, v in parsed.items()})
                continue
        except Exception:
            pass
        # Fallback to comma-separated pairs
        try:
            pairs = [p.strip() for p in raw.split(",") if p.strip()]
            for p in pairs:
                if "=" in p:
                    src, dst = p.split("=", 1)
                    mapping[src.strip()] = dst.strip()
        except Exception:
            # Ignore malformed alias env value
            logger.debug("Failed to parse model alias env %s", env_name)
    return mapping


def _normalize_model_for_provider(model: str, client) -> str:
    """Normalize a model string before sending to a provider.

    - Preserves LiteLLM provider prefixes like 'openrouter/...', 'anthropic/...'
    - Strips simple provider prefixes like 'zai/...'
    - Applies optional alias mapping from env
    - Returns the normalized model
    """
    if not isinstance(model, str):
        return model

    normalized = model

    # Preserve multi-part provider prefixes for LiteLLM (e.g., 'openrouter/google/gemini-2.5-flash')
    # These should be passed through as-is since LiteLLM handles the routing
    if (
        normalized.startswith("openrouter/")
        or normalized.startswith("anthropic/")
        or normalized.startswith("cohere/")
    ):
        # Keep the full model string for LiteLLM routing
        pass
    # If there's a slash, keep the rightmost segment (handles 'moa-zai/glm-4.6', 'zai/glm-4.6', etc.)
    elif "/" in normalized:
        normalized = normalized.rsplit("/", 1)[-1]

    # Apply optional alias mapping
    aliases = _load_model_aliases()
    if normalized in aliases:
        logger.info(
            "Remapping model '%s' to provider model '%s' via aliases",
            normalized,
            aliases[normalized],
        )
        normalized = aliases[normalized]

    # Extra: If client is Z.ai, allow alias keys with 'zai/...' too
    client_type = str(type(client)).lower()
    if "zai" in client_type:
        # Check full original string against aliases as well
        full = model
        if full in aliases:
            logger.info(
                "Remapping model '%s' to provider model '%s' via aliases",
                full,
                aliases[full],
            )
            normalized = aliases[full]

    return normalized


def safe_completions_create(client, provider_request: dict):
    """Call client's chat.completions.create with provider-aware sanitization and fallbacks.

    - Removes unsupported params via allowlist filtering
    - Normalizes model identifiers for the target provider
    - Handles provider-specific quirks (e.g., Z.ai 'n' in extra_body)
    - Retries with sanitized parameters if a TypeError indicates unsupported params
    """
    client_type = str(type(client)).lower()
    req = dict(provider_request)

    # Normalize model for the target provider
    if "model" in req:
        before = req["model"]
        req["model"] = _normalize_model_for_provider(req["model"], client)
        if before != req["model"]:
            logger.info(
                "Normalized model '%s' -> '%s' for provider %s",
                before,
                req["model"],
                client_type,
            )

    # Only OpenAI/Azure/LiteLLM should ever see `n`; others should ignore it
    supports_n = (
        "openai" in client_type or "azure" in client_type or "litellm" in client_type
    )
    if not supports_n and "n" in req:
        dropped = req.pop("n")
        logger.debug(
            "Dropping unsupported 'n'=%s for client type %s (handled via fan-out in approaches)",
            dropped,
            client_type,
        )

    # Apply allowlist filtering to remove unsupported parameters
    req = strip_unsupported_params(client, req)

    try:
        return client.chat.completions.create(**req)
    except TypeError as e:
        msg = str(e)
        # If the provider doesn't support some parameter, log and retry
        if "unexpected keyword argument" in msg:
            logger.warning(
                f"Parameter error from provider: {msg}. Retrying with sanitized request."
            )
            # Retry with fresh sanitization from original request
            sanitized = dict(provider_request)

            # Drop n again for non-OpenAI/Azure/LiteLLM providers
            if not supports_n and "n" in sanitized:
                sanitized.pop("n")

            # Normalize model on retry
            if "model" in sanitized:
                sanitized["model"] = _normalize_model_for_provider(
                    sanitized["model"], client
                )

            # Filter parameters
            sanitized = strip_unsupported_params(client, sanitized)

            return client.chat.completions.create(**sanitized)
        raise
