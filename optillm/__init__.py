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
]


import logging
import os
import json

logger = logging.getLogger(__name__)


def strip_unsupported_n(client, provider_request: dict) -> dict:
    """Return a sanitized copy of provider_request removing unsupported params for some providers.

    Currently strips 'n' for Z.ai clients which do not accept it.
    """
    sanitized = dict(provider_request)
    client_type = str(type(client)).lower()
    if "zai" in client_type:
        sanitized.pop("n", None)
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

    - Strips provider prefixes like 'zai/...'
    - Applies optional alias mapping from env
    - Returns the normalized model
    """
    if not isinstance(model, str):
        return model

    normalized = model

    # If there's a slash, keep the rightmost segment (handles 'moa-zai/glm-4.6', 'zai/glm-4.6', etc.)
    if "/" in normalized:
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

    - Removes unsupported params (e.g., 'n' for Z.ai)
    - Normalizes model identifiers for the target provider
    - Retries without 'n' if a TypeError indicates it's unsupported
    """
    # Pre-sanitize for known providers
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

    if "zai" in client_type:
        # Z.ai requires 'n' via extra_body, not as top-level parameter
        if "n" in req:
            n_value = req.pop("n")
            extra_body = req.get("extra_body", {})
            if isinstance(extra_body, dict):
                extra_body["n"] = n_value
                req["extra_body"] = extra_body
                logger.debug(f"Moved 'n={n_value}' to extra_body for Z.ai provider")

    try:
        return client.chat.completions.create(**req)
    except TypeError as e:
        msg = str(e)
        # If the provider doesn't support 'n', retry without it
        if "unexpected keyword argument 'n'" in msg or "n'" in msg:
            sanitized = strip_unsupported_n(client, provider_request)
            # Ensure model normalization is preserved on retry
            if "model" in sanitized:
                sanitized["model"] = _normalize_model_for_provider(
                    sanitized["model"], client
                )
            return client.chat.completions.create(**sanitized)
        raise
