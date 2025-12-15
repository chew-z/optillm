# Version information
__version__ = "0.3.11"

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


def strip_unsupported_n(client, provider_request: dict) -> dict:
    """Return a sanitized copy of provider_request removing unsupported params for some providers.

    Currently strips 'n' for Z.ai clients which do not accept it.
    """
    sanitized = dict(provider_request)
    client_type = str(type(client)).lower()
    if "zai" in client_type:
        sanitized.pop("n", None)
    return sanitized


def safe_completions_create(client, provider_request: dict):
    """Call client's chat.completions.create with a fallback that retries without unsupported 'n'."""
    # Pre-sanitize for known providers
    client_type = str(type(client)).lower()
    req = dict(provider_request)
    if "zai" in client_type:
        # Remove unsupported 'n'
        req.pop("n", None)
        # Normalize model if prefixed like 'zai/glm-4.6'
        if "model" in req and isinstance(req["model"], str) and "/" in req["model"]:
            req["model"] = req["model"].split("/", 1)[1]

    try:
        return client.chat.completions.create(**req)
    except TypeError as e:
        msg = str(e)
        # If the provider doesn't support 'n', retry without it
        if "unexpected keyword argument 'n'" in msg or "n'" in msg:
            sanitized = strip_unsupported_n(client, provider_request)
            return client.chat.completions.create(**sanitized)
        raise
