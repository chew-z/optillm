"""Helpers for loading BON/MOA complex-model profiles from JSON files."""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

APPROACHES = {"bon", "moa"}
CONFIG_ENV_VARS = {
    "bon": "OPTILLM_BON_CONFIG_FILE",
    "moa": "OPTILLM_MOA_CONFIG_FILE",
}
_DEFAULT_FILES = {
    "bon": os.path.join(os.path.dirname(__file__), "config", "bon_profiles.json"),
    "moa": os.path.join(os.path.dirname(__file__), "config", "moa_profiles.json"),
}

_PROFILE_CACHE: Dict[Tuple[str, str], Dict[str, List[Dict[str, Any]]]] = {}


def _resolve_config_path(approach: str) -> str:
    if approach not in APPROACHES:
        raise ValueError(f"Unsupported approach for config loading: {approach}")

    env_var = CONFIG_ENV_VARS[approach]
    override_path = os.environ.get(env_var)
    if override_path:
        return os.path.abspath(override_path)

    return os.path.abspath(_DEFAULT_FILES[approach])


def _load_profiles_from_path(
    approach: str, path: str
) -> Dict[str, List[Dict[str, Any]]]:
    cache_key = (approach, path)
    if cache_key in _PROFILE_CACHE:
        return _PROFILE_CACHE[cache_key]

    profiles: Dict[str, List[Dict[str, Any]]] = {}

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        logger.warning(
            "%s complex-model config file not found at %s", approach.upper(), path
        )
    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse %s complex-model config file %s: %s",
            approach.upper(),
            path,
            exc,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "Unexpected error reading %s complex-model config file %s: %s",
            approach.upper(),
            path,
            exc,
        )
    else:
        if not isinstance(data, dict):
            logger.error(
                "%s complex-model config file must map profile names to role arrays: %s",
                approach.upper(),
                path,
            )
        else:
            for name, role_list in data.items():
                if isinstance(role_list, list):
                    profiles[name] = role_list
                else:
                    logger.warning(
                        "Skipping %s profile '%s' because its value is not a list",
                        approach.upper(),
                        name,
                    )

    _PROFILE_CACHE[cache_key] = profiles
    return profiles


def load_complex_profile(
    approach: str, profile_name: str
) -> Optional[List[Dict[str, Any]]]:
    """Return a deep copy of the requested complex-model profile.

    Args:
        approach: Either "bon" or "moa".
        profile_name: Profile identifier (e.g., "rapid", "deep").

    Returns:
        List describing the role configuration, or None if not found/invalid.
    """

    path = _resolve_config_path(approach)
    profiles = _load_profiles_from_path(approach, path)
    if not profiles:
        return None

    profile = profiles.get(profile_name)
    if profile is None:
        return None

    return deepcopy(profile)


def reset_config_cache():
    """Clear cached config data (useful for tests)."""

    _PROFILE_CACHE.clear()
