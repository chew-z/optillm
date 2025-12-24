import time
import logging
import sys
import io
import builtins
from contextlib import redirect_stdout

# Configure litellm logging BEFORE importing litellm modules
logging.getLogger("litellm").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm.utils").setLevel(logging.WARNING)
logging.getLogger("litellm.integrations").setLevel(logging.WARNING)

# Monkey-patch builtins.print to suppress LiteLLM's verbose output
# This catches print() calls from background threads that redirect_stdout() misses
_original_print = print


def _suppress_print(*args, **kwargs):
    """Suppress print() output unless it contains error/debug info worth logging."""
    # Convert args to string to check content
    if args:
        msg = str(args[0]) if args else ""
        # Suppress specific LiteLLM messages
        if "Provider List:" in msg or "docs.litellm.ai" in msg:
            return
        # Allow other prints through (errors, important info)
        _original_print(*args, **kwargs)


# Only patch if not already patched
if builtins.print.__name__ != "_suppress_print":
    builtins.print = _suppress_print

import litellm
from litellm import completion
from litellm.utils import get_valid_models
from typing import List, Dict, Optional

# Configure litellm to drop unsupported parameters and suppress verbose output
litellm.drop_params = True
litellm.set_verbose = False  # Disable litellm's verbose mode

# Also try to suppress at the module level
try:
    litellm.logger.setLevel(logging.WARNING)
except:
    pass

SAFETY_SETTINGS = [
    {"category": cat, "threshold": "BLOCK_NONE"}
    for cat in [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
    ]
]


class LiteLLMWrapper:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = base_url
        self.chat = self.Chat()
        # litellm.set_verbose=True

    class Chat:
        class Completions:
            @staticmethod
            def create(model: str, messages: List[Dict[str, str]], **kwargs):
                # Suppress LiteLLM's verbose print() output during completion
                with redirect_stdout(io.StringIO()):
                    if model.startswith("gemini"):
                        response = completion(
                            model=model,
                            messages=messages,
                            **kwargs,
                            safety_settings=SAFETY_SETTINGS,
                        )
                    else:
                        response = completion(model=model, messages=messages, **kwargs)
                # Convert LiteLLM response to match OpenAI response structure
                return response

        completions = Completions()

    class Models:
        @staticmethod
        def list():
            try:
                # Get all valid models from LiteLLM
                valid_models = get_valid_models()

                # Format the response to match OpenAI's API format
                model_list = []
                for model in valid_models:
                    model_list.append(
                        {
                            "id": model,
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "litellm",
                        }
                    )

                return {"object": "list", "data": model_list}
            except Exception as e:
                # Fallback to a basic list if there's an error
                print(f"Error fetching LiteLLM models: {str(e)}")
                return {
                    "object": "list",
                    "data": [
                        {
                            "id": "gpt-4o-mini",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "litellm",
                        },
                        {
                            "id": "gpt-4o",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "litellm",
                        },
                        {
                            "id": "command-nightly",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "litellm",
                        },
                        {
                            "id": "claude-3-opus-20240229",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "litellm",
                        },
                        {
                            "id": "claude-3-sonnet-20240229",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "litellm",
                        },
                        {
                            "id": "gemini-1.5-pro-latest",
                            "object": "model",
                            "created": int(time.time()),
                            "owned_by": "litellm",
                        },
                    ],
                }

    models = Models()
