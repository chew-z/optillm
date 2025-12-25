import logging
import os
from typing import List, Tuple, Dict, Any, Optional
import optillm
from optillm.config_loader import load_complex_profile

logger = logging.getLogger(__name__)

# PlanSearch requires four role entries per configuration:
#   - OBSERVER: Generates initial observations
#   - BRAINSTORMER: Generates derived observations
#   - PLANNER: Generates natural language solution
#   - CODER: Implements solution in Python
PLANSEARCH_REQUIRED_ROLE_NAMES = [
    "OBSERVER",
    "BRAINSTORMER",
    "PLANNER",
    "CODER",
]
PLANSEARCH_GENERATION_ROLE_COUNT = 4
DEFAULT_REASONING_EFFORT = "high"

_litellm_client = None


def _validate_plansearch_profile(config: List[Dict[str, Any]]) -> bool:
    """Ensure a PlanSearch profile defines all required roles."""

    if len(config) < len(PLANSEARCH_REQUIRED_ROLE_NAMES):
        logger.error(
            "PlanSearch profile has %s roles but requires %s (%s)",
            len(config),
            len(PLANSEARCH_REQUIRED_ROLE_NAMES),
            ", ".join(PLANSEARCH_REQUIRED_ROLE_NAMES),
        )
        return False

    defined_names = {role.get("name") for role in config if isinstance(role, dict)}
    missing = [name for name in PLANSEARCH_REQUIRED_ROLE_NAMES if name not in defined_names]
    if missing:
        logger.error(
            "PlanSearch profile is missing required roles: %s",
            ", ".join(missing),
        )
        return False

    return True


def _get_litellm_client():
    """Get or create the LiteLLM client for OpenRouter models (thread-safe singleton)."""
    global _litellm_client
    if _litellm_client is None:
        from optillm.litellm_wrapper import LiteLLMWrapper

        _litellm_client = LiteLLMWrapper()
        logger.info("PlanSearch: Created LiteLLM client for OpenRouter models")
    return _litellm_client


def _get_role_by_name(
    config: List[Dict[str, Any]],
    role_name: str,
    default_model: str = "openrouter/google/gemini-3-flash-preview",
) -> Dict[str, Any]:
    """Find a role configuration by name from the config list.

    Args:
        config: Full configuration list
        role_name: Name/title of the role to find (e.g., 'OBSERVER', 'PLANNER')
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


def _build_homogeneous_plansearch_config(model: str) -> List[Dict[str, Any]]:
    """Build a config where all roles use the same model."""
    return [
        {"name": "OBSERVER", "model": model},
        {"name": "BRAINSTORMER", "model": model},
        {"name": "PLANNER", "model": model},
        {"name": "CODER", "model": model},
    ]


def _resolve_plansearch_config(
    model: Optional[str],
) -> Tuple[List[Dict[str, Any]], str]:
    """Return the PlanSearch role configuration and fallback model."""

    resolved_config = None

    if model:
        force_predefined = model.startswith("pre-")
        candidate_key = model[4:] if force_predefined else model
        resolved_config = load_complex_profile("plansearch", candidate_key)
        if resolved_config and not _validate_plansearch_profile(resolved_config):
            resolved_config = None

        if resolved_config:
            logger.info("PlanSearch: Using complex profile '%s'", candidate_key)
        elif force_predefined:
            logger.warning(
                "PlanSearch: Requested complex profile '%s' not found; falling back to default roles",
                candidate_key,
            )

    if resolved_config:
        default_model = _determine_default_model(None)
        return resolved_config, default_model

    default_model = _determine_default_model(model)
    if not model or model in {"auto", "none"}:
        logger.info(
            "PlanSearch: No model specified, using default OPTILLM_MODEL: %s",
            default_model,
        )
    else:
        logger.info(
            "PlanSearch: Using model '%s' for all PlanSearch roles",
            default_model,
        )

    return _build_homogeneous_plansearch_config(default_model), default_model


class PlanSearch:
    def __init__(
        self,
        system_prompt: str,
        client,
        config: List[Dict[str, Any]],
        default_model: str,
        request_config: dict = None,
        request_id: str = None,
    ):
        self.system_prompt = system_prompt
        self.default_client = client
        self.config = config
        self.default_model = default_model
        self.request_id = request_id
        self.plansearch_completion_tokens = 0

        # Extract max_tokens from request_config with default
        self.max_tokens = 4096
        if request_config:
            self.max_tokens = request_config.get("max_tokens", self.max_tokens)

    def _get_client_model_and_params(self, role_name: str) -> Tuple[Any, str, str, str]:
        """Get the appropriate client, model, system suffix, and reasoning effort for a given role."""
        role_config = _get_role_by_name(self.config, role_name, self.default_model)
        model = role_config.get("model", self.default_model)
        system_suffix = role_config.get("system_suffix", "")
        reasoning_effort = role_config.get("reasoning_effort", DEFAULT_REASONING_EFFORT)

        # Route to appropriate client based on model prefix
        # OpenRouter models go through LiteLLM, others use the default client (e.g., Z.ai)
        if "/" in model:  # Heuristic for OpenRouter/LiteLLM models
            provider = "OpenRouter (via LiteLLM)"
        else:
            provider = f"Direct ({type(self.default_client).__name__})"

        # Log the role configuration with model and parameters
        logger.info(
            f"{role_name}: Using model='{model}' via {provider}, "
            f"reasoning_effort={reasoning_effort}, max_tokens={self.max_tokens}"
        )

        if "/" in model:  # Heuristic for OpenRouter/LiteLLM models
            return _get_litellm_client(), model, system_suffix, reasoning_effort
        return self.default_client, model, system_suffix, reasoning_effort

    def generate_observations(
        self, problem: str, num_observations: int = 3
    ) -> List[str]:
        client, model, suffix, effort = self._get_client_model_and_params("OBSERVER")
        prompt = f"""You are an expert Python programmer. You will be given a competitive programming question
(problem specification). You will return several useful, non-obvious, and correct observations
about the problem, like hints to solve the problem. You will NOT return any code. Be as
creative as possible, going beyond what you think is intuitively correct.

Here is the competitive programming problem:
{problem}

Please provide {num_observations} observations."""

        # Prepare request for logging
        provider_request = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt + suffix},
                {"role": "user", "content": prompt},
            ],
            "reasoning_effort": effort,
        }

        response = optillm.safe_completions_create(client, provider_request)

        # Log provider call if conversation logging is enabled
        if (
            hasattr(optillm, "conversation_logger")
            and optillm.conversation_logger
            and self.request_id
        ):
            response_dict = (
                response.model_dump() if hasattr(response, "model_dump") else response
            )
            optillm.conversation_logger.log_provider_call(
                self.request_id, provider_request, response_dict
            )
        self.plansearch_completion_tokens += response.usage.completion_tokens

        # Check for valid response with None-checking
        if (
            response is None
            or not response.choices
            or response.choices[0].message.content is None
            or response.choices[0].finish_reason == "length"
        ):
            logger.warning(
                "Observations response truncated or empty, returning empty list"
            )
            return []

        observations = response.choices[0].message.content.strip().split("\n")
        return [obs.strip() for obs in observations if obs.strip()]

    def generate_derived_observations(
        self, problem: str, observations: List[str], num_new_observations: int = 2
    ) -> List[str]:
        client, model, suffix, effort = self._get_client_model_and_params(
            "BRAINSTORMER"
        )
        prompt = f"""You are an expert Python programmer. You will be given a competitive programming question
(problem specification) and several correct observations about the problem.
You will brainstorm several new, useful, and correct observations about the problem, derived
from the given observations. You will NOT return any code. Be as creative as possible, going
beyond what you think is intuitively correct.

Here is the competitive programming problem:
{problem}

Here are the existing observations:
{chr(10).join(f"{i + 1}. {obs}" for i, obs in enumerate(observations))}

Please provide {num_new_observations} new observations derived from the existing ones."""

        # Prepare request for logging
        provider_request = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt + suffix},
                {"role": "user", "content": prompt},
            ],
            "reasoning_effort": effort,
        }

        response = optillm.safe_completions_create(client, provider_request)

        # Log provider call if conversation logging is enabled
        if (
            hasattr(optillm, "conversation_logger")
            and optillm.conversation_logger
            and self.request_id
        ):
            response_dict = (
                response.model_dump() if hasattr(response, "model_dump") else response
            )
            optillm.conversation_logger.log_provider_call(
                self.request_id, provider_request, response_dict
            )
        self.plansearch_completion_tokens += response.usage.completion_tokens

        # Check for valid response with None-checking
        if (
            response is None
            or not response.choices
            or response.choices[0].message.content is None
            or response.choices[0].finish_reason == "length"
        ):
            logger.warning(
                "Derived observations response truncated or empty, returning empty list"
            )
            return []

        new_observations = response.choices[0].message.content.strip().split("\n")
        return [obs.strip() for obs in new_observations if obs.strip()]

    def generate_solution(self, problem: str, observations: List[str]) -> str:
        client, model, suffix, effort = self._get_client_model_and_params("PLANNER")
        prompt = f"""Here is the competitive programming problem:
{problem}

Here are the intelligent observations to help solve the problem:
{chr(10).join(f"Observation {i + 1}: {obs}" for i, obs in enumerate(observations))}

Use these observations above to brainstorm a natural language solution to the problem above.
Note that your intuition may lead you astray, so come up with simple, creative ideas that
go beyond what you would usually come up with and exceeds your narrow intuition.
Quote relevant parts of the observations EXACTLY before each step of the solution. QUOTING
IS CRUCIAL."""

        # Prepare request for logging
        provider_request = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt + suffix},
                {"role": "user", "content": prompt},
            ],
            "reasoning_effort": effort,
        }

        response = optillm.safe_completions_create(client, provider_request)

        # Log provider call if conversation logging is enabled
        if (
            hasattr(optillm, "conversation_logger")
            and optillm.conversation_logger
            and self.request_id
        ):
            response_dict = (
                response.model_dump() if hasattr(response, "model_dump") else response
            )
            optillm.conversation_logger.log_provider_call(
                self.request_id, provider_request, response_dict
            )
        self.plansearch_completion_tokens += response.usage.completion_tokens

        # Check for valid response with None-checking
        if (
            response is None
            or not response.choices
            or response.choices[0].message.content is None
            or response.choices[0].finish_reason == "length"
        ):
            logger.error(
                "Solution generation response truncated or empty. Consider increasing max_tokens."
            )
            return "Error: Response was truncated due to token limit. Please increase max_tokens or max_completion_tokens."

        return response.choices[0].message.content.strip()

    def implement_solution(self, problem: str, solution: str) -> str:
        client, model, suffix, effort = self._get_client_model_and_params("CODER")
        prompt = f"""You are an expert Python programmer. You will be given a question (problem specification)
and a natural language solution/tutorial that describes how to solve the problem. You will
generate a correct Python program that matches said specification and tutorial and passes
all tests. You will NOT return anything except for the program inside markdown codeblocks.

Problem:
{problem}

Solution:
{solution}

Please implement the solution in Python."""

        # Prepare request for logging
        provider_request = {
            "model": model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt + suffix},
                {"role": "user", "content": prompt},
            ],
            "reasoning_effort": effort,
        }

        response = optillm.safe_completions_create(client, provider_request)

        # Log provider call if conversation logging is enabled
        if (
            hasattr(optillm, "conversation_logger")
            and optillm.conversation_logger
            and self.request_id
        ):
            response_dict = (
                response.model_dump() if hasattr(response, "model_dump") else response
            )
            optillm.conversation_logger.log_provider_call(
                self.request_id, provider_request, response_dict
            )
        self.plansearch_completion_tokens += response.usage.completion_tokens

        # Check for valid response with None-checking
        if (
            response is None
            or not response.choices
            or response.choices[0].message.content is None
            or response.choices[0].finish_reason == "length"
        ):
            logger.error(
                "Implementation response truncated or empty. Consider increasing max_tokens."
            )
            return "Error: Response was truncated due to token limit. Please increase max_tokens or max_completion_tokens."

        return response.choices[0].message.content.strip()

    def solve(
        self,
        problem: str,
        num_initial_observations: int = 3,
        num_derived_observations: int = 2,
    ) -> Tuple[str, str]:
        logger.info("Generating initial observations")
        initial_observations = self.generate_observations(
            problem, num_initial_observations
        )

        logger.info("Generating derived observations")
        derived_observations = self.generate_derived_observations(
            problem, initial_observations, num_derived_observations
        )

        all_observations = initial_observations + derived_observations

        logger.info("Generating solution based on observations")
        natural_language_solution = self.generate_solution(problem, all_observations)

        logger.info("Implementing solution in Python")
        python_implementation = self.implement_solution(
            problem, natural_language_solution
        )

        return natural_language_solution, python_implementation

    def solve_multiple(
        self,
        problem: str,
        n: int,
        num_initial_observations: int = 3,
        num_derived_observations: int = 2,
    ) -> List[str]:
        solutions = []
        for _ in range(n):
            _, python_implementation = self.solve(
                problem, num_initial_observations, num_derived_observations
            )
            solutions.append(python_implementation)
        return solutions


def plansearch(
    system_prompt: str,
    initial_query: str,
    client,
    model: str,
    n: int = 1,
    request_config: dict = None,
    request_id: str = None,
) -> List[str]:
    # Resolve config and default model
    config, default_model = _resolve_plansearch_config(model)

    planner = PlanSearch(
        system_prompt,
        client,
        config,
        default_model,
        request_config=request_config,
        request_id=request_id,
    )
    return (
        planner.solve_multiple(initial_query, n),
        planner.plansearch_completion_tokens,
    )
