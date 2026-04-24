"""Provider-agnostic structured-LLM subsystem."""
from _system.llm.client import call_structured
from _system.llm.config import (
    DEFAULT_TEMPERATURE,
    LlmConfig,
    Provider,
    all_env_vars,
    config_path,
    env_var_for,
    load_config,
    save_config,
)
from _system.llm.errors import (
    LLMError,
    LLMPermanentError,
    LLMTransientError,
    PromptPlaceholderError,
    ProviderAmbiguous,
    ProviderConfigError,
    ProviderKeyMissing,
    ProviderUnconfigured,
)
from _system.llm.prompt_loader import LoadedPrompt, load_prompt
from _system.llm.selection import ResolvedProvider, resolve_provider

__all__ = [
    "DEFAULT_TEMPERATURE",
    "LLMError",
    "LLMPermanentError",
    "LLMTransientError",
    "LlmConfig",
    "LoadedPrompt",
    "PromptPlaceholderError",
    "Provider",
    "ProviderAmbiguous",
    "ProviderConfigError",
    "ProviderKeyMissing",
    "ProviderUnconfigured",
    "ResolvedProvider",
    "all_env_vars",
    "call_structured",
    "config_path",
    "env_var_for",
    "load_config",
    "load_prompt",
    "resolve_provider",
    "save_config",
]
