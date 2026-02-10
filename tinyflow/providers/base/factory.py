import logging
from typing import Optional

from tinyflow.config.helpers import filter_none_kwargs, get_config_value
from tinyflow.core.exceptions import ConfigurationError
from tinyflow.providers.anthropic_llm import AnthropicProvider
from tinyflow.providers.base.llm import BaseLLM
from tinyflow.providers.gemini_llm import GeminiProvider
from tinyflow.providers.openai_llm import OpenAIProvider

logger = logging.getLogger("tinyflow.providers.factory")


class LLMFactory:
    """Factory for creating LLM provider instances.

    Supports both configuration-driven (env vars) and explicit instantiation.
    Configuration Precedence:
        1. Explicit parameters (highest)
        2. Environment variables (LLM_*)
        3. Settings (from .env or defaults)
        4. Provider defaults (lowest)
    """

    @staticmethod
    def create(
        provider: Optional[str] = None,
        **kwargs,
    ) -> BaseLLM:
        """Create an LLM provider instance.

        Args:
            provider: Provider name ('openai', 'anthropic', 'gemini').
                     Falls back to LLM_PROVIDER env var -> settings -> default.
            **kwargs: Additional provider-specific arguments (api_key, model, base_url, etc.)
                     Common args (model, api_key, base_url) are resolved against
                     LLM_* env vars and settings if not provided.

        Returns:
            Configured provider instance.

        Raises:
            ValueError: If provider is not specified and not found in configuration.
        """
        # Determine provider
        provider = get_config_value(
            provider,
            env_key="LLM_PROVIDER",
            settings_field="LLM_PROVIDER",
            default_value="openai",
        )

        if not provider:
            raise ConfigurationError("LLM Provider must be specified")

        provider = provider.lower()
        logger.info(f"Creating LLM provider: {provider}")

        common_kwargs = {
            "base_url": get_config_value(kwargs.get("base_url"), "LLM_BASE_URL", "LLM_BASE_URL"),
            "api_key": get_config_value(kwargs.get("api_key"), "LLM_API_KEY", "LLM_API_KEY"),
            "model": get_config_value(kwargs.get("model"), "LLM_MODEL", "LLM_MODEL"),
        }

        provider_kwargs = {**kwargs, **common_kwargs}
        provider_kwargs = filter_none_kwargs(**provider_kwargs)

        if provider == "openai":
            return OpenAIProvider(**provider_kwargs)
        elif provider == "deepseek":
            if "base_url" not in provider_kwargs:
                provider_kwargs["base_url"] = "https://api.deepseek.com"
            return OpenAIProvider(**provider_kwargs)
        elif provider == "anthropic":
            return AnthropicProvider(**provider_kwargs)
        elif provider == "gemini":
            return GeminiProvider(**provider_kwargs)
        else:
            raise ConfigurationError(
                f"Unsupported provider: {provider}. Supported: openai, anthropic, gemini"
            )
