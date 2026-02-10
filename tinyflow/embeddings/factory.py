"""Embedding Factory

Select different Embedding implementations based on configuration

Supports:
- openai: OpenAI API Embedding
- local / sentence-transformers: Local Sentence Transformers model

Configuration precedence: Parameters > settings > Environment variables > Default values
"""

from typing import Optional

from tinyflow.config.helpers import filter_none_kwargs, get_config_value
from tinyflow.config.settings import settings
from tinyflow.embeddings.base import BaseEmbedding
from tinyflow.embeddings.openai_embedding import OpenAIEmbedding


class EmbeddingFactory:
    """Embedding Factory class.

    Configuration Precedence:
        1. Explicit parameters (highest)
        2. Environment variables (EMBEDDING_PROVIDER)
        3. Settings (from .env or defaults)
        4. Provider defaults (lowest)
    """

    @staticmethod
    def create(
        provider: Optional[str] = None,
        **kwargs,
    ) -> BaseEmbedding:
        """Create Embedding provider instance.

        Args:
            provider: Provider name ('openai', 'local').
                     Falls back to EMBEDDING_PROVIDER env var -> settings -> default.
            **kwargs: Provider-specific configuration parameters.
                     - OpenAI: api_key, model, base_url
                     - Local: model_path, device

        Returns:
            Configured embedding provider instance.

        Raises:
            ValueError: If provider is not specified or supported.
        """
        provider = get_config_value(
            provider,
            env_key="EMBEDDING_PROVIDER",
            settings_field="EMBEDDING_PROVIDER",
            default_value="openai",
        )

        if not provider:
            raise ValueError("Embedding Provider must be specified")

        provider = provider.lower()

        if provider == "openai":
            openai_kwargs = {
                "api_key": get_config_value(
                    kwargs.get("api_key"), "EMBEDDING_API_KEY", "EMBEDDING_API_KEY"
                ),
                "base_url": get_config_value(
                    kwargs.get("base_url"), "EMBEDDING_BASE_URL", "EMBEDDING_BASE_URL"
                ),
                "model": get_config_value(
                    kwargs.get("model"), "EMBEDDING_MODEL", "EMBEDDING_MODEL"
                ),
            }
            # Use LLM_API_KEY as fallback if EMBEDDING_API_KEY is not set
            if not openai_kwargs["api_key"]:
                openai_kwargs["api_key"] = get_config_value(
                    None, "LLM_API_KEY", "LLM_API_KEY"
                )

            openai_kwargs = filter_none_kwargs(**openai_kwargs)
            return OpenAIEmbedding(**openai_kwargs)

        elif provider in ["local", "sentence-transformers"]:
            # Lazy import to avoid hard dependency on torch/sentence_transformers
            from tinyflow.embeddings.local_embedding import SentenceTransformerEmbedding

            local_kwargs = {
                "model_path": get_config_value(
                    kwargs.get("model_path"),
                    "EMBEDDING_MODEL_PATH",
                    "EMBEDDING_MODEL_PATH",
                ),
                "device": get_config_value(
                    kwargs.get("device"), "EMBEDDING_MODEL_DEVICE", "EMBEDDING_MODEL_DEVICE"
                ),
            }
            local_kwargs = filter_none_kwargs(**local_kwargs)
            return SentenceTransformerEmbedding(**local_kwargs)

        else:
            raise ValueError(
                f"Unsupported embedding provider: {provider}. Supported: openai, local"
            )

        if not provider:
            raise ValueError("Embedding Provider must be specified")

        provider = provider.lower()

        if provider == "openai":
            openai_kwargs = {
                "api_key": get_config_value(
                    kwargs.get("api_key"), "EMBEDDING_API_KEY", "EMBEDDING_API_KEY"
                ),
                "base_url": get_config_value(
                    kwargs.get("base_url"), "EMBEDDING_BASE_URL", "EMBEDDING_BASE_URL"
                ),
                "model": get_config_value(
                    kwargs.get("model"), "EMBEDDING_MODEL", "EMBEDDING_MODEL"
                ),
            }
            openai_kwargs = filter_none_kwargs(**openai_kwargs)
            return OpenAIEmbedding(**openai_kwargs)

        elif provider in ("local", "sentence-transformers"):
            local_kwargs = {
                "model_path": get_config_value(
                    kwargs.get("model_path"), "LOCAL_MODEL", "EMBEDDING_MODEL_PATH"
                ),
                "device": get_config_value(
                    kwargs.get("device"), "DEVICE", "EMBEDDING_MODEL_DEVICE"
                ),
            }
            local_kwargs = filter_none_kwargs(**local_kwargs)
            return SentenceTransformerEmbedding(**local_kwargs)
        else:
            raise ValueError(
                f"Unsupported embedding provider: {provider}. "
                "Supported providers: openai, local, sentence-transformers"
            )
