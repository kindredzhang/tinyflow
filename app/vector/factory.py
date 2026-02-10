"""Vector Database Factory

Select different vector database implementations based on configuration
"""

from typing import Optional

from app.config.helpers import filter_none_kwargs, get_config_value
from app.config.settings import settings
from app.core.exceptions import ConfigurationError, VectorError
from app.vector.base import BaseVectorDB


class VectorDBFactory:
    """Vector Database Factory class.

    Configuration Precedence:
        1. Explicit parameters (highest)
        2. Environment variables (VECTOR_DB_* or Provider specific)
        3. Settings (from .env or defaults)
        4. Provider defaults (lowest)
    """

    @staticmethod
    def create(
        provider: Optional[str] = None,
        **kwargs,
    ) -> BaseVectorDB:
        """Create Vector Database instance.

        Args:
            provider: Provider name ('chroma', 'qdrant').
                     Falls back to VECTOR_DB_PROVIDER env var -> settings -> default.
            **kwargs: Provider-specific configuration parameters.
                     - Chroma: persist_directory
                     - Qdrant: url, api_key

        Returns:
            Vector Database instance

        Raises:
            ValueError: If provider is not supported
        """
        provider = get_config_value(
            provider,
            env_key="VECTOR_DB_PROVIDER",
            settings_field="VECTOR_DB_PROVIDER",
            default_value="chroma",
        )

        if not provider:
            raise ConfigurationError("Vector DB Provider must be specified")

        provider = provider.lower()

        if provider == "chroma":
            # Lazy import
            from app.vector.chroma_db import ChromaVectorDB

            chroma_kwargs = {
                "persist_directory": get_config_value(
                    kwargs.get("persist_directory"),
                    "VECTOR_DB_PATH",
                    "VECTOR_DB_PATH",
                )
            }
            chroma_kwargs = filter_none_kwargs(**chroma_kwargs)
            return ChromaVectorDB(**chroma_kwargs)

        elif provider == "qdrant":
            # Lazy import
            from app.vector.qdrant_db import QdrantVectorDB

            qdrant_kwargs = {
                "url": get_config_value(kwargs.get("url"), "QDRANT_URL", "VECTOR_DB_URL"),
                "api_key": get_config_value(
                    kwargs.get("api_key"), "QDRANT_API_KEY", "VECTOR_DB_API_KEY"
                ),
            }
            qdrant_kwargs = filter_none_kwargs(**qdrant_kwargs)
            return QdrantVectorDB(**qdrant_kwargs)

        else:
            raise VectorError(f"Unsupported vector database provider: {provider}")
