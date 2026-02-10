"""OpenAI Embedding Implementation

Use OpenAI API for text embedding
"""

from typing import List, Optional

from openai import AsyncOpenAI

from tinyflow.config.settings import settings
from tinyflow.embeddings.base import BaseEmbedding


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI Embedding Implementation

    Supports:
    - text-embedding-3-small
    - text-embedding-3-large
    - text-embedding-ada-002

    Configuration precedence: Parameters > settings > Environment variables > Default values
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize OpenAI Embedding

        Args:
            api_key: OpenAI API key (defaults to settings or environment variables)
            base_url: API base URL (defaults to settings or environment variables)
            model: Model name (defaults to settings or environment variables)
        """
        api_key = api_key or settings.EMBEDDING_API_KEY or settings.LLM_API_KEY or ""
        base_url = base_url or settings.EMBEDDING_BASE_URL or settings.LLM_BASE_URL
        model = model or settings.EMBEDDING_MODEL

        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert text list to vectors"""
        # Batch processing, max 2048 texts per batch
        embeddings = []
        batch_size = 2048

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = await self.client.embeddings.create(
                input=batch,
                model=self.model,
            )

            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)

        return embeddings

    def get_dimension(self) -> int:
        """Get vector dimension"""
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dimensions.get(self.model, 1536)
