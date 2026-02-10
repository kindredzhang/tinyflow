"""Sentence Transformers Local Embedding Implementation

Use open source models for local embedding

Note: This module requires torch and sentence_transformers.
If installation fails, use app.embeddings.openai_embedding.OpenAIEmbedding instead.
"""

from typing import Any, List, Optional

from sentence_transformers import SentenceTransformer

from app.config.settings import settings
from app.embeddings.base import BaseEmbedding


class SentenceTransformerEmbedding(BaseEmbedding):
    """Local Sentence Transformers Embedding Implementation

    Supports:
    - all-MiniLM-L6-v2 (384 dimensions)
    - all-mpnet-base-v2 (768 dimensions)
    - paraphrase-multilingual-MiniLM-L12-v2 (768 dimensions)

    Configuration precedence: Parameters > settings > Environment variables > Default values
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """Initialize Sentence Transformer Embedding

        Args:
            model_path: Model path or name (defaults to settings or environment variables)
            device: Computation device (auto/cpu/cuda/mps, defaults to settings or environment variables)

        Raises:
            ImportError: If torch/sentence_transformers is not installed
        """
        model_path = model_path or settings.EMBEDDING_MODEL_PATH
        device = device or settings.EMBEDDING_MODEL_DEVICE

        self.model = SentenceTransformer(model_path)
        self.device = device

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert text list to vectors"""
        self.model.to(self.device)

        embeddings: Any = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        return list(embeddings)

    def get_dimension(self) -> int:
        """Get vector dimension"""
        dim = self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError("Unable to determine embedding dimension")
        return dim
