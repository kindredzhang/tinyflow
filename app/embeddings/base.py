"""Embedding Abstraction Layer

Supports multiple embedding model providers
"""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbedding(ABC):
    """Embedding abstract base class"""

    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Convert text list to vectors

        Args:
            texts: List of text strings

        Returns:
            List of vectors (one for each text)
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get vector dimension

        Returns:
            Vector dimension
        """
        pass
