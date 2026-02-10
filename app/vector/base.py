"""Vector Database Abstraction Layer

Focuses on vector database operations, supports multiple providers
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseVectorDB(ABC):
    """Vector Database Abstract Base Class

    Focuses on vector operations, agnostic of conversation logic
    """

    @abstractmethod
    async def add(self, texts: List[str], metadata: Optional[dict] = None) -> List[str]:
        """Add documents to vector database

        Args:
            texts: List of texts to add
            metadata: Optional metadata (e.g. role, timestamp)

        Returns:
            List of generated document IDs
        """
        pass

    @abstractmethod
    async def search(
        self, query: str, limit: int = 5, filter: Optional[dict] = None
    ) -> List[dict]:
        """Search in vector database

        Args:
            query: Query text
            limit: Maximum number of results to return
            filter: Optional filter conditions (e.g. role="user")

        Returns:
            Search result list, each result containing id, text, metadata, distance
        """
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> int:
        """Delete documents with specified IDs

        Args:
            ids: List of document IDs

        Returns:
            Number of deleted documents
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """Clear vector database"""
        pass

    @abstractmethod
    async def count(self) -> int:
        """Get total number of documents"""
        pass
