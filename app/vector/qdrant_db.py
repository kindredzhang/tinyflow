"""Qdrant vector database implementation

Use Qdrant Cloud for vector storage and retrieval
"""

import uuid
from typing import List, Optional, Union

from app.vector.base import BaseVectorDB


class QdrantVectorDB(BaseVectorDB):
    """Qdrant vector database implementation

    Suitable for:
    - Production environment
    - High-performance retrieval
    - Cloud deployment

    Configuration:
    - QDRANT_URL: Qdrant server address
    - QDRANT_API_KEY: API key
    - COLLECTION_NAME: Collection name (default conversations)
    """

    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize Qdrant client

        Args:
            url: Qdrant server address
            api_key: Qdrant API key
        """
        import os

        from qdrant_client import AsyncQdrantClient

        url = url or os.getenv("QDRANT_URL")
        api_key = api_key or os.getenv("QDRANT_API_KEY")

        if not url or not api_key:
            raise ValueError("QDRANT_URL and QDRANT_API_KEY must be configured")

        self.collection_name = "conversations"
        self.client = AsyncQdrantClient(
            url=url,
            api_key=api_key,
        )

    async def _ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams

        try:
            await self.client.get_collection(collection_name=self.collection_name)
        except Exception:
            # Collection doesn't exist, create new one
            vector_params = VectorParams(
                size=1536,  # OpenAI embedding dimension
                distance=Distance.COSINE,
            )
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=vector_params,
            )

    async def add(self, texts: List[str], metadata: Optional[dict] = None) -> List[str]:
        """Add documents to Qdrant

        Args:
            texts: List of texts to add
            metadata: Optional metadata

        Returns:
            Generated document ID list
        """
        await self._ensure_collection()
        # Prepare metadata
        if metadata is None:
            metadata = {}

        # Batch insert
        from qdrant_client.models import PointStruct

        points = []
        generated_ids = []
        for i, text in enumerate(texts):
            # Generate UUID as string to ensure proper serialization
            doc_id = str(uuid.uuid4())
            points.append(
                PointStruct(
                    id=doc_id,  # Qdrant accepts UUID as string or UUID object
                    vector=[0.0] * 1536,  # Temporary placeholder, actual needs embedding
                    payload={
                        "text": text,
                        **metadata,
                    },
                )
            )
            generated_ids.append(doc_id)

        # Upsert points using client
        await self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        # Return the IDs of the inserted points
        return generated_ids

    async def search(
        self, query: str, limit: int = 5, filter: Optional[dict] = None
    ) -> List[dict]:
        """Search in Qdrant

        Args:
            query: Query text
            limit: Maximum number of results to return
            filter: Optional filter condition

        Returns:
            Search result list, each result contains id, text, metadata, distance
        """
        await self._ensure_collection()
        if filter is None:
            filter = {}

        # Need to vectorize query text first
        # Simplified handling here, actual needs embedding service call
        from qdrant_client.models import Filter

        # Use query_points for Qdrant client
        search_result = await self.client.query_points(
            collection_name=self.collection_name,
            query=[0.0] * 1536,  # Temporary placeholder
            query_filter=Filter(**filter) if filter else None,
            limit=limit,
            with_payload=True,
        )

        # Convert result format
        search_results = []
        for hit in search_result.points:
            search_results.append(
                {
                    "id": str(hit.id),
                    "text": hit.payload.get("text", "") if hit.payload else "",
                    "metadata": hit.payload if hit.payload else {},
                    "distance": hit.score,
                }
            )

        return search_results

    async def delete(self, ids: Optional[List[Union[str, uuid.UUID]]] = None) -> int:
        """Delete documents with specified IDs"""
        if not ids:
            return 0

        deleted_count = 0
        for doc_id in ids:
            try:
                # Convert string IDs to UUID if needed
                if isinstance(doc_id, str):
                    doc_id = uuid.UUID(doc_id)
                await self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=[doc_id],
                )
                deleted_count += 1
            except Exception:
                pass
        return deleted_count

    async def clear(self) -> None:
        """Clear Qdrant collection"""
        try:
            await self.client.delete_collection(collection_name=self.collection_name)
        except Exception:
            pass
        from qdrant_client.models import Distance, VectorParams

        await self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    async def count(self) -> int:
        """Get total number of documents"""
        try:
            collection_info = await self.client.get_collection(
                collection_name=self.collection_name
            )
            return (
                collection_info.points_count
                if collection_info and collection_info.points_count is not None
                else 0
            )
        except Exception:
            return 0
