"""Chroma vector database implementation"""

import logging
from typing import TYPE_CHECKING, Any, List, Optional, cast

from app.config.settings import settings
from app.vector.base import BaseVectorDB

if TYPE_CHECKING:
    import chromadb

logger = logging.getLogger("tinyflow.vector.chroma")


class ChromaVectorDB(BaseVectorDB):
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError(
                "ChromaDB dependency not found. "
                "Please install with `pip install tinyflow[vector]`"
            ) from e

        import os

        self.persist_directory = persist_directory or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data",
            "chroma_db",
        )
        self.collection_name = collection_name or settings.VECTOR_DB_COLLECTION or "conversations"

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        logger.info(f"Initialized ChromaDB client at: {self.persist_directory}")

        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.debug(f"Loaded existing ChromaDB collection: {self.collection_name}")
        except Exception:
            logger.info(f"Creating new ChromaDB collection: {self.collection_name}")
            self.collection = self.client.create_collection(
                name=self.collection_name, metadata={"hnsw:space": "cosine"}
            )

    async def add(self, texts: List[str], metadata: Optional[dict] = None) -> List[str]:
        logger.debug(f"Adding {len(texts)} documents to ChromaDB")
        if metadata is None:
            metadata = {}

        ids = [f"doc_{i}" for i in range(len(texts))]

        # ChromaDB doesn't accept empty metadata dicts
        # Only pass metadatas if we have actual non-empty metadata
        if metadata:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=cast(Any, [metadata] * len(texts)),
            )
        else:
            self.collection.add(
                ids=ids,
                documents=texts,
            )

        return ids

    async def search(
        self, query: str, limit: int = 5, filter: Optional[dict] = None
    ) -> List[dict]:
        logger.debug(f"Searching ChromaDB for: {query} (limit={limit}, filter={filter})")
        if filter is None:
            filter = {}

        # ChromaDB doesn't accept empty where dicts
        # Only pass where parameter if we have actual non-empty filter
        if filter:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
                where=filter,
            )
        else:
            results = self.collection.query(
                query_texts=[query],
                n_results=limit,
            )

        search_results = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                search_results.append(
                    {
                        "id": results["ids"][0][i]
                        if results["ids"] and results["ids"][0]
                        else "",
                        "text": doc,
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"] and results["metadatas"][0]
                        else {},
                        "distance": results["distances"][0][i]
                        if results["distances"] and results["distances"][0]
                        else 0.0,
                    }
                )

        return search_results

    async def delete(self, ids: List[str]) -> int:
        if not ids:
            return 0

        deleted_count = 0
        for doc_id in ids:
            try:
                self.collection.delete(ids=[doc_id])
                deleted_count += 1
            except Exception:
                pass
        return deleted_count

    async def clear(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    async def count(self) -> int:
        return self.collection.count()
