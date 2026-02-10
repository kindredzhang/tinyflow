import logging
from typing import List

from tinyflow.vector.base import BaseVectorDB

from .base import BaseMemory

logger = logging.getLogger("tinyflow.memory.vector")


class VectorMemory(BaseMemory):
    def __init__(self, vector_db: BaseVectorDB):
        self.db = vector_db

    async def add(self, text: str):
        logger.debug(f"Adding to vector memory: {text[:50]}...")
        await self.db.add([text])

    async def search(self, query: str, limit: int = 3) -> List[str]:
        logger.debug(f"Searching vector memory for: {query}")
        results = await self.db.search(query, limit=limit)
        return [r["text"] for r in results]

    async def clear(self):
        logger.info("Clearing vector memory")
        await self.db.clear()
