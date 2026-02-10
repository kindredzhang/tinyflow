from typing import List

from .base import BaseMemory


class SimpleMemory(BaseMemory):
    def __init__(self):
        self.memories: List[str] = []

    async def add(self, text: str):
        if text not in self.memories:
            self.memories.append(text)

    async def search(self, query: str, limit: int = 3) -> List[str]:
        if not self.memories:
            return []

        if len(self.memories) <= limit:
            return self.memories

        query_words = set(query.lower().split())

        scored_memories = []
        for i, text in enumerate(self.memories):
            text_words = set(text.lower().split())
            score = len(query_words.intersection(text_words))
            score += i / 1000000
            scored_memories.append((score, text))

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [text for score, text in scored_memories[:limit]]

    async def clear(self):
        self.memories = []
