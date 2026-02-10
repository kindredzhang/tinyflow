import abc
from typing import List


class BaseMemory(abc.ABC):
    @abc.abstractmethod
    async def add(self, text: str):
        pass

    @abc.abstractmethod
    async def search(self, query: str, limit: int = 3) -> List[str]:
        pass

    @abc.abstractmethod
    async def clear(self):
        pass
