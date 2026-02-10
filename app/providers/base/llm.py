from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional

from app.core.tools import Tool
from app.core.types import LLMResponse, Message, StreamChunk, StreamPart


class BaseLLM(ABC):
    @abstractmethod
    async def generate(
        self, messages: List[Message], tools: Optional[List[Tool]] = None
    ) -> LLMResponse:
        """Generate a complete response from the LLM."""
        pass

    @abstractmethod
    def stream(
        self, messages: List[Message], tools: Optional[List[Tool]] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream response chunks (legacy method - use stream_text for rich types)."""
        pass

    @abstractmethod
    def stream_text(
        self, messages: List[Message], tools: Optional[List[Tool]] = None
    ) -> AsyncGenerator[StreamPart, None]:
        """Stream response with rich part types (text, reasoning, tool calls).

        Yields StreamPart objects that can be:
        - TextStreamDelta: Text content delta
        - ReasoningStreamDelta: Reasoning/thinking trace
        - ToolCallStreamStart: Tool call streaming begins
        - ToolCallStreamDelta: Tool call argument delta
        - ToolCallComplete: Complete tool call
        - ToolResultStream: Tool execution result
        """
        pass
