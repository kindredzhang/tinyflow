import os
from typing import AsyncGenerator, List, Optional

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from tinyflow.core.tools import Tool
from tinyflow.core.types import (
    LLMResponse,
    Message,
    StreamChunk,
    StreamPart,
    TextStreamDelta,
)
from tinyflow.providers.base.llm import BaseLLM


class GeminiProvider(BaseLLM):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        # Configuration precedence: Manual > Environment Variable > Error
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY")
        self.model = (
            model or os.getenv("GEMINI_MODEL") or os.getenv("LLM_MODEL") or "gemini-1.5-flash"
        )
        self.base_url = base_url  # Gemini supports custom endpoints via client options

        if not self.api_key:
            raise ValueError(
                "Gemini API key required. Set GEMINI_API_KEY env var or pass api_key."
            )

        self.client = genai.Client(
            api_key=self.api_key,
        )

    def _convert_messages(self, messages: List[Message]) -> List[types.Content]:
        """
        Gemini requires converting OpenAI-style messages to Google Content type
        Note: This is a simplified conversion; complex Tool Results require more refined handling
        """
        gemini_contents = []
        for m in messages:
            role = "user" if m.role in ["user", "tool"] else "model"
            if m.role == "system":
                # Gemini V1 SDK suggests passing system prompt via config's system_instruction
                # Temporarily doing simple merging here, or extracting separately during generation
                continue

            parts = [types.Part(text=m.content)] if m.content else []

            # Simple conversion logic; production needs to handle function_call conversion
            gemini_contents.append(types.Content(role=role, parts=parts))

        return gemini_contents

    def _get_system_prompt(self, messages: List[Message]) -> Optional[str]:
        for m in messages:
            if m.role == "system":
                return m.content
        return None

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(
        self, messages: List[Message], tools: Optional[List[Tool]] = None
    ) -> LLMResponse:
        contents = self._convert_messages(messages)
        system_instr = self._get_system_prompt(messages)

        # Gemini supports passing functions directly, SDK automatically parses Schema
        # But we already have Tool; for simplicity, momentarily disabling Tools for Gemini
        if tools:
            pass

        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=system_instr),
        )

        return LLMResponse(content=response.text, raw_response=response)

    def stream(
        self, messages: List[Message], tools: Optional[List[Tool]] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        contents = self._convert_messages(messages)
        system_instr = self._get_system_prompt(messages)

        async def _stream():
            # Call streaming interface
            response_stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system_instr),
            )

            async for chunk in response_stream:
                if chunk.text:
                    yield StreamChunk(content=chunk.text)

        return _stream()

    def stream_text(
        self, messages: List[Message], tools: Optional[List[Tool]] = None
    ) -> AsyncGenerator[StreamPart, None]:
        """Stream response with rich part types."""
        contents = self._convert_messages(messages)
        system_instr = self._get_system_prompt(messages)

        async def _stream_text():
            response_stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(system_instruction=system_instr),
            )

            async for chunk in response_stream:
                if chunk.text:
                    yield TextStreamDelta(type="text", text=chunk.text)

        return _stream_text()
