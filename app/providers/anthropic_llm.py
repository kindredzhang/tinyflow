import logging
import os
from typing import AsyncGenerator, List, Optional

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.tools import Tool
from app.core.types import (
    LLMResponse,
    Message,
    StreamChunk,
    StreamPart,
    TextStreamDelta,
)
from app.providers.base.llm import BaseLLM

logger = logging.getLogger("tinyflow.providers.anthropic")


class AnthropicProvider(BaseLLM):
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 60.0,
    ):
        # Configuration precedence: Manual > Environment Variable > Error
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")
        self.model = (
            model
            or os.getenv("ANTHROPIC_MODEL")
            or os.getenv("LLM_MODEL")
            or "claude-3-opus-20240229"
        )
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY env var or pass api_key."
            )

        self.client = anthropic.AsyncAnthropic(
            base_url=base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(
        self, messages: List[Message], tools: Optional[List[Tool]] = None
    ) -> LLMResponse:
        logger.debug(f"Anthropic Generate call with {len(messages)} messages")
        system_prompt = ""
        claude_messages = []

        for m in messages:
            if m.role == "system":
                system_prompt += (m.content or "") + "\n"
            else:
                claude_messages.append({"role": m.role, "content": m.content})

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=claude_messages,
        )

        content_text = ""
        for block in response.content:
            if isinstance(block, anthropic.types.TextBlock):
                content_text += block.text

        logger.debug(f"Anthropic Generate response: {content_text[:50]}...")

        return LLMResponse(content=content_text, raw_response=response)

    def stream(
        self, messages: List[Message], tools: Optional[List[Tool]] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        system_prompt = ""
        claude_messages = []

        for m in messages:
            if m.role == "system":
                system_prompt += (m.content or "") + "\n"
            else:
                claude_messages.append({"role": m.role, "content": m.content})

        async def _stream():
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=claude_messages,
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield StreamChunk(content=text)

        return _stream()

    def stream_text(
        self, messages: List[Message], tools: Optional[List[Tool]] = None
    ) -> AsyncGenerator[StreamPart, None]:
        system_prompt = ""
        claude_messages = []

        for m in messages:
            if m.role == "system":
                system_prompt += (m.content or "") + "\n"
            else:
                claude_messages.append({"role": m.role, "content": m.content})

        async def _stream_text():
            async with self.client.messages.stream(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=claude_messages,
            ) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield TextStreamDelta(type="text", text=text)

        return _stream_text()
