import logging
import os
from typing import AsyncGenerator, List, Optional

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.tools import Tool
from app.core.types import (
    LLMResponse,
    Message,
    ReasoningStreamDelta,
    StreamChunk,
    StreamPart,
    TextStreamDelta,
    ToolCallComplete,
    ToolCallStreamDelta,
    ToolCallStreamStart,
)
from app.providers.base.llm import BaseLLM

logger = logging.getLogger("tinyflow.providers.openai")


class OpenAIProvider(BaseLLM):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 60.0,
    ):
        # Configuration precedence: Manual > Environment Variable > Error
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LLM_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o"
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL")
        self.timeout = timeout

        logging.info(
            f"OpenAIProvider Initialized: model='{self.model}', base_url='{self.base_url}'"
        )

        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key."
            )

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    def _convert_messages(self, messages: List[Message]):
        openai_msgs = []
        for m in messages:
            msg = {"role": m.role, "content": m.content}
            if m.tool_calls:
                msg["tool_calls"] = m.tool_calls
            if m.tool_call_id:
                msg["tool_call_id"] = m.tool_call_id
            if m.name:
                msg["name"] = m.name

            if m.reasoning_content:
                msg["reasoning_content"] = m.reasoning_content

            openai_msgs.append(msg)
        return openai_msgs

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    async def generate(
        self, messages: List[Message], tools: Optional[List[Tool]] = None
    ) -> LLMResponse:
        logger.debug(f"OpenAI Generate call with {len(messages)} messages")
        openai_messages = self._convert_messages(messages)
        openai_tools = [t.to_openai_schema(strict=True) for t in tools] if tools else None

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            tools=openai_tools,  # pyright: ignore[reportArgumentType]
        )
        msg = response.choices[0].message
        logger.debug(
            f"OpenAI Generate response: {msg.content[:50] if msg.content else 'None'}..."
        )

        # Process tool_calls format
        tool_calls_dict = None
        if msg.tool_calls:
            tool_calls_dict = [
                {
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                    "type": "function",
                }
                for tc in msg.tool_calls
            ]

        return LLMResponse(content=msg.content, tool_calls=tool_calls_dict, raw_response=response)

    def stream(
        self, messages: List[Message], tools: Optional[List[Tool]] = None
    ) -> AsyncGenerator[StreamChunk, None]:
        openai_messages = self._convert_messages(messages)

        async def _stream():
            stream = await self.client.chat.completions.create(
                model=self.model, messages=openai_messages, stream=True
            )

            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    yield StreamChunk(content=delta.content)

        return _stream()

    def stream_text(
        self, messages: List[Message], tools: Optional[List[Tool]] = None
    ) -> AsyncGenerator[StreamPart, None]:
        openai_messages = self._convert_messages(messages)
        openai_tools = [t.to_openai_schema(strict=True) for t in tools] if tools else None

        async def _stream_text():
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                tools=openai_tools,  # type: ignore[reportCallIssue]
                stream=True,
            )

            current_tool_call: Optional[dict] = None

            async for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta

                if delta.content:
                    yield TextStreamDelta(type="text", text=delta.content)

                # Support DeepSeek/o1 "reasoning_content"
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    yield ReasoningStreamDelta(type="reasoning", text=delta.reasoning_content)

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if tc_delta.id:
                            current_tool_call = {
                                "id": tc_delta.id,
                                "name": tc_delta.function.name,
                                "args": tc_delta.function.arguments or "",
                            }
                            yield ToolCallStreamStart(
                                type="tool-call-streaming-start",
                                tool_call_id=current_tool_call["id"],
                                tool_name=current_tool_call["name"],
                            )
                        else:
                            if current_tool_call:
                                current_tool_call["args"] += tc_delta.function.arguments or ""
                                yield ToolCallStreamDelta(
                                    type="tool-call-delta",
                                    tool_call_id=current_tool_call["id"],
                                    tool_name=current_tool_call["name"],
                                    args_text_delta=tc_delta.function.arguments or "",
                                )

                if chunk.choices[0].finish_reason:
                    if current_tool_call:
                        import json

                        try:
                            args = json.loads(current_tool_call["args"])
                        except json.JSONDecodeError:
                            args = current_tool_call["args"]

                        yield ToolCallComplete(
                            type="tool-call",
                            tool_call_id=current_tool_call["id"],
                            tool_name=current_tool_call["name"],
                            input=args,
                        )
                        current_tool_call = None

        return _stream_text()
