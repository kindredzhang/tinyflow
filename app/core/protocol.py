import json
from typing import AsyncGenerator, Set

from app.core.types import (
    DataUIPart,
    ReasoningUIPart,
    TextUIPart,
    ToolUIPart,
    UIMessage,
)


class JSONStreamAdapter:
    """
    Adapts a UIMessage stream to a structured JSON delta stream.
    Each yield is a valid JSON string representing a specific part update.
    """

    def __init__(self):
        self._last_lengths = {}  # Track lengths per part type/id
        self._emitted_states: Set[str] = set()

    async def transform(self, message_stream: AsyncGenerator[UIMessage, None]) -> AsyncGenerator[str, None]:
        async for message in message_stream:
            if message.role != "assistant":
                continue

            for part in message.parts:
                # Use a combination of type and optional ID for tracking
                part_id = getattr(part, "tool_call_id", None) or getattr(part, "id", "default")
                track_key = f"{part.type}:{part_id}"
                
                # 1. Handle Text & Reasoning (Incremental Deltas)
                if isinstance(part, (TextUIPart, ReasoningUIPart)):
                    current_text = part.text
                    last_len = self._last_lengths.get(track_key, 0)
                    
                    if len(current_text) > last_len:
                        delta = current_text[last_len:]
                        yield json.dumps({
                            "type": f"{part.type}-delta",
                            "delta": delta,
                            "index": message.parts.index(part) # Help frontend locate the part
                        }) + "\n"
                        self._last_lengths[track_key] = len(current_text)

                # 2. Handle Tool Calls (State-based Events)
                elif isinstance(part, ToolUIPart):
                    state_key = f"tool:{part.tool_call_id}:{part.state}"
                    if state_key not in self._emitted_states:
                        payload = {
                            "type": "tool-call" if "input" in part.state.value else "tool-result",
                            "status": part.state.value,
                            "toolCallId": part.tool_call_id,
                            "toolName": part.tool_name,
                        }
                        if part.input is not None:
                            payload["input"] = part.input
                        if part.output is not None:
                            payload["output"] = part.output
                        if part.error_text:
                            payload["error"] = part.error_text
                            
                        yield json.dumps(payload) + "\n"
                        self._emitted_states.add(state_key)

                # 3. Handle Custom Data
                elif isinstance(part, DataUIPart):
                    if track_key not in self._emitted_states:
                        yield json.dumps({
                            "type": "data",
                            "id": part.id,
                            "data": part.data
                        }) + "\n"
                        self._emitted_states.add(track_key)

async def to_json_stream(message_stream: AsyncGenerator[UIMessage, None]) -> AsyncGenerator[str, None]:
    """Helper function for the most common DX: agent.stream() -> to_json_stream()"""
    adapter = JSONStreamAdapter()
    async for chunk in adapter.transform(message_stream):
        yield chunk
