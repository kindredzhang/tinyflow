import asyncio
import json
import logging
import uuid
from typing import AsyncGenerator, Callable, Dict, List, Optional, Union

from tinyflow.core.tools import Tool
from tinyflow.core.tools import tool as tool_decorator
from tinyflow.core.types import (
    Message,
    ReasoningUIPart,
    StepStartUIPart,
    TextPartState,
    TextUIPart,
    ToolPartState,
    ToolUIPart,
    UIMessage,
)
from tinyflow.memory import BaseMemory
from tinyflow.providers.base.llm import BaseLLM

logger = logging.getLogger("tinyflow.core.agent")


class Agent:
    def __init__(
        self,
        llm: BaseLLM,
        tools: Optional[List[Tool]] = None,
        system_prompt: str = "",
        max_history: int = 10,
        memory: Optional[BaseMemory] = None,
    ):
        self.llm = llm
        self.tools: Dict[str, Tool] = {}

        if tools:
            for t in tools:
                if isinstance(t, Tool):
                    self.tools[t.name] = t
                elif callable(t):
                    wrapped_tool = tool_decorator()(t)
                    self.tools[wrapped_tool.name] = wrapped_tool
                else:
                    raise TypeError(f"Tool must be Tool or Callable, got {type(t)}")

        self.history: List[Message] = []
        self.max_history = max_history
        self.system_prompt = system_prompt
        self.memory = memory
        if system_prompt:
            self.history.append(Message(role="system", content=system_prompt))

    def _get_context_messages(self, memories: Optional[List[str]] = None) -> List[Message]:
        if not self.history:
            return []

        system_msgs = [m for m in self.history if m.role == "system"]
        other_msgs = [m for m in self.history if m.role != "system"]

        if memories:
            memory_str = "\n".join([f"- {m}" for m in memories])
            memory_block = f"\n[CRITICAL CONTEXT MEMORY]\n{memory_str}\n[END OF CONTEXT MEMORY]"

            if system_msgs:
                first_system = system_msgs[0]
                new_content = (first_system.content or "") + "\n" + memory_block
                system_msgs[0] = Message(role="system", content=new_content)
            else:
                system_msgs = [
                    Message(
                        role="system",
                        content=f"You are a helpful assistant.{memory_block}",
                    )
                ]

        trimmed_history = other_msgs[-self.max_history :]
        return system_msgs + trimmed_history

    async def run(self, user_input: str, max_steps: int = 10):
        """
        Core loop: Thinking -> Acting -> Observing -> Thinking
        """
        self.history.append(Message(role="user", content=user_input))
        current_step = 0

        # Retrieve memories before conversation
        memories = None
        if self.memory:
            memories = await self.memory.search(user_input)

        while current_step < max_steps:
            current_step += 1
            try:
                context = self._get_context_messages(memories)
                response = await self.llm.generate(context, tools=list(self.tools.values()))

                self.history.append(
                    Message(
                        role="assistant",
                        content=response.content,
                        tool_calls=response.tool_calls,
                    )
                )

                if response.tool_calls:
                    logger.info(
                        f"Step {current_step}: LLM requested {len(response.tool_calls)} tool calls"
                    )
                    tasks = [self._execute_tool(tc) for tc in response.tool_calls]
                    tool_messages = await asyncio.gather(*tasks)
                    self.history.extend(tool_messages)
                else:
                    return response.content

            except Exception as e:
                logger.error(f"Error in agent loop at step {current_step}: {str(e)}")
                error_msg = f"An internal error occurred: {str(e)}. Please try to recover or summarize the current state."
                self.history.append(Message(role="system", content=error_msg))

                if current_step >= max_steps:
                    return f"Agent failed after {max_steps} steps due to: {str(e)}"

        return "Max steps reached."

    async def _execute_tool(self, tool_call) -> Message:
        func_name = tool_call["function"]["name"]
        func_args = tool_call["function"]["arguments"]
        call_id = tool_call["id"]

        logger.info(f"Executing tool: {func_name}")

        if func_name not in self.tools:
            content = f"Error: Tool '{func_name}' not found."
        else:
            try:
                args = json.loads(func_args)
                content = await asyncio.wait_for(
                    self.tools[func_name].execute(**args), timeout=30.0
                )
            except asyncio.TimeoutError:
                content = f"Error: Tool '{func_name}' execution timed out after 30s."
                logger.warning(content)
            except json.JSONDecodeError as e:
                content = f"Error: Invalid JSON arguments: {str(e)}."
            except Exception as e:
                content = f"Error executing tool '{func_name}': {str(e)}"
                logger.error(content)
                # We catch exception here to return it to LLM, but we could also wrap it
                # For now, keeping it as string message to LLM is better for recovery

        return Message(role="tool", content=content, tool_call_id=call_id, name=func_name)

    async def stream(
        self, user_input: str, max_steps: int = 10
    ) -> AsyncGenerator[UIMessage, None]:
        """Stream chat responses as UIMessage objects with rich content types.

        Yields UIMessage objects that can be directly rendered by frontend.
        Each message contains typed parts (text, reasoning, tool calls, etc.)
        with state information for streaming UI updates.
        """
        # Create and yield user message
        user_message = UIMessage(
            id=str(uuid.uuid4()),
            role="user",
            parts=[TextUIPart(type="text", text=user_input, state=TextPartState.DONE)],
        )
        self.history.append(Message(role="user", content=user_input))
        yield user_message

        # Create assistant message placeholder
        assistant_message = UIMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            parts=[],
        )

        current_step = 0
        memories = None
        tool_args_buffer: Dict[str, str] = {}
        if self.memory:
            memories = await self.memory.search(user_input)

        while current_step < max_steps:
            current_step += 1

            # Add step marker for multi-step execution
            if current_step > 1:
                assistant_message.parts.append(StepStartUIPart())
                yield assistant_message

            try:
                context = self._get_context_messages(memories)

                # Check if LLM supports stream_text
                if not hasattr(self.llm, "stream_text"):
                    raise NotImplementedError(
                        f"LLM {type(self.llm).__name__} does not support stream_text. "
                        "Use agent.run() for non-streaming execution."
                    )

                # Use new stream_text method
                logger.info(f"Step {current_step}: Starting stream_text from LLM")

                # Reset buffers for new turn
                tool_args_buffer = {}
                reasoning_buffer = ""

                async for part in self.llm.stream_text(context, tools=list(self.tools.values())):
                    if part.type == "text":
                        if assistant_message.parts:
                            last = assistant_message.parts[-1]
                            if (
                                isinstance(last, ReasoningUIPart)
                                and last.state == TextPartState.STREAMING
                            ):
                                last.state = TextPartState.DONE

                        if assistant_message.parts:
                            last = assistant_message.parts[-1]
                            if (
                                isinstance(last, TextUIPart)
                                and last.state == TextPartState.STREAMING
                            ):
                                last.text += part.text
                            else:
                                assistant_message.parts.append(
                                    TextUIPart(
                                        type="text",
                                        text=part.text,
                                        state=TextPartState.STREAMING,
                                    )
                                )
                        else:
                            assistant_message.parts.append(
                                TextUIPart(
                                    type="text",
                                    text=part.text,
                                    state=TextPartState.STREAMING,
                                )
                            )
                        yield assistant_message

                    elif part.type == "reasoning":
                        reasoning_buffer += part.text

                        if assistant_message.parts:
                            last = assistant_message.parts[-1]
                            if (
                                isinstance(last, ReasoningUIPart)
                                and last.state == TextPartState.STREAMING
                            ):
                                last.text += part.text
                            else:
                                assistant_message.parts.append(
                                    ReasoningUIPart(
                                        type="reasoning",
                                        text=part.text,
                                        state=TextPartState.STREAMING,
                                    )
                                )
                        else:
                            assistant_message.parts.append(
                                ReasoningUIPart(
                                    type="reasoning",
                                    text=part.text,
                                    state=TextPartState.STREAMING,
                                )
                            )
                        yield assistant_message

                    elif part.type == "tool-call-streaming-start":
                        if assistant_message.parts:
                            last = assistant_message.parts[-1]
                            if (
                                isinstance(last, (TextUIPart, ReasoningUIPart))
                                and last.state == TextPartState.STREAMING
                            ):
                                last.state = TextPartState.DONE

                        tool_args_buffer[part.tool_call_id] = ""
                        new_tool_part = ToolUIPart(
                            type=f"tool-{part.tool_name}",
                            tool_call_id=part.tool_call_id,
                            tool_name=part.tool_name,
                            state=ToolPartState.INPUT_STREAMING,
                        )
                        assistant_message.parts.append(new_tool_part)
                        yield assistant_message

                    elif part.type == "tool-call-delta":
                        tool_args_buffer[part.tool_call_id] = (
                            tool_args_buffer.get(part.tool_call_id, "") + part.args_text_delta
                        )

                    elif part.type == "tool-call":
                        for p in assistant_message.parts:
                            if isinstance(p, ToolUIPart) and p.tool_call_id == part.tool_call_id:
                                p.state = ToolPartState.INPUT_AVAILABLE
                                p.input = part.input
                                break
                        yield assistant_message

                for p in assistant_message.parts:
                    if (
                        isinstance(p, (TextUIPart, ReasoningUIPart))
                        and p.state == TextPartState.STREAMING
                    ):
                        p.state = TextPartState.DONE

                # Check for tool calls to execute
                tool_parts = [
                    p
                    for p in assistant_message.parts
                    if isinstance(p, ToolUIPart) and p.state == ToolPartState.INPUT_AVAILABLE
                ]

                if tool_parts:
                    # Add Assistant Message to History (CRITICAL for OpenAI validation)
                    text_content = ""
                    for p in assistant_message.parts:
                        if isinstance(p, TextUIPart):
                            text_content += p.text

                    tool_calls_payload = []
                    for p in tool_parts:
                        tool_calls_payload.append(
                            {
                                "id": p.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": p.tool_name,
                                    "arguments": json.dumps(p.input) if p.input else "{}",
                                },
                            }
                        )

                    assistant_msg_kwargs = {
                        "role": "assistant",
                        "content": text_content,
                        "tool_calls": tool_calls_payload,
                    }

                    if reasoning_buffer:
                        assistant_msg_kwargs["reasoning_content"] = reasoning_buffer

                    self.history.append(Message(**assistant_msg_kwargs))

                    # Execute tools
                    for tool_part in tool_parts:
                        tool_part.provider_executed = True

                        if tool_part.tool_name in self.tools:
                            try:
                                if tool_part.input is not None:
                                    result = await self.tools[tool_part.tool_name].execute(
                                        **tool_part.input
                                    )
                                else:
                                    result = await self.tools[tool_part.tool_name].execute()
                                tool_part.state = ToolPartState.OUTPUT_AVAILABLE
                                tool_part.output = result
                            except Exception as e:
                                tool_part.state = ToolPartState.OUTPUT_ERROR
                                tool_part.error_text = str(e)
                        else:
                            tool_part.state = ToolPartState.OUTPUT_ERROR
                            tool_part.error_text = f"Tool '{tool_part.tool_name}' not found"

                        yield assistant_message

                    # Add tool results to history
                    for tool_part in tool_parts:
                        self.history.append(
                            Message(
                                role="tool",
                                content=str(tool_part.output)
                                if tool_part.state == ToolPartState.OUTPUT_AVAILABLE
                                else tool_part.error_text or "Error",
                                tool_call_id=tool_part.tool_call_id,
                                name=tool_part.tool_name,
                            )
                        )

                    # Continue for next step
                    continue
                else:
                    text_content = ""
                    for p in assistant_message.parts:
                        if isinstance(p, TextUIPart):
                            text_content += p.text

                    if reasoning_buffer:
                        self.history.append(
                            Message(
                                role="assistant",
                                content=text_content,
                                reasoning_content=reasoning_buffer,
                            )
                        )
                    else:
                        self.history.append(Message(role="assistant", content=text_content))
                    yield assistant_message
                    return

            except Exception as e:
                logger.error(f"Error in stream loop at step {current_step}: {str(e)}")
                error_part = TextUIPart(
                    type="text",
                    text=f"\n[Error]: {str(e)}. Attempting to recover...",
                    state=TextPartState.DONE,
                )
                assistant_message.parts.append(error_part)
                yield assistant_message

                if current_step >= max_steps:
                    fatal_part = TextUIPart(
                        type="text",
                        text="\n[Fatal]: Maximum steps reached with errors.",
                        state=TextPartState.DONE,
                    )
                    assistant_message.parts.append(fatal_part)
                    yield assistant_message
                    return

    def use_llm(self, llm: BaseLLM) -> None:
        """Dynamically switch LLM Provider."""
        self.llm = llm

    def add_tool(self, tool_item: Union[Tool, Callable]) -> None:
        if isinstance(tool_item, Tool):
            self.tools[tool_item.name] = tool_item
        elif callable(tool_item):
            wrapped = tool_decorator()(tool_item)
            self.tools[wrapped.name] = wrapped
        else:
            raise TypeError(f"Tool must be Tool or Callable, got {type(tool_item)}")

    def add_tools(self, tools: List[Union[Tool, Callable]]) -> None:
        for t in tools:
            self.add_tool(t)

    def save_state(self, file_path: str):
        state = {"history": [m.model_dump() for m in self.history]}
        with open(file_path, "w") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self, file_path: str):
        import os

        from tinyflow.core.types import Message

        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                state = json.load(f)
                self.history = [Message(**m) for m in state.get("history", [])]
