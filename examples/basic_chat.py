import asyncio
import logging
import json
from typing import Any, List

from pydantic import BaseModel, Field

from tinyflow.core.agent import Agent
from tinyflow.core.tools import Tool
from tinyflow.core.types import UIMessage, TextUIPart, ToolUIPart, ReasoningUIPart
from tinyflow.providers.base.factory import LLMFactory

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- 1. Define Tool Input Schema ---
class WeatherInput(BaseModel):
    location: str = Field(description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(default="celsius", description="Temperature unit")

# --- 2. Define Tool Execution Function ---
async def get_weather(args: WeatherInput) -> str:
    """Get the current weather for a location."""
    return f"The weather in {args.location} is 25°{args.unit.upper()} and sunny."

# --- 3. Create Tool Instance ---
weather_tool = Tool(
    name="get_weather",
    description="Get weather information for a specific location.",
    parameters=WeatherInput,
    execute=get_weather,
)

# --- 4. Setup LLM ---
import os
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY") or "sk-241c1a257fd1429eb18719dd47ae171f"
llm = LLMFactory.create(
    provider="deepseek", 
    model="deepseek-reasoner",
    api_key=deepseek_api_key
)

# --- 5. Create Agents ---
basic_agent = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant. Be concise.",
)

tool_agent = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant. Use tools when necessary.",
    tools=[weather_tool],
)

from tinyflow.core.protocol import to_json_stream

# --- 6. Helper for Stream Printing ---
async def print_stream(generator):
    """
    Intelligently print the streaming UIMessage which accumulates parts.
    """
    current_msg_id = None
    processed_parts = 0
    last_text_len = 0
    printed_tool_outputs = set()
    
    async for message in generator:
        if message.role == "user":
            continue
            
        if message.id != current_msg_id:
            if current_msg_id is not None:
                print()
            current_msg_id = message.id
            processed_parts = 0
            last_text_len = 0
            printed_tool_outputs = set()
            
        parts = message.parts
        
        if len(parts) > processed_parts:
            for i in range(processed_parts, len(parts)):
                part = parts[i]
                if isinstance(part, TextUIPart):
                    last_text_len = 0
                    if part.text:
                        print(part.text, end="", flush=True)
                        last_text_len = len(part.text)
                elif isinstance(part, ToolUIPart):
                    print(f"\n[Tool Call: {part.tool_name}]", end="", flush=True)
                elif isinstance(part, ReasoningUIPart):
                    # For basic print, we might just want to show reasoning in a distinct way
                    if part.text:
                        print(f"\n[Thinking]: {part.text}", end="", flush=True)
            processed_parts = len(parts)
            
        for i, part in enumerate(parts):
            if isinstance(part, ToolUIPart):
                if part.state == "output-available" and part.tool_call_id not in printed_tool_outputs:
                    print(f"\n[Tool Result]: {part.output}", end="", flush=True)
                    printed_tool_outputs.add(part.tool_call_id)
                elif part.state == "output-error" and part.tool_call_id not in printed_tool_outputs:
                    print(f"\n[Tool Error]: {part.error_text}", end="", flush=True)
                    printed_tool_outputs.add(part.tool_call_id)
            
            if i == len(parts) - 1 and isinstance(part, TextUIPart):
                if len(part.text) > last_text_len:
                    print(part.text[last_text_len:], end="", flush=True)
                    last_text_len = len(part.text)

    print() # Newline at end

# --- 7. Test Functions ---

async def run_basic_chat():
    print("\n" + "="*50)
    print("TEST 1: Basic Chat (No Tools)")
    print("="*50)
    user_input = "What is the capital of France?"
    print(f"User: {user_input}")
    
    response = await basic_agent.run(user_input)
    print(f"Assistant: {response}")

async def run_basic_stream():
    print("\n" + "="*50)
    print("TEST 2: Basic Stream (No Tools)")
    print("="*50)
    user_input = "Tell me a short joke about programming."
    print(f"User: {user_input}")
    print("Assistant (Streaming): ", end="", flush=True)
    
    await print_stream(basic_agent.stream(user_input))

async def run_tool_chat():
    print("\n" + "="*50)
    print("TEST 3: Chat with Tool Execution")
    print("="*50)
    user_input = "What's the weather like in Tokyo?"
    print(f"User: {user_input}")
    
    response = await tool_agent.run(user_input)
    print(f"Assistant: {response}")

async def run_tool_stream():
    print("\n" + "="*50)
    print("TEST 4: Stream with Tool Execution")
    print("="*50)
    user_input = "Check the weather in Paris."
    print(f"User: {user_input}")
    print("Assistant (Streaming): ", end="", flush=True)
    
    await print_stream(tool_agent.stream(user_input))

async def run_http_simulation():
    """
    Simulates how an HTTP server would stream the response.
    NEW DX: to_json_stream(agent.stream(input))
    """
    print("\n" + "="*50)
    print("TEST 5: HTTP Stream Simulation (Structured JSON DX)")
    print("="*50)
    user_input = "Check the weather in Paris."
    print(f"User: {user_input}")
    print("Response (Raw JSON Data Stream):")
    
    # NEW DX: Pure and simple
    async for delta_json in to_json_stream(tool_agent.stream(user_input)):
        print(f"data: {delta_json}", end="", flush=True)

    print()

async def main():
    # await run_basic_chat()
    # await run_basic_stream()
    # await run_tool_chat()
    # await run_tool_stream()
    await run_http_simulation()

if __name__ == '__main__':
    asyncio.run(main())
