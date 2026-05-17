# TinyFlow

<!-- markdownlint-disable-next-line -->
![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-%3E%3D3.12-3776AB?logo=python&logoColor=white)
![PyPI](https://img.shields.io/badge/PyPI-tinyflow--llm-1E40AF?logo=pypi)

A lightweight, extensible Python framework for building LLM agents. Designed for developers who want a clean abstraction layer over major LLM providers, with first-class support for tools, memory, RAG, and streaming UX.

## What it solves

TinyFlow gives you the building blocks for agentic workflows without locking you into a specific provider or vector store:

- **Provider agnostic** — switch between OpenAI, Anthropic, and Google Gemini by changing one line of config
- **Clean agent primitives** — `Thinking → Acting → Observing` loop, exposed through a minimal API
- **Tool authoring as a decorator** — `@tool` turns any async function into a tool the agent can call
- **Streaming by default** — structured JSON streaming so you can build real-time UIs with reasoning step visibility
- **Memory & RAG** — plug in ChromaDB or Qdrant with a few lines; embed with OpenAI or locally

## Quick Start

### Installation

```bash
# pip
pip install tinyflow-llm

# uv (recommended)
uv add tinyflow-llm

# Extras
pip install "tinyflow-llm[local]"       # sentence-transformers embeddings
pip install "tinyflow-llm[vector]"      # ChromaDB + Qdrant
```

### A complete agent in 30 lines

This is the canonical example from `examples/basic_chat.py`. It defines a tool, wires it to an agent, and runs a streaming conversation:

```python
import asyncio
from pydantic import BaseModel, Field

from tinyflow.core.agent import Agent
from tinyflow.core.tools import Tool
from tinyflow.core.protocol import to_json_stream
from tinyflow.providers.base.factory import LLMFactory


# 1. Define the tool's input schema
class WeatherInput(BaseModel):
    location: str = Field(description="City and state, e.g. San Francisco, CA")
    unit: str = Field(default="celsius", description="Temperature unit")


# 2. Implement the tool
async def get_weather(args: WeatherInput) -> str:
    """Get the current weather for a location."""
    return f"The weather in {args.location} is 25°C and sunny."


# 3. Register the tool
weather_tool = Tool(
    name="get_weather",
    description="Get weather information for a specific location.",
    parameters=WeatherInput,
    execute=get_weather,
)

# 4. Create agent
llm = LLMFactory.create(provider="deepseek", model="deepseek-chat")
agent = Agent(
    llm=llm,
    system_prompt="You are a helpful assistant. Use tools when necessary.",
    tools=[weather_tool],
)


# 5. Stream a conversation
async def main():
    # Non-streaming
    response = await agent.run("What's the weather in Tokyo?")
    print(response)

    # Streaming (for real-time UIs)
    async for delta in to_json_stream(agent.stream("What's the weather in Paris?")):
        print(f"data: {delta}", end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration

TinyFlow resolves config from three sources, in priority order:

1. **Explicit parameters** — passed to `LLMFactory.create()`
2. **Environment variables** — `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, etc.
3. **Settings / defaults** — fallbacks when the above are not set

```bash
# .env
LLM_PROVIDER=deepseek
LLM_API_KEY=sk-...
LLM_MODEL=deepseek-chat
```

| Variable | Description |
|---|---|
| `LLM_PROVIDER` | `openai`, `anthropic`, or `gemini` |
| `LLM_API_KEY` | API key for your provider |
| `LLM_MODEL` | Model name, e.g. `gpt-4o`, `claude-3-5-sonnet` |
| `LLM_BASE_URL` | Optional proxy base URL |
| `EMBEDDING_PROVIDER` | `openai`, `local`, or `sentence-transformers` |
| `VECTOR_DB_PROVIDER` | `chroma` or `qdrant` |

## Key Features

### Provider Agnostic

```python
# Switch provider by changing one argument
llm = LLMFactory.create(provider="anthropic", model="claude-3-5-sonnet")
```

### Tool Support

The `@tool` decorator wraps any async function. Pydantic validates inputs automatically:

```python
from tinyflow.core.tools import tool

@tool
def web_search(query: str, num_results: int = 5) -> str:
    """Search the web for information."""
    ...

agent = Agent(llm=llm, tools=[web_search])
```

### Memory & RAG

```python
from tinyflow.vector.factory import VectorDBFactory
from tinyflow.embeddings.factory import EmbeddingFactory
from tinyflow.memory.vector import VectorMemory

embedding = EmbeddingFactory.create()
vector_db = VectorDBFactory.create()
memory = VectorMemory(vector_db=vector_db, embedding_model=embedding)

agent = Agent(llm=llm, memory=memory)
```

### Streaming UI Support

`to_json_stream()` emits structured JSON deltas that frontend clients can consume to show real-time reasoning steps and tool execution:

```python
async for delta in to_json_stream(agent.stream(user_input)):
    # delta: {"role":"assistant","part":{"type":"text","text":"..."}}
    #        {"role":"assistant","part":{"type":"tool_call","tool_name":"get_weather",...}}
    #        {"role":"assistant","part":{"type":"tool_result","tool_name":"get_weather","output":"..."}}
    print(delta)
```

## Project Structure

```
tinyflow/
├── tinyflow/
│   ├── config/       # Unified configuration system
│   ├── core/         # Agent, Tool, Types, protocol
│   ├── providers/    # OpenAI, Anthropic, Gemini, DeepSeek
│   ├── embeddings/   # OpenAI + local (sentence-transformers)
│   ├── vector/       # ChromaDB, Qdrant adapters
│   └── memory/       # VectorMemory, session memory
├── examples/
│   ├── basic_chat.py       # Tool use + streaming (start here)
│   └── config_value.py     # Configuration resolution demo
└── pyproject.toml
```

## Development

```bash
uv sync

# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run ruff format .

# Type check
uv run pyright
```

## License

[MIT](LICENSE)