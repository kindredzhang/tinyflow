# TinyFlow

**TinyFlow** is a lightweight, extensible, and provider-agnostic Python framework for building LLM agents. It provides a clean abstraction layer over major LLM providers (OpenAI, Anthropic, Gemini), embedding models, and vector databases, enabling you to build complex agentic workflows with ease.

## 🚀 Key Features

- **Provider Agnostic**: Switch seamlessly between OpenAI, Anthropic, and Google Gemini models using a unified `LLMFactory`.
- **Agentic Workflow**: Built-in "Thinking -> Acting -> Observing" loop for autonomous task execution.
- **Tool Support**: Easy-to-use `@tool` decorator to give your agents capabilities (web search, code execution, etc.).
- **Memory & RAG**: Integrated Vector Database support (ChromaDB, Qdrant) and Embedding abstraction (OpenAI, SentenceTransformers).
- **Unified Configuration**: flexible configuration system supporting parameters, environment variables, and settings files with clear precedence.
- **Streaming UI**: Rich streaming support for building interactive chat interfaces, including reasoning steps and tool execution visibility.
- **Modern Stack**: Built with Python 3.12+, Pydantic, Asyncio, and managed with `uv`.

## 📦 Installation

You can install TinyFlow using `pip` or `uv`.

### Using pip

```bash
pip install tinyflow-llm
```

### Using uv (Recommended)

```bash
uv add tinyflow-llm
```

### Installation with Extras

TinyFlow supports optional dependencies for specific features:

- **Local Embeddings**: `pip install "tinyflow-llm[local]"`
- **Vector Databases**: `pip install "tinyflow-llm[vector]"` (includes ChromaDB and Qdrant)

## ⚙️ Configuration

TinyFlow uses a unified configuration system. You can configure providers via:

1.  **Explicit Parameters** (passed to Factory `create` methods) - _Highest Priority_
2.  **Environment Variables** - _Medium Priority_
3.  **Settings / Defaults** - _Lowest Priority_

### Common Environment Variables

| Category       | Variable             | Description                                      |
| -------------- | -------------------- | ------------------------------------------------ |
| **LLM**        | `LLM_PROVIDER`       | `openai`, `anthropic`, or `gemini`               |
|                | `LLM_API_KEY`        | API Key for the selected provider                |
|                | `LLM_MODEL`          | Model name (e.g., `gpt-4o`, `claude-3-5-sonnet`) |
|                | `LLM_BASE_URL`       | Optional custom base URL (e.g., for proxies)     |
| **Embeddings** | `EMBEDDING_PROVIDER` | `openai`, `local`, or `sentence-transformers`    |
|                | `EMBEDDING_API_KEY`  | API Key for embedding provider                   |
|                | `EMBEDDING_MODEL`    | Model name (e.g., `text-embedding-3-small`)      |
| **Vector DB**  | `VECTOR_DB_PROVIDER` | `chroma` or `qdrant`                             |
|                | `VECTOR_DB_PATH`     | Path for local ChromaDB persistence              |
|                | `VECTOR_DB_URL`      | URL for remote Vector DB (e.g., Qdrant Cloud)    |
|                | `VECTOR_DB_API_KEY`  | API Key for remote Vector DB                     |

## 💡 Usage

### 1. Basic LLM Usage

Use the `LLMFactory` to create a provider instance. It automatically handles configuration.

```python
import asyncio
from tinyflow.providers.base.factory import LLMFactory
from tinyflow.core.types import Message

async def main():
    # Automatically loads config from env vars
    llm = LLMFactory.create()

    # Or specify explicitly
    # llm = LLMFactory.create(provider="anthropic", model="claude-3-opus-20240229")

    messages = [
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="Explain quantum computing in one sentence.")
    ]

    response = await llm.generate(messages)
    print(response.content)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. Building an Agent with Tools

Create an `Agent` equipped with custom tools.

```python
import asyncio
from tinyflow.core.agent import Agent
from tinyflow.providers.base.factory import LLMFactory
from tinyflow.core.tools import tool

# Define a tool
@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location."""
    # In a real app, call a weather API here
    return f"The weather in {location} is 25°{unit.upper()} and sunny."

async def main():
    # 1. Initialize LLM
    llm = LLMFactory.create()

    # 2. Create Agent with tools
    agent = Agent(
        llm=llm,
        tools=[get_weather],
        system_prompt="You are a helpful assistant with access to weather tools."
    )

    # 3. Run Agent
    response = await agent.run("What's the weather like in Paris?")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Using Vector Memory

Integrate RAG (Retrieval-Augmented Generation) capabilities.

```python
from tinyflow.vector.factory import VectorDBFactory
from tinyflow.embeddings.factory import EmbeddingFactory
from tinyflow.memory.vector import VectorMemory

# Initialize components
embedding_model = EmbeddingFactory.create()
vector_db = VectorDBFactory.create()

# Create memory interface
memory = VectorMemory(
    vector_db=vector_db,
    embedding_model=embedding_model
)

# Use in Agent
agent = Agent(
    llm=llm,
    memory=memory,
    system_prompt="Use your memory to answer questions."
)
```

## 🏗️ Project Structure

```
tinyflow/
├── tinyflow/
│   ├── config/       # Configuration and helper utilities
│   ├── core/         # Core abstractions (Agent, Tools, Types)
│   ├── providers/    # LLM Provider implementations (OpenAI, Anthropic, Gemini)
│   ├── embeddings/   # Embedding models (OpenAI, Local)
│   ├── vector/       # Vector Database adapters (Chroma, Qdrant)
│   └── memory/       # Memory implementations
├── tests/            # Unit and integration tests
├── main.py           # Entry point example
└── pyproject.toml    # Project dependencies and config
```

## 🧪 Development

### Running Tests

TinyFlow uses `pytest` for testing.

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_factories.py -v
```

### Code Style

The project uses `ruff` for linting and formatting.

```bash
# Lint
uv run ruff check .

# Format
uv run ruff format .
```

## 📄 License

[MIT License](LICENSE)
