# TinyFlow - Agent Development Guide

This document defines the standards and workflows for AI agents working on the TinyFlow codebase. Follow these guidelines strictly to maintain consistency and quality.

## 1. Project Overview & Core Tech

TinyFlow is a Python 3.12+ agent framework using `uv` for dependency management. It provides a provider-agnostic abstraction over LLMs with a focus on streaming UI components.

- **Core Path**: `tinyflow/` contains all application logic.
- **Tests Path**: `tests/` (create if missing). Use `pytest` and `pytest-asyncio`.
- **Key Libraries**:
  - **Pydantic V2**: Mandatory for data validation. Use `model_dump()` and `model_validate()`.
  - **Asyncio**: Every I/O operation MUST be async. Use `httpx` instead of `requests`.
  - **Uv**: Use `uv run` for all commands.

## 2. Build, Lint, & Test Commands

Run these commands from the project root.

### Dependency Management
```bash
uv sync                     # Sync all dependencies (creates .venv)
uv add <package>            # Add a runtime dependency
uv add --dev <package>      # Add a development dependency
```

### Testing (pytest)
```bash
uv run pytest               # Run all tests
uv run pytest tests/test_factories.py # Run specific file
uv run pytest tests/test_factories.py::TestLLMFactory::test_create # Single test
uv run pytest -v -s         # Verbose output with stdout
```

### Quality Control
```bash
uv run ruff check .         # Lint (checks imports, errors, naming)
uv run ruff check --fix .   # Auto-fix linting issues
uv run black .              # Formatting (88 chars line length)
uv run basedpyright         # Strict type checking (MANDATORY)
```

## 3. Architecture & Patterns

### 3.1. Factory Pattern
Providers (LLM, Embeddings, VectorDB) MUST be created via Factories.
- **Correct**: `llm = LLMFactory.create(provider="openai")`
- **Incorrect**: Directly instantiating `OpenAIProvider`.

### 3.2. Tool Implementation Guide
Tools are defined using the `@tool` decorator or `Tool` class.
- **Docstrings**: MUST use Google-style docstrings. Parameters and descriptions are automatically parsed to create Pydantic schemas for LLMs.
- **Decorator Usage**:
  ```python
  from tinyflow.core.tools import tool

  @tool()
  async def get_weather(city: str, unit: str = "celsius"):
      """Get weather info for a city.
      Args:
          city: Name of the city.
          unit: Temperature unit (celsius/fahrenheit).
      """
      return f"25° in {city}"
  ```

### 3.3. UI Message Streaming (Protocol)
The system uses a strict `UIMessage` protocol in `tinyflow/core/types.py`.
- **Parts**: `TextUIPart`, `ReasoningUIPart`, `ToolUIPart`, `StepStartUIPart`.
- **States**: `streaming`, `done`, `input-available`, `output-available`, `output-error`.
- **Agent Output**: The `Agent.stream()` method yields `UIMessage` objects, updating the `parts` list incrementally.

## 4. Code Style & Typing

- **Imports**: ALWAYS use absolute imports (e.g., `from tinyflow.core.agent import Agent`).
- **Typing**: Use strict type annotations for ALL function signatures. Prefer `typing` module (List, Dict, Optional).
- **Naming**: `PascalCase` for classes, `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants.
- **Private Members**: Use `_leading_underscore` for internal helper methods.
- **Error Handling**: Use specific exceptions from `tinyflow.core.exceptions`. NEVER use bare `except:`.
- **Logging**: Use the shared logger: `logger = logging.getLogger("tinyflow.<module>")`.

## 5. Development Workflow

1. **Analyze**: Identify the module (providers, core, vector, etc.).
2. **Setup**: Ensure `.venv` is synced with `uv sync`.
3. **Implement**: 
   - Follow Pydantic V2 patterns.
   - Ensure all I/O is non-blocking.
   - Maintain UI Streaming Protocol compatibility in `agent.py` or providers.
4. **Test**: Create `tests/test_<feature>.py` if it doesn't exist. Mock LLM/External API calls using `unittest.mock` or `pytest-mock`.
5. **Verify**: Run `ruff`, `black`, `basedpyright`, and `pytest` before finishing.

## 6. Common Pitfalls

- **Blocking Calls**: Never use `time.sleep()` or `requests`. Use `asyncio.sleep()` and `httpx`.
- **Mutable Defaults**: Never use `def func(a=[])`. Use `a: Optional[List] = None`.
- **Circular Imports**: Use `if TYPE_CHECKING:` blocks for complex type hints in `core/`.
- **Pydantic V1**: Do not use `model.dict()`. Use `model.model_dump()`.
- **Missing Rules**: If `.cursorrules` or `.github/copilot-instructions.md` exist, follow them alongside these.
