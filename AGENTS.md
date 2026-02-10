# TinyFlow - Agent Development Guide

This document outlines the standards and workflows for AI agents working on the TinyFlow codebase.
Follow these guidelines strictly to maintain code quality and consistency.

## 1. Project Overview

TinyFlow is a Python 3.12+ agent framework using `uv` for dependency management.
It provides a provider-agnostic abstraction over LLMs (OpenAI, Anthropic, Gemini) with a focus on streaming UI components.

- **Core Path**: `app/` contains all application logic.
- **Tests Path**: `tests/` (create if missing).
- **Key Libraries**:
  - **Pydantic V2**: For data validation and serialization (use `model_dump()`).
  - **Asyncio**: Core concurrency model.
  - **Uv**: Dependency management.

## 2. Build, Lint, & Test Commands

Use `uv` for all environment interactions. Do not use `pip` or `python` directly unless running through `uv`.

### Dependency Management

```bash
uv sync                     # Install/sync all dependencies (creates .venv)
uv add <package>            # Add a runtime package
uv add --dev <package>      # Add a development package
```

### Testing (pytest)

Tests must be placed in `tests/`. If the directory is missing, create it.

```bash
# Run all tests
uv run pytest

# Run a specific test file
uv run pytest tests/test_factories.py

# Run a single test case (CRITICAL for focused debugging)
uv run pytest tests/test_factories.py::TestLLMFactory::test_create_model_override

# Run with verbose output
uv run pytest -v -s
```

### Linting & Formatting

Run these before finishing any task.

```bash
# Lint code (Ruff) - checks for errors and imports
uv run ruff check .

# Fix auto-fixable lint errors
uv run ruff check --fix .

# Format code (Black)
uv run black .

# Type checking (basedpyright) - Strict typing compliance
uv run basedpyright
```

## 3. Code Style Guidelines

### General

- **Language**: Python 3.12+ features (type aliases `type X = ...` are allowed but `typing` module preferred for compatibility).
- **Docstrings**: Google-style docstrings for complex functions.
- **Imports**: Absolute imports from `app`.
  - Correct: `from app.core.agent import Agent`
  - Incorrect: `from ..core.agent import Agent`
  - **Sort Order**: Standard Library > Third Party > Local Application (`app`).

### Naming Conventions

- **Classes**: `PascalCase` (`LLMFactory`, `OpenAIProvider`, `UIMessage`).
- **Functions/Methods**: `snake_case` (`execute_tool`, `get_context`).
- **Variables**: `snake_case` (`user_input`, `max_history`).
- **Constants**: `UPPER_SNAKE_CASE` (`DEFAULT_TIMEOUT`, `MAX_RETRIES`).
- **Private Members**: `_leading_underscore` (`_execute_tool`, `_get_context_messages`).

### Type Annotations (Strict)

- **Mandatory**: All function signatures (args and return) must be typed.
- **Generics**: Use `typing.List`, `typing.Dict`, `typing.Optional`, `typing.Union`, `typing.AsyncGenerator`.
- **Pydantic**: Use Pydantic models for complex data structures.
  - **V2 Semantics**: Use `model.model_dump()` instead of `model.dict()`.
  - Use `Field(..., description="...")` for clear schema documentation.

### Error Handling

- **Specific Exceptions**: Catch specific errors (`ValueError`, `json.JSONDecodeError`).
- **Avoid Bare Except**: Never use `except:` or `except Exception:` without logging/re-raising.
- **Logging**: Use the shared logger.
  ```python
  import logging
  logger = logging.getLogger("tinyflow")
  logger.error(f"Context: {str(e)}")
  ```

### Asynchronous Programming

- Use `async def` for all I/O bound operations (LLM calls, database queries).
- Use `await` properly; avoid `asyncio.run()` inside library code (only in entry points like `main.py`).
- Use `asyncio.gather()` for parallel execution where appropriate (e.g., executing multiple tool calls).

## 4. Architecture & Patterns

### 4.1. Factory Pattern

Providers (LLM, Embeddings, VectorDB) are instantiated via Factories.

- **Usage**: `LLMFactory.create(provider="openai")`
- **Anti-Pattern**: `OpenAIProvider(api_key=...)` directly.

### 4.2. UI Message Streaming (Protocol)

The system uses a strict `UIMessage` protocol for frontend rendering.

- **Location**: `app/core/types.py`
- **Structure**: `UIMessage` contains a list of `UIMessagePart`s.
- **Parts**: `TextUIPart`, `ReasoningUIPart`, `ToolUIPart`, `StepStartUIPart`.
- **State**: Parts have states (`streaming`, `done`, `input-available`).
- **Agent Output**: Agents must yield `UIMessage` objects or update existing parts in-place during streaming.

### 4.3. Configuration

- Use `app.config.helpers.get_config_value` for resolving settings.
- Precedence: Explicit Arguments > Environment Variables > Defaults.

## 5. Development Workflow for Agents

1.  **Analyze**: Read the task and identify which `app/` modules are involved.
2.  **Verify Structure**: Check if `tests/` exists; if not, create it for your new tests.
3.  **Implement**:
    - Add new features in `app/`.
    - Use `pydantic` models for data passing.
    - Ensure `async/await` is used correctly.
4.  **Test**:
    - Create a test file `tests/test_<feature>.py`.
    - Use `pytest.mark.asyncio` for async tests.
    - Mock external LLM calls where possible to avoid API costs/latency during testing.
5.  **Quality Check**:
    - Run `uv run ruff check .`
    - Run `uv run basedpyright` to ensure type safety.
    - Run `uv run pytest` to ensure no regressions.

## 6. Common Pitfalls to Avoid

- **Blocking Code**: Do not use blocking I/O (e.g., `requests`, `time.sleep`) in async functions. Use `httpx` and `asyncio.sleep`.
- **Mutable Defaults**: Do not use mutable default arguments (e.g., `def func(items=[])`).
- **Circular Imports**: Be careful with imports in `app/core`. Use `TYPE_CHECKING` blocks if necessary.
