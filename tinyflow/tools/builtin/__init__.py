"""Built-in tools for TinyFlow Agent framework.

This package provides ready-to-use tools for common Agent tasks:
- Search: Tavily AI and DuckDuckGo
- Web Reading: Jina AI Reader
- Code Execution: Python Interpreter (restricted)
"""

from tinyflow.tools.builtin.code_execution import (
    create_python_interpreter_tool,
    python_interpreter_tool,
)
from tinyflow.tools.builtin.search import (
    create_duckduckgo_search_tool,
    create_tavily_search_tool,
    duckduckgo_search_tool,
    tavily_search_tool,
)
from tinyflow.tools.builtin.web_reader import (
    create_jina_reader_tool,
    jina_reader_tool,
)

__all__ = [
    # Search tools
    "create_tavily_search_tool",
    "create_duckduckgo_search_tool",
    "tavily_search_tool",
    "duckduckgo_search_tool",
    # Web reader tools
    "create_jina_reader_tool",
    "jina_reader_tool",
    # Code execution
    "create_python_interpreter_tool",
    "python_interpreter_tool",
]
