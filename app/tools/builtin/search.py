"""Built-in search tools using mainstream APIs.

This module provides search capabilities through:
- Tavily AI (high-quality AI search, requires API key)
- DuckDuckGo (free, no API key required)
"""

import os
from typing import Optional

from pydantic import BaseModel, Field

from app.core.tools import Tool


class TavilySearchInput(BaseModel):
    """Input parameters for Tavily search."""

    query: str = Field(description="The search query to execute")
    max_results: int = Field(default=5, description="Maximum number of results to return (1-10)")
    include_answer: bool = Field(
        default=True, description="Include an AI-generated answer summary"
    )


class DuckDuckGoSearchInput(BaseModel):
    """Input parameters for DuckDuckGo search."""

    query: str = Field(description="The search query to execute")
    max_results: int = Field(default=5, description="Maximum number of results to return")


def _get_tavily_client(api_key: Optional[str] = None):
    """Get or create Tavily client with the provided or environment API key."""
    try:
        from tavily import TavilyClient
    except ImportError:
        raise ImportError(
            "tavily-python is required for TavilySearchTool. "
            "Install with: pip install tavily-python"
        )

    # Configuration precedence: Manual > Environment Variable > Error
    key = api_key or os.getenv("TAVILY_API_KEY")
    if not key:
        raise ValueError(
            "Tavily API key required. Set TAVILY_API_KEY environment variable "
            "or pass api_key parameter to the tool."
        )

    return TavilyClient(api_key=key)


def create_tavily_search_tool(api_key: Optional[str] = None) -> Tool:
    """Create a Tavily search tool instance.

    Args:
        api_key: Tavily API key. If not provided, uses TAVILY_API_KEY env var.

    Returns:
        Configured Tool instance for Tavily search.
    """

    def execute_tavily_search(args: TavilySearchInput) -> str:
        """Execute Tavily search and format results."""
        client = _get_tavily_client(api_key)

        response = client.search(
            query=args.query,
            max_results=args.max_results,
            include_answer=args.include_answer,
        )

        # Format results
        results_text = f"Search results for: {args.query}\n\n"

        if args.include_answer and "answer" in response:
            results_text += f"AI Summary: {response['answer']}\n\n"

        results = response.get("results", [])
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            content = result.get("content", "No content")
            url = result.get("url", "")
            results_text += f"{i}. {title}\n{content}\nURL: {url}\n\n"

        return results_text.strip()

    return Tool(
        name="tavily_search",
        description="Search the web using Tavily AI for high-quality, relevant results. "
        "Best for research, fact-checking, and finding current information.",
        parameters=TavilySearchInput,
        execute=execute_tavily_search,
    )


def create_duckduckgo_search_tool() -> Tool:
    """Create a DuckDuckGo search tool instance.

    Note: DuckDuckGo search does not require an API key.

    Returns:
        Configured Tool instance for DuckDuckGo search.
    """

    def execute_duckduckgo_search(args: DuckDuckGoSearchInput) -> str:
        """Execute DuckDuckGo search and format results."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ImportError(
                "duckduckgo-search is required for DuckDuckGoSearchTool. "
                "Install with: pip install duckduckgo-search"
            )

        with DDGS() as ddgs:
            results = list(ddgs.text(args.query, max_results=args.max_results))

        # Format results
        results_text = f"Search results for: {args.query}\n\n"

        if not results:
            return f"No results found for: {args.query}"

        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            body = result.get("body", "No description")
            href = result.get("href", "")
            results_text += f"{i}. {title}\n{body}\nURL: {href}\n\n"

        return results_text.strip()

    return Tool(
        name="duckduckgo_search",
        description="Search the web using DuckDuckGo. Free, no API key required. "
        "Good for general web searches when Tavily is not available.",
        parameters=DuckDuckGoSearchInput,
        execute=execute_duckduckgo_search,
    )


# Convenience instances (will use env vars for API keys)
tavily_search_tool = create_tavily_search_tool()
duckduckgo_search_tool = create_duckduckgo_search_tool()
