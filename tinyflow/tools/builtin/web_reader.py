"""Web reader tool using Jina AI Reader API.

This tool extracts clean, readable content from web pages
without the need for complex scraping logic.
"""

import os
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from tinyflow.core.tools import Tool


class JinaReaderInput(BaseModel):
    """Input parameters for Jina Reader."""

    url: str = Field(description="The URL of the web page to read")
    extract_images: bool = Field(
        default=False, description="Whether to include image references in the output"
    )


def _get_jina_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Get Jina API key if provided."""
    return api_key or os.getenv("JINA_API_KEY")


def create_jina_reader_tool(api_key: Optional[str] = None) -> Tool:
    """Create a Jina Reader tool instance.

    Jina Reader converts any URL into clean, LLM-friendly Markdown.
    No API key required for basic usage, but a key increases rate limits.

    Args:
        api_key: Optional Jina API key for higher rate limits.

    Returns:
        Configured Tool instance for Jina Reader.
    """

    def execute_jina_reader(args: JinaReaderInput) -> str:
        """Fetch and convert web page to Markdown."""
        key = _get_jina_api_key(api_key)

        # Construct the Jina Reader API URL
        reader_url = f"https://r.jina.ai/{args.url}"

        headers = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        try:
            response = httpx.get(reader_url, headers=headers, timeout=30.0)
            response.raise_for_status()

            content = response.text

            # Add metadata
            result = f"Content from: {args.url}\n\n{content}"

            if args.extract_images:
                # Note: Jina Reader includes images in Markdown format by default
                result += "\n\n(Images have been extracted and included as Markdown references)"

            return result

        except httpx.TimeoutException:
            return f"Error: Request timed out while fetching {args.url}"
        except httpx.HTTPStatusError as e:
            return f"Error: Failed to fetch {args.url} - HTTP {e.response.status_code}"
        except Exception as e:
            return f"Error reading URL {args.url}: {str(e)}"

    return Tool(
        name="read_webpage",
        description="Read and extract content from any web page URL. "
        "Converts the page to clean Markdown format, removing ads and clutter. "
        "Perfect for research, documentation reading, and content analysis. "
        "No API key required for basic usage.",
        parameters=JinaReaderInput,
        execute=execute_jina_reader,
    )


# Convenience instance
jina_reader_tool = create_jina_reader_tool()
