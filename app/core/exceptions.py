"""TinyFlow Exceptions

This module defines the public exception hierarchy for the TinyFlow framework.
Users can catch these exceptions to handle errors in a structured way.
"""

from typing import Any, Optional


class TinyFlowError(Exception):
    """Base class for all TinyFlow exceptions."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ConfigurationError(TinyFlowError):
    """Raised when configuration is missing or invalid."""

    pass


class ProviderError(TinyFlowError):
    """Raised when an LLM provider fails (API errors, rate limits, etc.)."""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        raw_response: Optional[Any] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.raw_response = raw_response


class ToolError(TinyFlowError):
    """Raised when a tool execution fails."""

    def __init__(self, message: str, tool_name: str):
        super().__init__(message)
        self.tool_name = tool_name


class WorkflowError(TinyFlowError):
    """Raised when workflow execution fails (dependencies, routing, etc.)."""

    pass


class VectorError(TinyFlowError):
    """Raised when vector database operations fail."""

    pass


class MemoryError(TinyFlowError):
    """Raised when memory operations fail."""

    pass
