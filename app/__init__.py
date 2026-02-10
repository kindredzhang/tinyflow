"""TinyFlow: A lightweight, provider-agnostic AI Agent framework.

Exposes the core components for building agents and workflows.
"""

from app.core.agent import Agent
from app.core.exceptions import (
    ConfigurationError,
    MemoryError,
    ProviderError,
    TinyFlowError,
    ToolError,
    VectorError,
    WorkflowError,
)
from app.core.message import convert_to_model_messages
from app.core.tools import Tool, tool
from app.core.types import Message, UIMessage
from app.embeddings.factory import EmbeddingFactory
from app.providers.base.factory import LLMFactory
from app.vector.factory import VectorDBFactory
from app.workflow.executor import WorkflowExecutor
from app.workflow.state import WorkflowState
from app.workflow.step import if_step, workflow_step

__version__ = "0.1.0"

__all__ = [
    # Core
    "Agent",
    "Message",
    "UIMessage",
    "Tool",
    "tool",
    "convert_to_model_messages",
    # Exceptions
    "TinyFlowError",
    "ConfigurationError",
    "ProviderError",
    "ToolError",
    "WorkflowError",
    "MemoryError",
    "VectorError",
    # Factories
    "LLMFactory",
    "EmbeddingFactory",
    "VectorDBFactory",
    # Workflow
    "WorkflowExecutor",
    "WorkflowState",
    "workflow_step",
    "if_step",
]
