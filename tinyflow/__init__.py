"""TinyFlow: A lightweight, provider-agnostic AI Agent framework.

Exposes the core components for building agents and workflows.
"""

from tinyflow.core.agent import Agent
from tinyflow.core.exceptions import (
    ConfigurationError,
    MemoryError,
    ProviderError,
    TinyFlowError,
    ToolError,
    VectorError,
    WorkflowError,
)
from tinyflow.core.message import convert_to_model_messages
from tinyflow.core.tools import Tool, tool
from tinyflow.core.types import Message, UIMessage
from tinyflow.embeddings.factory import EmbeddingFactory
from tinyflow.providers.base.factory import LLMFactory
from tinyflow.vector.factory import VectorDBFactory
from tinyflow.workflow.executor import WorkflowExecutor
from tinyflow.workflow.state import WorkflowState
from tinyflow.workflow.step import if_step, workflow_step

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
