"""Workflow hooks system for debugging and monitoring."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, List, Optional

logger = logging.getLogger("tinyflow.workflow.hooks")


@dataclass
class StepContext:
    """Context information for a workflow step.

    Attributes:
        step_name: Name of the step
        workflow_id: ID of the workflow instance
        timestamp: When the context was created
        args: Arguments passed to the step
        kwargs: Keyword arguments passed to the step
    """

    step_name: str
    workflow_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    args: tuple = ()
    kwargs: Optional[dict] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class StepResult:
    """Result of a workflow step execution.

    Attributes:
        step_name: Name of the step
        result: The result returned by the step
        workflow_id: ID of the workflow instance
        timestamp: When the step completed
        duration_ms: Execution duration in milliseconds
    """

    step_name: str
    result: Any
    workflow_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    duration_ms: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class StepError:
    """Error information for a failed workflow step.

    Attributes:
        step_name: Name of the step
        error: The exception that was raised
        workflow_id: ID of the workflow instance
        timestamp: When the error occurred
    """

    step_name: str
    error: Exception
    workflow_id: Optional[str] = None
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


# Type aliases for hook callbacks
StartHook = Callable[[StepContext], None]
CompleteHook = Callable[[StepResult], None]
ErrorHook = Callable[[StepError], None]


class WorkflowHooks:
    """Execution hooks for debugging and monitoring workflow execution.

    This class provides a hook system that allows registering callbacks
    for workflow step lifecycle events: start, complete, and error.

    Example:
        hooks = WorkflowHooks()

        # Register a hook
        def on_step_start(context):
            logger.info(f"Starting step: {context.step_name}")

        hooks.register_start_hook(on_step_start)

        # Later, trigger hooks
        context = StepContext(step_name="my_step")
        hooks.trigger_start_hooks(context)
    """

    def __init__(self):
        """Initialize the hooks system with empty hook lists."""
        self._start_hooks: List[StartHook] = []
        self._complete_hooks: List[CompleteHook] = []
        self._error_hooks: List[ErrorHook] = []

    def register_start_hook(self, callback: StartHook) -> None:
        """Register a callback to be called when a step starts.

        Args:
            callback: Function to call with StepContext when step starts
        """
        self._start_hooks.append(callback)

    def register_complete_hook(self, callback: CompleteHook) -> None:
        """Register a callback to be called when a step completes.

        Args:
            callback: Function to call with StepResult when step completes
        """
        self._complete_hooks.append(callback)

    def register_error_hook(self, callback: ErrorHook) -> None:
        """Register a callback to be called when a step errors.

        Args:
            callback: Function to call with StepError when step errors
        """
        self._error_hooks.append(callback)

    def trigger_start_hooks(self, context: StepContext) -> None:
        """Trigger all registered start hooks.

        Args:
            context: StepContext with information about the starting step
        """
        for hook in self._start_hooks:
            try:
                hook(context)
            except Exception as e:
                # Hooks should not break execution, just log the error
                logger.warning(f"Start hook failed: {e}")

    def trigger_complete_hooks(self, result: StepResult) -> None:
        """Trigger all registered complete hooks.

        Args:
            result: StepResult with information about the completed step
        """
        for hook in self._complete_hooks:
            try:
                hook(result)
            except Exception as e:
                # Hooks should not break execution, just log the error
                logger.warning(f"Complete hook failed: {e}")

    def trigger_error_hooks(self, error: StepError) -> None:
        """Trigger all registered error hooks.

        Args:
            error: StepError with information about the failed step
        """
        for hook in self._error_hooks:
            try:
                hook(error)
            except Exception as e:
                # Hooks should not break execution, just log the error
                logger.warning(f"Error hook failed: {e}")

    def clear_all_hooks(self) -> None:
        """Clear all registered hooks."""
        self._start_hooks.clear()
        self._complete_hooks.clear()
        self._error_hooks.clear()

    def get_hook_counts(self) -> dict:
        """Get the number of registered hooks of each type.

        Returns:
            Dictionary with counts for start, complete, and error hooks
        """
        return {
            "start": len(self._start_hooks),
            "complete": len(self._complete_hooks),
            "error": len(self._error_hooks),
        }
