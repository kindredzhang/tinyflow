"""Workflow executor."""

import inspect
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from app.core.exceptions import WorkflowError
from app.workflow.hooks import StepContext, StepError, StepResult, WorkflowHooks

logger = logging.getLogger("tinyflow.workflow.executor")


class WorkflowExecutor:
    """Manual workflow execution engine."""

    def __init__(self, workflow_instance, hooks: Optional[WorkflowHooks] = None):
        self.workflow = workflow_instance
        self.state = getattr(workflow_instance, "state", None)
        self.graph = defaultdict(list)
        self.entry_points = []
        self.router_edges = {}
        self.hooks = hooks

    def _build_graph(self):
        """Build dependency graph from workflow steps."""
        self.graph.clear()
        self.entry_points.clear()
        self.router_edges.clear()
        self._name_to_method = {}  # Map decorator name to method name

        # Scan all methods
        for name in dir(self.workflow):
            if name.startswith("_"):
                continue

            method = getattr(self.workflow, name, None)
            if not callable(method):
                continue

            if hasattr(method, "_workflow_meta"):
                meta = method._workflow_meta
                decorator_name = meta.get("name", name)

                # Map decorator name to method name
                self._name_to_method[decorator_name] = name

                # Entry point (no dependencies)
                if not any(meta.get(k) for k in ["listen_to", "parallel_to"]):
                    self.entry_points.append(name)

                # Build dependency graph - use decorator name as key
                listen_to = meta.get("listen_to")
                if listen_to:
                    if isinstance(listen_to, list):
                        for source in listen_to:
                            if isinstance(source, str):
                                self.graph[source].append(name)
                            elif callable(source):
                                self.graph[source.__name__].append(name)
                    elif isinstance(listen_to, str):
                        self.graph[listen_to].append(name)
                    elif callable(listen_to):
                        self.graph[listen_to.__name__].append(name)

                parallel_to = meta.get("parallel_to")
                if parallel_to is True and listen_to:
                    if isinstance(listen_to, list):
                        for source in listen_to:
                            if isinstance(source, str):
                                self.graph[source].append(name)
                            elif callable(source):
                                self.graph[source.__name__].append(name)
                elif parallel_to and parallel_to is not True:
                    for source in parallel_to:
                        if isinstance(source, str):
                            self.graph[source].append(name)
                        elif callable(source):
                            self.graph[source.__name__].append(name)

                # Router
                if meta.get("router"):
                    self.router_edges[name] = meta["router"]

    async def run(self, initial_state: Optional[Dict[str, Any]] = None):
        """Execute the workflow."""
        logger.info("Starting workflow execution")
        self._build_graph()

        # Store initial_state for later use
        self._initial_state = initial_state or {}

        if initial_state and self.state:
            self.state.update(**initial_state)

        results = {}
        visited = set()

        # Execute entry points
        for entry in self.entry_points:
            await self._execute_step(entry, results, visited)

        return results

    async def _execute_step(self, step_name: str, results: Dict[str, Any], visited: set):
        """Execute a single step."""
        if step_name in visited:
            return

        method = getattr(self.workflow, step_name)
        meta = getattr(method, "_workflow_meta", {})

        # Check if dependencies are satisfied
        if not await self._check_dependencies(step_name, meta, results):
            logger.debug(f"Dependencies for step '{step_name}' not yet satisfied, skipping")
            return

        visited.add(step_name)
        logger.info(f"Executing step: {step_name}")

        step_result = await self._run_with_retry(method, meta, results, self._initial_state)
        decorator_name = meta.get("name", step_name)
        results[decorator_name] = step_result
        results[step_name] = step_result

        if decorator_name in self.graph:
            for next_step in self.graph[decorator_name]:
                await self._execute_step(next_step, results, visited)

        # Handle routing
        if step_name in self.router_edges:
            route_func = self.router_edges[step_name]
            route = route_func(step_result)
            logger.info(f"Router for step '{step_name}' returned route: {route}")
            if route in self.graph:
                for next_step in self.graph[route]:
                    await self._execute_step(next_step, results, visited)

    async def _check_dependencies(self, step_name: str, meta: Dict, results: Dict) -> bool:
        """Check if all dependencies are satisfied."""
        listen_to = meta.get("listen_to")
        parallel_to = meta.get("parallel_to")

        # Handle parallel_to=True case - wait for all listen_to sources
        if parallel_to is True and listen_to:
            if isinstance(listen_to, list):
                for source in listen_to:
                    if isinstance(source, str):
                        if source not in results:
                            return False
                    elif callable(source):
                        if source.__name__ not in results:
                            return False
            return True

        # Handle regular listen_to
        if listen_to:
            if isinstance(listen_to, list):
                for source in listen_to:
                    if isinstance(source, str):
                        if source not in results:
                            return False
                    elif callable(source):
                        if source.__name__ not in results:
                            return False
            elif isinstance(listen_to, str):
                return listen_to in results
            elif callable(listen_to):
                return listen_to.__name__ in results

        # Handle parallel_to as list
        if parallel_to and parallel_to is not True:
            for source in parallel_to:
                if isinstance(source, str):
                    if source not in results:
                        return False
                elif callable(source):
                    if source.__name__ not in results:
                        return False

        return True

    async def _run_with_retry(
        self, method: Callable, meta: Dict, results: Dict, initial_state: Dict
    ):
        """Run a method with retry logic."""
        retry_count = meta.get("retry", 0) or 0
        step_name = meta.get("name", method.__name__)
        workflow_id = self.state.id if self.state else None

        # Get arguments based on listen_to
        listen_to = meta.get("listen_to")
        parallel_to = meta.get("parallel_to")
        args = []

        if listen_to:
            if isinstance(listen_to, list):
                for source in listen_to:
                    if isinstance(source, str) and source in results:
                        args.append(results[source])
                    elif callable(source) and source.__name__ in results:
                        args.append(results[source.__name__])
            elif isinstance(listen_to, str):
                if listen_to in results:
                    args.append(results[listen_to])
            elif callable(listen_to):
                source_name = listen_to.__name__
                if source_name in results:
                    args.append(results[source_name])
        else:
            # Entry point - get argument from initial_state
            method_name = meta.get("name", method.__name__)
            if method_name in initial_state:
                args.append(initial_state[method_name])

        # Trigger start hooks
        if self.hooks:
            context = StepContext(
                step_name=step_name, workflow_id=workflow_id, args=tuple(args), kwargs={}
            )
            self.hooks.trigger_start_hooks(context)

        # Run with retry
        last_error = None
        start_time = datetime.now()

        for attempt in range(retry_count + 1):
            try:
                if attempt > 0:
                    logger.info(f"Retrying step '{step_name}' (attempt {attempt}/{retry_count})")

                if args:
                    result = method(*args)
                else:
                    result = method()

                # Handle async result
                if inspect.iscoroutine(result):
                    result = await result

                # Calculate duration
                duration_ms = (datetime.now() - start_time).total_seconds() * 1000

                # Trigger complete hooks
                if self.hooks:
                    step_result = StepResult(
                        step_name=step_name,
                        result=result,
                        workflow_id=workflow_id,
                        duration_ms=duration_ms,
                    )
                    self.hooks.trigger_complete_hooks(step_result)

                return result

            except Exception as e:
                last_error = e
                logger.error(f"Error executing step '{step_name}': {str(e)}")
                if attempt < retry_count:
                    continue

                # All retries failed - trigger error hooks
                if self.hooks:
                    step_error = StepError(step_name=step_name, error=e, workflow_id=workflow_id)
                    self.hooks.trigger_error_hooks(step_error)

        # All retries failed
        error_msg = f"Step '{step_name}' failed after {retry_count} retries"
        if last_error:
            error_msg += f": {str(last_error)}"
        raise WorkflowError(error_msg) from last_error
