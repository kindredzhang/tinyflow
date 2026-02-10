"""Workflow step decorator."""
from typing import Any, Callable, Optional, Union, List
from functools import wraps


def workflow_step(
    func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    listen_to: Optional[Union[str, List[Union[str, Callable]], Callable]] = None,
    parallel_to: Optional[Union[bool, List[Union[str, Callable]]]] = None,
    parallel_wait_for: Optional[List[str]] = None,
    router: Optional[Callable[[Any], str]] = None,
    retry: Optional[int] = None,
    condition: Optional[Callable[[], bool]] = None,
):
    """Decorator for defining workflow steps.
    
    Args:
        name: Optional step name
        listen_to: Single source dependency (str or list)
        parallel_to: List of sources to wait for, or True to wait for all in listen_to
        parallel_wait_for: Clearer alternative to parallel_to - list of step names to wait for
        router: Conditional routing function
        retry: Number of retries on failure
        condition: Execution condition function
    """
    def decorator(actual_func: Callable):
        @wraps(actual_func)
        def wrapper(self, *args, **kwargs):
            return actual_func(self, *args, **kwargs)
        
        wrapper._workflow_meta = {
            "name": name or actual_func.__name__,
            "listen_to": listen_to,
            "parallel_to": parallel_to,
            "parallel_wait_for": parallel_wait_for,
            "router": router,
            "retry": retry,
            "condition": condition,
        }
        
        return wrapper
    
    if func is not None:
        return decorator(func)
    return decorator


def if_step(condition: Callable[[], bool], on_true: str, on_false: Optional[str] = None) -> Callable:
    """Create a conditional router for workflow steps.
    
    This function creates a router that selects a path based on a condition.
    
    Args:
        condition: A callable that returns True or False
        on_true: The step name to route to if condition is True
        on_false: The step name to route to if condition is False (optional)
        
    Returns:
        A router function for use with workflow_step decorator
        
    Example:
        @workflow_step(
            name='classify',
            router=if_step(lambda: state.get('type') == 'error', 'error_handler', 'success_handler')
        )
        def classify_step(self, data):
            return {'type': 'error'}
    """
    def router(result: Any) -> str:
        return on_true if condition() else (on_false or "")
    
    return router
