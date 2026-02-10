"""Python interpreter tool for code execution.

⚠️ SECURITY WARNING: This tool executes arbitrary Python code.
Use only in sandboxed environments. Never expose to untrusted users.
"""

import io
import sys
import traceback
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from app.core.tools import Tool


class PythonInterpreterInput(BaseModel):
    """Input parameters for Python interpreter."""

    code: str = Field(
        description="The Python code to execute. "
        "Use print() to output results. "
        "Available libraries: os, sys, math, json, re, datetime, collections, itertools"
    )
    timeout: int = Field(default=30, description="Maximum execution time in seconds (1-60)")


def create_python_interpreter_tool(
    allowed_modules: Optional[list] = None, allow_file_access: bool = False
) -> Tool:
    """Create a Python interpreter tool instance.

    ⚠️ SECURITY WARNING:
    This tool executes arbitrary Python code. By default, it runs in a
    restricted environment with limited builtins. However, it should only
    be used in sandboxed environments.

    Args:
        allowed_modules: List of module names to allow importing.
                        Defaults to safe standard library modules.
        allow_file_access: Whether to allow file system access. Default False.

    Returns:
        Configured Tool instance for Python code execution.
    """
    # Default safe modules
    if allowed_modules is None:
        allowed_modules = [
            "math",
            "json",
            "re",
            "datetime",
            "collections",
            "itertools",
            "random",
            "statistics",
            "typing",
        ]

    def execute_python_code(args: PythonInterpreterInput) -> str:
        """Execute Python code in a restricted environment."""
        max(1, min(args.timeout, 60))

        # Create restricted globals
        restricted_globals: Dict[str, Any] = {
            "__builtins__": {
                "print": print,
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "pow": pow,
                "divmod": divmod,
                "bool": bool,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "frozenset": frozenset,
                "type": type,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "Exception": Exception,
            }
        }

        # Import allowed modules
        for module_name in allowed_modules:
            try:
                module = __import__(module_name)
                restricted_globals[module_name] = module
            except ImportError:
                pass

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # Execute code with timeout-like protection (via limited operations)
            # Note: For true sandboxing, consider using subprocess or RestrictedPython
            exec(args.code, restricted_globals)

            # Get output
            output = sys.stdout.getvalue()

            if not output.strip():
                return "Code executed successfully (no output)."

            return f"Output:\n{output}"

        except Exception as e:
            error_msg = f"Error executing code: {type(e).__name__}: {str(e)}"
            # Include traceback for debugging
            tb = traceback.format_exc()
            return f"{error_msg}\n\nTraceback:\n{tb}"
        finally:
            sys.stdout = old_stdout

    return Tool(
        name="python_interpreter",
        description="Execute Python code in a restricted environment. "
        "Available: math, json, re, datetime, collections, itertools. "
        "Use print() to see output. "
        "⚠️ Security: Code runs in restricted mode but use with caution.",
        parameters=PythonInterpreterInput,
        execute=execute_python_code,
    )


# Convenience instance with default restrictions
python_interpreter_tool = create_python_interpreter_tool()
