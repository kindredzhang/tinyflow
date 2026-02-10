import inspect
import logging
from typing import Any, Callable, Dict, Optional, Type

from docstring_parser import parse
from pydantic import BaseModel, Field, create_model

from app.core.exceptions import ToolError

logger = logging.getLogger("tinyflow.core.tools")


class Tool:
    """
    Explicit declarative tool definition (Vercel AI SDK style).

    This is the unified tool class of the framework, providing clear control:
    - name: Tool name
    - description: Detailed description (LLM will decide when to call based on this)
    - parameters: Pydantic model defining parameter structure
    - execute: Execution function, receiving the validated Pydantic model instance

    Example:
        ```python
        class WeatherInput(BaseModel):
            city: str = Field(description="City name")
            unit: str = Field(default="celsius", description="Temperature unit")

        weather_tool = Tool(
            name="get_weather",
            description="Get weather information for a specific city.",
            parameters=WeatherInput,
            execute=lambda args: f"{args.city} is sunny"
        )
        ```
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: Type[BaseModel],
        execute: Callable[[Any], Any],
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.execute_func = execute
        self.args_schema = parameters
        self.func = execute
        logger.debug(f"Tool initialized: {self.name}")

    async def execute(self, **kwargs) -> Any:
        """Execute the tool with automatic parameter validation."""
        try:
            validated_args = self.parameters.model_validate(kwargs)

            if inspect.iscoroutinefunction(self.execute_func):
                result = await self.execute_func(validated_args)
            else:
                result = self.execute_func(validated_args)

            # If result is an Agent or WorkflowExecutor, run it
            # This enables "Agent as a Tool" pattern
            if hasattr(result, "run"):
                if inspect.iscoroutinefunction(result.run):
                    # For Agents/Workflows, we might want to return the result directly
                    # or handle the run execution here.
                    # Simpler approach: execute .run() and return output
                    # For WorkflowExecutor, run() returns dict of results
                    # For Agent, run() returns str
                    output = await result.run(**kwargs)  # Pass args if compatible?
                    # Actually, usually the tool function prepares the agent/workflow
                    # and might run it with specific args derived from validated_args.
                    # If the tool function *returns* the runner instance, we run it.
                    # But usually the tool body does the running.
                    # Let's assume the tool body returns the FINAL result (str/dict).
                    # So no change needed here if tool body handles execution.

                    # BUT, if we want to support returning an AsyncGenerator (Streaming Workflow),
                    # we must NOT convert to str immediately.
                    pass

            return result
        except Exception as e:
            if hasattr(e, "errors"):
                logger.warning(f"Validation Error for tool '{self.name}': {str(e)}")
                # Raise ToolError for validation failure?
                # Or keep string return for LLM loop?
                # The execute method is typically called by Agent which wants a string.
                # However, for library users executing tools directly, an exception is better.
                # Let's wrap it in ToolError but format it friendly.
                raise ToolError(f"Validation Error: {str(e)}", tool_name=self.name) from e

            logger.error(f"Error executing tool '{self.name}': {str(e)}")
            raise ToolError(f"Execution Error: {str(e)}", tool_name=self.name) from e

    def to_openai_schema(self, strict: bool = False) -> Dict[str, Any]:
        """
        Convert to OpenAI tool format.

        Extract field descriptions from Pydantic model Field descriptions to ensure
        LLM accurately understands the purpose of each parameter.
        """
        schema = self.parameters.model_json_schema()

        parameters = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }

        if strict:
            parameters["additionalProperties"] = False
            parameters["required"] = list(schema.get("properties", {}).keys())

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
                "strict": strict,
            },
        }


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
):
    """
    Tool decorator supporting automatic tool definition derivation from functions.

    Supports two modes:
    1. Decorating normal functions: Automatically parse description and parameter descriptions from docstring.
    2. Decorating class methods: Use with Tool class.

    Example:
        ```python
        @tool()
        async def get_weather(city: str):
            '''Get weather.

            Args:
                city: City name
            '''
            return "Sunny"

        @tool(name="custom_name", description="Custom description")
        def my_tool(input: str):
            return input
        ```
    """

    def decorator(func: Callable) -> Tool:
        tool_name = name or func.__name__

        tool_description = description
        if tool_description is None:
            doc = parse(func.__doc__ or "")
            tool_description = doc.short_description or f"Tool: {tool_name}"

        # Create Pydantic model from function signature
        args_schema = _get_args_schema(func)

        # Create execute function that unwraps Pydantic model to kwargs
        def execute_wrapper(args):
            kwargs = args.model_dump()
            if inspect.iscoroutinefunction(func):
                # Return the coroutine - it will be awaited by Tool.execute
                return func(**kwargs)
            else:
                return func(**kwargs)

        return Tool(
            name=tool_name,
            description=tool_description,
            parameters=args_schema,
            execute=execute_wrapper,
        )

    return decorator


def _get_args_schema(func: Callable) -> Type[BaseModel]:
    """Helper function to create Pydantic model from function signature."""
    sig = inspect.signature(func)
    doc = parse(func.__doc__ or "")

    param_descriptions = {p.arg_name: p.description for p in doc.params}

    fields = {}
    for param_name, param in sig.parameters.items():
        annotation = param.annotation if param.annotation != inspect.Parameter.empty else str
        default = param.default if param.default != inspect.Parameter.empty else ...
        description = param_descriptions.get(param_name, "")
        fields[param_name] = (annotation, Field(default=default, description=description))

    return create_model(f"{func.__name__}Schema", **fields)
