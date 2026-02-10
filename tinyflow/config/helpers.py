"""Configuration helper utilities.

This module provides utility functions for handling configuration precedence
(parameter > environment variable > settings > default) and parameter filtering.
"""

import os
from typing import Any, Dict, Optional

from tinyflow.config.settings import settings


def get_config_value(
    param_value: Optional[Any],
    env_key: Optional[str] = None,
    settings_field: Optional[str] = None,
    default_value: Any = None,
) -> Any:
    """Get configuration value with precedence: param > env > settings > default.

    Args:
        param_value: Value explicitly passed as parameter (highest priority)
        env_key: Environment variable name (e.g. "LLM_PROVIDER")
        settings_field: Field name in settings object (e.g. "LLM_PROVIDER")
        default_value: Default value if no other source provides a value

    Returns:
        The resolved configuration value
    """
    if param_value is not None:
        return param_value

    # Use settings (which already loads .env) before raw os.getenv
    # Pydantic Settings handles .env loading better than raw os.getenv
    # if python-dotenv is not explicitly loaded in main.
    if settings_field and hasattr(settings, settings_field):
        settings_val = getattr(settings, settings_field)
        # Check for empty string specifically for API keys
        if settings_val is not None and settings_val != "":
            return settings_val

    if env_key:
        env_val = os.getenv(env_key)
        if env_val is not None and env_val != "":
            return env_val

    return default_value


def filter_none_kwargs(**kwargs) -> Dict[str, Any]:
    """Filter out None values from keyword arguments.

    Useful for preparing arguments to pass to a provider that has its own
    fallback logic for missing (None) parameters.

    Args:
        **kwargs: Keyword arguments to filter

    Returns:
        Dictionary containing only non-None values
    """
    return {k: v for k, v in kwargs.items() if v is not None}
