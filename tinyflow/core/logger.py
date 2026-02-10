import logging


def setup_logger():
    """Setup the main logger for the library with a NullHandler to prevent 'No handler found' warnings."""
    logger = logging.getLogger("tinyflow")
    # Only add NullHandler if no handlers are present to avoid duplication if called multiple times
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())
    return logger


logger = setup_logger()
