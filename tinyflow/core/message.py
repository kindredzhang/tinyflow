from tinyflow.core.types import Message


def convert_to_model_messages(dict_messages: list[dict]) -> list[Message]:
    """Convert a list of dictionaries to a list of Message objects."""
    return [Message(**msg) for msg in dict_messages]
