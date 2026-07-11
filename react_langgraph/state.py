from typing import TypedDict, Annotated
from langgraph.graph.message import AnyMessage, add_messages

class ReactState(TypedDict):
    """
    Define the state schema for our hotel assistant.

    Attributes:
        messages: list of messages
    """

    messages: Annotated[list[AnyMessage], add_messages]
