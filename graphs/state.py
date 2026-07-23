from typing import Annotated
from typing import TypedDict

from langgraph.graph.message import AnyMessage, add_messages


class GraphState(TypedDict):
    """
    Represents the state of our graphs.

    Attributes:
        question: question
        generation: LLM generation
        acceptable: whether the answer is qualified to be acceptable
        messages: list of messages
    """

    question: str
    generation: str
    acceptable: bool
    messages: Annotated[list[AnyMessage], add_messages]
    loop_count: int
