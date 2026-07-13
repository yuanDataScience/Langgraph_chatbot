from typing import TypedDict, Annotated
from langgraph.graph.message import AnyMessage, add_messages

from typing import TypedDict
from langchain_core.documents import Document


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