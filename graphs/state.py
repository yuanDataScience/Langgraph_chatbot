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
        web_search: whether to add search
        messages: list of messages
    """

    question: str
    generation: str
    web_search: bool
    messages: Annotated[list[AnyMessage], add_messages]