from typing import TypedDict, Optional


# include web_search as an option to implement in the future
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    searched_web: bool
    documents: list[str]
    quality: dict