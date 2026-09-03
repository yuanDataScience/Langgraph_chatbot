from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    Field,
)

SupportedTextModels: TypeAlias = Literal["gpt-3.5", "gpt-4o"]
TokenCount = Annotated[int, Field(ge=0)]


class RAGRequest(BaseModel):
    query: str


class RAGResponse(BaseModel):
    answer: str
    web_search: bool
