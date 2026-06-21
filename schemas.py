from typing import Annotated, Literal, TypeAlias
from uuid import uuid4
import tiktoken
from loguru import logger
from datetime import datetime

from pydantic import (
    BaseModel,
    Field,
    computed_field,
    IPvAnyAddress,
    HttpUrl,
)

SupportedTextModels: TypeAlias = Literal["gpt-3.5", "gpt-4o"]
TokenCount = Annotated[int, Field(ge=0)]

def count_tokens(text: str | None) -> int:
    if text is None:
        logger.warning("Response is None. Assuming 0 tokens used")
        return 0
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(text))


class RAGRequest(BaseModel):
    prompt: str
    

class RephraseRequest(BaseModel):
    question: str
    chat_history: list[dict[str, str]]    


class RAGResponse(BaseModel):
    answer: str
    hallucination_grade: bool
    relavant_grade: bool
    num_orginial_documents: int