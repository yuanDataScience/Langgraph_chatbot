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

class RAGRequest(BaseModel):
    prompt: str
    

class RAGResponse(BaseModel):
    answer: str
    num_original_documents: int