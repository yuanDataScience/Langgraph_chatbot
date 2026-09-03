from fastapi import (
    FastAPI,
    Depends
)

from dependencies import get_generation
from schemas import RAGResponse

app = FastAPI()


@app.post("/generate_text", response_model=RAGResponse)
async def query_by_RAG_controller(generation: dict = Depends(get_generation)) -> RAGResponse:
    return RAGResponse(**generation)
