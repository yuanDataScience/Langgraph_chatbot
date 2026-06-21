from fastapi import Body, HTTPException
from schemas import RAGRequest
from nodes import generate


async def get_generation(body: RAGRequest=Body(...)) -> dict:
    try:
        generation = await generate(body.prompt)

        return {
            "answer": generation.get("generation", ""),
            "num_original_documents": generation.get("num_original_documents", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    

