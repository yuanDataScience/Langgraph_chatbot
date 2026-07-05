from fastapi import Body, HTTPException
from schemas import RAGRequest
from graphs.graph import app


async def get_generation(body: RAGRequest=Body(...)) -> dict:
    try:
        generation = await app.ainvoke(body.dict())

        return {
            "answer": generation.get("generation", ""),
            "web_search": generation.get("web_search", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    

