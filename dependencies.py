from fastapi import Body, HTTPException
from schemas import RAGRequest
from graphs.graph import app
from langchain_core.messages import HumanMessage


async def get_generation(body: RAGRequest=Body(...)) -> dict:
    try:
        message = HumanMessage(content=body.question)
        generation = await app.ainvoke({"messages": [message]})

        return {
            "answer": generation.get("generation", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    

