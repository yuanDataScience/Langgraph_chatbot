from fastapi import Body, HTTPException
from schemas import RAGRequest
from graphs.graph import build_graph
from langchain_core.messages import HumanMessage


async def get_generation(body: RAGRequest=Body(...)) -> dict:
    try:
        message = HumanMessage(content=body.question)

        app = await build_graph()
        generation = await app.ainvoke({"messages": [message]})

        return {
            "answer": generation.get("generation", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    

