from fastapi import Body, HTTPException

from graphs.multi_agent_supervisor import supervisor
from schemas import RAGRequest


async def get_generation(body: RAGRequest = Body(...)) -> dict:
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}
    try:

        generation = await supervisor.ainvoke({"messages": [{"role": "user",
                       "content": body.query}]}, config
        )

        return {
            "answer": generation.get("generation", ""),
            "web_search": generation.get("web_search", False)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
