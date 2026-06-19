from fastapi import Body, HTTPException
from .graph import graph_app, rephrase_question
from .schemas import RAGRequest, RephraseRequest


async def get_generation(body: RAGRequest=Body(...)) -> dict:
    try:
        generation = await graph_app.ainvoke({"question": body.prompt})
        quality = generation.get("quality", {})
        return {
            "answer": generation.get("generation", ""),
            "hallucination_grade": quality.get("hallucination_grade", False),
            "relavant_grade": quality.get("relavant_grade", False),
            "num_orginial_documents": quality.get("num_document", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    
async def generate_rephrased_question(body: RephraseRequest=Body(...)) -> str:
    try:
        rephrased_question = await rephrase_question(body.question, body.chat_history)
        return rephrased_question
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
