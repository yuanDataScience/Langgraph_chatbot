from typing import Dict, Any
from .state import GraphState
from .chains import (generate_answer, 
                     web_search,
                     filter_documents,
                     hallucination_grader,
                     answer_grader
                     )
from .constants import MIN_DOCUMENTS
from ..rag_process import vector_service


async def retrieve(state: GraphState) -> Dict[str, Any]:
    print("___RETRIEVE---")
    question = state['question']
        
    retrieved_documents = await vector_service.search_documents(question)
    documents = await filter_documents(retrieved_documents, question)
    
    return {"documents": documents}


async def search_web(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")

    question = state["question"] 
    web_docs = await web_search(question) 
    documents = await filter_documents(web_docs, question)  
    
    return {"documents": documents, "searched_web": True}


async def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    
    question = state["question"]
    documents = state.get("documents", [])
    hallucination_grade = False
    relavant_grade = False

    if not documents:
        generation = ""
    else:
        generation = await generate_answer(documents, question)
        hallucination_grade = await hallucination_grader(documents, generation)
        relavant_grade = await answer_grader(question, generation)

    quality = {"hallucination_grade": hallucination_grade, 
               "relavant_grade": relavant_grade, 
               "num_document": len(documents)
               }

    return {"generation": generation, "quality": quality }




    



