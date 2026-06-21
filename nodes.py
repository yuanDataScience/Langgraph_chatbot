from typing import Dict, Any
from langchain_core.documents import Document
from chains import (generate_answer, filter_documents,
                    web_search,
                    )
from rag_process import vector_service


async def retrieve(question: str) -> Dict[str, Any]:
    print("___RETRIEVE---")

    retrieved_documents = await vector_service.search_documents(question)
    documents = await filter_documents(retrieved_documents, question)
    
    return {"documents": documents}


async def search_web(question) -> Dict[str, Any]:
    print("---WEB SEARCH---")

    web_docs = await web_search(question)
    documents = await filter_documents(web_docs, question)  
    
    return {"documents": documents, "searched_web": True}


async def generate(question: str) -> Dict[str, Any]:
    print("---GENERATE---")

    retrieved_docs = await retrieve(question)
    documents = retrieved_docs["documents"]
    if not documents:
        documents = (await search_web(question))["documents"]

    if not documents:
        generation = ""
    else:
        generation = await generate_answer(documents, question)

    return {"generation": generation, "num_original_documents": len(documents) if documents else 0}




    



