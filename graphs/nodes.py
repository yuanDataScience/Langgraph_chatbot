from typing import Any

from graphs.chains import (answer_grader, hallucination_grader, retrieve_documents,
                           retrieval_grader, select_routes, web_search, generate_answer,
                           RouteQuery, GradeHallucinations
                           )
from graphs.consts import RETRIEVE, GENERATE, WEBSEARCH
from graphs.state import GraphState


async def generate(state: GraphState) -> dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = await generate_answer(documents, question)
    return {"generation": generation}


async def retrieve(state: GraphState) -> dict[str, Any]:
    print("___RETRIEVE---")
    question = state['question']

    documents = await retrieve_documents(question)
    return {"documents": documents}


async def search_web(state: GraphState) -> dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]

    web_docs = await web_search(question)

    documents = state.get("documents", []) + web_docs

    return {"documents": documents, "searched_web": True}


async def grade_documents(state: GraphState) -> dict[str, Any]:
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search_used = False

    for d in documents:
        score = await retrieval_grader(d, question)
        grade = score.binary_score
        if grade:
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search_used = True
            continue

    return {"documents": filtered_docs, "web_search": web_search_used}


async def grade_generation(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score: GradeHallucinations = await hallucination_grader(documents, generation)

    hallucination_grade = score.binary_score
    if hallucination_grade:
        print("---DECISION: GENERATION IS GROUNDED IN DOUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = await answer_grader(question, generation)

        answer_grade = score.binary_score
        if answer_grade:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"

    else:
        print("---DECISION: GENERATION IS NOT GOOUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def decide_to_generate(state):
    print("---CHECK GRADED DOCUMENTS")

    if state.get("web_search", False):
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE


async def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]

    route: RouteQuery = await select_routes(question)
    if route.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif route.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE