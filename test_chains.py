import pytest
from chains import (generate_answer, retrieval_grader, select_routes,
                    hallucination_grader, relevance_grader
                    )
from rag_process import vector_service


@pytest.mark.asyncio
async def test_generate_answer_yes() -> None:
    question = "what is generative agents"
    context = """Generative agents are AI systems that combine generative models (like GPT) with 
    autonomous decision-making to simulate realistic, goal-driven behavior over time. 
    They can perceive, plan, act, and reflect — often used to model human-like 
    characters or assistants in simulations, games, or productivity tools."""

    generated_answer = await generate_answer(context, question)
    assert len(generated_answer) > 1


@pytest.mark.asyncio
async def test_rag_retrieval_yes() -> None:
    prompt = "what is prompt engineer ?"
    searched_documents = await vector_service.search_documents(prompt)
    searched_document = searched_documents[0]
    # logger.info(f"searched_content: {searched_document.page_content}")
    retrieval_grade = await retrieval_grader(searched_document, prompt)
    assert retrieval_grade


@pytest.mark.asyncio
async def test_rag_retrieval_no() -> None:
    prompt = "what is prompt engineer ?"
    searched_documents = await vector_service.search_documents(prompt)
    searched_document = searched_documents[0]
    # logger.info(f"searched_content: {searched_document.page_content}")
    retrieval_grade = await retrieval_grader(searched_document, "how to make pizza")
    assert not retrieval_grade

@pytest.mark.asyncio
async def test_hallucination_grader_answer_yes() -> None:
    question = "generative agents"
    docs = await vector_service.search_documents(question)

    answer = await generate_answer(docs, question)
    res = await hallucination_grader(docs, answer)

    assert res.binary_score

@pytest.mark.asyncio
async def test_hallucination_grader_answer_no() -> None:
    question = "generative agents"
    docs = await vector_service.search_documents(question)

    answer = "In order to make pizza we need to first start with the dough"
    res = await hallucination_grader(docs, answer)
    assert not res.binary_score


@pytest.mark.asyncio
async def test_router_to_vectorstore() -> None:
    question = "generative agents"

    res = await select_routes(question)
    assert res.datasource == "vectorstore"


@pytest.mark.asyncio
async def test_router_to_websearch() -> None:
    question = "how to become a millionaire?"

    res = await select_routes(question)
    assert res.datasource == "websearch"

