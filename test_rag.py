import pytest
from .rag_process import vector_service
from loguru import logger
from .graph import (answer_grader, 
                     generate_answer, 
                     retrieval_grader, 
                     hallucination_grader,
                     web_search)
import sys

logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO")  

# test if content retrieved from rag is relavant to the prompt
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
async def test_hallucination_yes() -> None:
    prompt = "what is prompt engineer ?"
    searched_documents = await vector_service.search_documents(prompt)
    
    generated_answer = await generate_answer(searched_documents, prompt)
    is_fact_based = await hallucination_grader(searched_documents, generated_answer)

    assert is_fact_based


@pytest.mark.asyncio
async def test_hallucination_no() -> None:
    prompt = "what is prompt engineer ?"

    searched_documents = await vector_service.search_documents(prompt)
    generated_answer = "In order to make pizza we need to first start with the dough"
    is_fact_based = await hallucination_grader(searched_documents, generated_answer)

    assert not is_fact_based    


# the answer is not hallucination and relavant to the question
@pytest.mark.asyncio
async def test_answer_yes() -> None:
    prompt = "what is prompt engineer ?"
    
    searched_documents = await vector_service.search_documents(prompt)
    generated_answer = await generate_answer(searched_documents, prompt)

    is_fact_based = await hallucination_grader(searched_documents, generated_answer)
    is_relavant = await answer_grader(prompt, generated_answer)

    assert is_fact_based and is_relavant  


# the answer is not hallucination but irrelavant to the question
@pytest.mark.asyncio
async def test_answer_yes() -> None:
    prompt = "what is prompt engineer ?"
    searched_documents = await vector_service.search_documents(prompt)
    generated_answer = await generate_answer(searched_documents, prompt)

    is_fact_based = await hallucination_grader(searched_documents, generated_answer)
    is_relavant = await answer_grader("how to make a pizza", generated_answer)

    assert is_fact_based and not is_relavant  


@pytest.mark.asyncio
async def test_web_search_yes() -> None:
    question = "agent memory"
    web_doc = await web_search(question)
    generated_answer = await generate_answer(web_doc, question)

    is_fact_based = await hallucination_grader(web_doc, generated_answer)
    is_relavant = await answer_grader(question, generated_answer)

    assert is_fact_based and is_relavant  


@pytest.mark.asyncio
async def test_web_search_no() -> None:
    question = "agent memory"
    web_doc = await web_search(question)
    generated_answer = await generate_answer(web_doc, question)

    is_fact_based = await hallucination_grader(web_doc, generated_answer)
    is_relavant = await answer_grader("how to make a pizza", generated_answer)

    assert is_fact_based and not is_relavant      



