import pytest
from .graph.graph import graph_app
from .graph.chains import answer_grader
from loguru import logger
import sys


logger.remove()  # Remove default handler
logger.add(sys.stderr, level="INFO") 


@pytest.mark.asyncio
async def test_app_answer_yes() -> None:
    prompt = "what is prompt engineer ?"
    answer = await graph_app.ainvoke(input={"question": prompt})

    logger.info(f"anaswer: {answer}")
    answer_score = await answer_grader(prompt, answer)    

    assert answer_score


@pytest.mark.asyncio
async def test_app_answer_from_web_yes() -> None:
    prompt = "what is generative AI ?"
    answer = await graph_app.ainvoke(input={"question": prompt})

    logger.info(f"anaswer: {answer}")
    answer_score = await answer_grader(prompt, answer)    

    assert answer_score



