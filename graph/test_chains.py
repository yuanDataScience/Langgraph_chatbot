import pytest
from .chains import answer_grader, generate_answer,rephrase_question
from loguru import logger

@pytest.mark.asyncio
async def test_qa_grader_yes() -> None:
    question = "generative agents"
    answer = """Generative agents are AI systems that combine generative models (like GPT) with autonomous decision-making to simulate realistic, goal-driven behavior over time. They can perceive, plan, act, and reflect — often used to model human-like 
    characters or assistants in simulations, games, or productivity tools."""

    grade = await answer_grader(question, answer)
    assert grade == True

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
async def test_rephrased_question_yes() -> None:
    question = "what did I just asked you?"
    chat_history = [{"role": "user", "content": "what is prompt engineer ?"}, 
                    {"role": "assistant", "content": "Prompt engineering, also known as in-context prompting, involves methods to communicate with large language models (LLMs) to guide their behavior towards desired outcomes without altering the model's weights. It is an empirical science that requires experimentation and heuristics, as its effectiveness can vary across different models. The primary goal is to achieve alignment and steerability of the model."}]
    rephrased_question = await rephrase_question(question, chat_history)
    assert rephrased_question.find("prompt engineering") > -1    