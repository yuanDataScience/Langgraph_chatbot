from typing import Literal

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langsmith.client import Client
from pydantic import BaseModel, Field
from rag_process import vector_service

from config import BaseConfig

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY
TAVILY_API_KEY = settings.TAVILY_API_KEY

web_search_tool = TavilySearch(max_results=3, tavily_api_key=TAVILY_API_KEY)


async def retrieve_documents(question: str) -> list[str]:
    retrieved_documents = await vector_service.search_documents(question)
    documents = [d.page_content for d in retrieved_documents]

    return documents

async def generate_answer(context_doc: list[str], question: str) -> str:
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
    hub_client = Client()
    prompt = hub_client.pull_prompt("rlm/rag-prompt")
    context_str = "\n".join([c for c in context_doc])

    generation_chain = prompt | llm | StrOutputParser()
    generated_text = await generation_chain.ainvoke({"context": context_str, "question": question})
    return generated_text


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: bool = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


async def retrieval_grader(document: str, question: str):
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.\n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    if the document is relevant, give it a score 'yes', otherwise give it a score 'no'
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} User question: {question}"),

        ]
    )

    grader = grade_prompt | structured_llm_grader
    grade = await grader.ainvoke({"document": document, "question": question})

    return grade


async def web_search(question: str) -> list[str]:
    tavily_results = await web_search_tool.ainvoke({"query": question})

    documents = [
        result["content"] for result in tavily_results["results"]
    ]

    return documents



# define grader to check if the answer addressed the question
class GradeAnswer(BaseModel):
    """grade if the answer addresses the question """

    binary_score: bool = Field(
        description="answer addresses the question, yes or no"
    )


async def answer_grader(question: str, answer: str):
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    system = """You are a grader assessing whether an LLM generation addresses/resolves a question \n
      Give a binary score of 'yes' or 'no'. 'Yes' means that the LLM generation resolves the question
    """

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ]
    )

    answer_grader = answer_prompt | structured_llm_grader
    grade = await answer_grader.ainvoke({"question": question, "generation": answer})

    return grade


# define grader to check if the answer addressed the question
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in-generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


async def hallucination_grader(documents: list[str], generation: str):
    """

    :param documents:
    :param generation:
    :return:
    """
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system = """You are a grader assessing whether an LLM generation is grounded in /
    supported by a set of retrieved facts. \n
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in 
        / supported by the set of facts."""

    doc_str = "\n".join([d for d in documents])

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

    grade = await hallucination_grader.ainvoke({"documents": doc_str, "generation": generation})

    return grade


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource"""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )


async def select_routes(question: str):
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    system = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. For all else, use web search
    """

    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )

    question_router = route_prompt | structured_llm_router

    route = await question_router.ainvoke({"question": question})
    return route


if __name__ == "__main__":
    # print(os.getenv("OPENAI_API_KEY"))

    question = "what is generative agents"
    answer = """Generative agents are AI systems that combine generative models (like GPT) with 
    autonomous decision-making to simulate realistic, goal-driven behavior over time. 
    They can perceive, plan, act, and reflect — often used to model human-like 
    characters or assistants in simulations, games, or productivity tools."""