from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langsmith.client import Client
from pydantic import BaseModel, Field

from config import BaseConfig

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY
TAVILY_API_KEY = settings.TAVILY_API_KEY

web_search_tool = TavilySearch(max_results=3, tavily_api_key=TAVILY_API_KEY)


async def generate_answer(context_doc: list[Document], question: str) -> str:
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
    hub_client = Client()
    prompt = hub_client.pull_prompt("rlm/rag-prompt")

    generation_chain = prompt | llm | StrOutputParser()
    generated_text = await generation_chain.ainvoke({"context": context_doc, "question": question})
    return generated_text


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: bool = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


async def retrieval_grader(document: Document, question: str):
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

    retrieval_grader = grade_prompt | structured_llm_grader
    grade = await retrieval_grader.ainvoke({"document": document, "question": question})

    return grade.binary_score


async def web_search(question: str) -> list[Document]:
    tavily_results = await web_search_tool.ainvoke({"query": question})

    documents = [
        Document(page_content=result["content"], metadata={"source": result.get("url", "")})
        for result in tavily_results["results"]
    ]

    return documents


async def filter_documents(documents: list[Document], question) -> list[Document]:
    filtered_docs = []
    for d in documents:
        score = await retrieval_grader(d, question)

        if score:
            filtered_docs.append(d)
    return filtered_docs


if __name__ == "__main__":
    # print(os.getenv("OPENAI_API_KEY"))

    question = "what is generative agents"
    answer = """Generative agents are AI systems that combine generative models (like GPT) with 
    autonomous decision-making to simulate realistic, goal-driven behavior over time. 
    They can perceive, plan, act, and reflect — often used to model human-like 
    characters or assistants in simulations, games, or productivity tools."""


