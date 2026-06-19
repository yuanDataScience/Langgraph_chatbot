from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from ..config import BaseConfig
from dotenv import load_dotenv
from langsmith.client import Client
from langchain_core.output_parsers import StrOutputParser
from langchain_tavily import TavilySearch


settings = BaseConfig()
api_key = settings.OPENAI_API_KEY
load_dotenv()

web_search_tool = TavilySearch(max_results=3)

# restrict the output format
class GradeAnswer(BaseModel):
    """grade if the answer addresses the question """

    binary_score: bool = Field(
        description="answer addresses the question, yes or no"
    )

async def answer_grader(question:str, answer:str) -> GradeAnswer:

    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeAnswer)

    system="""You are a grader assessing whether an LLM generation addresses/resolves a question \n
    Give a binary score of 'yes' or 'no'. 'Yes' means that the LLM generation resolves the question
    """

    answer_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: \n\n {question} \n\n LLM generation: {answer}"),
        ]
    )

    answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
    grade = await answer_grader.ainvoke({"question": question, "answer": answer})
    return grade.binary_score


async def generate_answer(context_doc: list[Document], question: str) -> str:
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
    hub_client = Client()
    prompt = hub_client.pull_prompt("rlm/rag-prompt")
    
    generation_chain = prompt | llm | StrOutputParser()
    generated_text = await generation_chain.ainvoke({"context": context_doc, "question": question})
    return generated_text


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in-generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


async def hallucination_grader(context_docs: list[Document], generation: str) -> bool:

    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)

    system = """You are a grader assessing whether an LLM generation is grounded in /
    supported by a set of retrieved facts. \n
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in 
        / supported by the set of facts."""

    hallucination_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Set of facts: \n\n {context_docs} \n\n LLM generation: {generation}"),
        ]
    )

    hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
    grade = await hallucination_grader.ainvoke({"context_docs": context_docs, "generation": generation})
    return grade.binary_score

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
    

async def rephrase_question(question: str, chat_history: list[dict[str, str]]) -> str:
    chat_history_str = ""
    for d in chat_history:
        role = 'Human' if d["role"] == 'user' else 'AI'
        chat_history_str += role + ":" + d["content"] + "\n"

    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
    hub_client = Client()
    # prompt = hub_client.pull_prompt("langchain-ai/chat-langchain-rephrase")
    prompt = hub_client.pull_prompt("joeywhelan/rephrase")
    rephrase_chain = prompt | llm | StrOutputParser()
    rephrased_question = await rephrase_chain.ainvoke({"chat_history": chat_history_str, "input": question})
    return rephrased_question



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

    # grade = asyncio.run(grade_answer(question, answer))
    # print(grade.binary_score)

    # generated_text = asyncio.run(generate_answer(answer, question))
    # print(generated_text)

    

    


