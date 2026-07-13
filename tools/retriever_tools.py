from langchain.tools import tool
import asyncio
from rag_process.service import vector_service


@tool
async def search_agentic_docs(query: str) -> str:
    """
    Good for questions about generative agents, prompt engineering, and adversarial attacks.
    Pass a natural language search query to retrieve context from the agentic ai database.
    """
    # Simply invoke your existing vector service retriever
    docs = await vector_service.retriever.ainvoke(query)

    # Flatten the document content into a single string for the LLM
    return "\n\n".join([d.page_content for d in docs])

async def main():
    question_1 ="what is prompt engineering?"
    question_2 = "what is generative agents?"

    rs = await search_agentic_docs.ainvoke(question_1)
    print(rs)


if __name__ == "__main__":
    asyncio.run(main())




