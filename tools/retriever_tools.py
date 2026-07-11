from langchain_core.tools import create_retriever_tool

from rag_process.service import vector_service

# Wrap your retrievers into standard tools
RAG_agentic_description = "Good for questions about agents, prompt engineering, and adversarial attacks."

RAG_agentic_tool = create_retriever_tool(vector_service.retriever, "agentic_docs", RAG_agentic_description)
