import os
from loguru import logger
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import aiofiles

from config import BaseConfig

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY
pinecone_key = settings.PINECONE_API_KEY


class VectorService:
    def __init__(self, index_name: str = "knowledgebase"):
        self.index_name = index_name

        # Embedding model
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

        # Pinecone vectorstore wrapper
        self.vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=self.embeddings,
            pinecone_api_key=pinecone_key,
        )

        # Retriever
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # Text splitter (LangChain-native)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    async def store_file_content_in_db(self, filepath: str) -> None:
        """
        Read the uploaded text file, split the file, embed chunks and save chunks to vector database
        :param filepath: filepath to load to vectordb
        :return: None
        """
        logger.debug(f"Loading file: {filepath}")

        # Load entire file (safe for most RAG use cases)
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            raw_text = await f.read()

        # Minimal cleaning (preserve structure)
        raw_text = raw_text.replace("\r", "")

        # Split into LangChain Documents
        docs = self.splitter.create_documents(
            [raw_text],
            metadatas=[{"source": os.path.basename(filepath)}]
        )

        logger.debug(f"Storing {len(docs)} chunks into Pinecone")

        # Ingest into Pinecone
        PineconeVectorStore.from_documents(
            documents=docs,
            embedding=self.embeddings,
            index_name=self.index_name,
            pinecone_api_key=pinecone_key,
        )

    async def search_documents(self, query: str) -> list[Document]:
        logger.debug(f"Searching for: {query}")
        return self.retriever.invoke(query)


# Singleton instance
vector_service = VectorService()
