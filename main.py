from contextlib import asynccontextmanager
from typing import Annotated
from schemas import RAGRequest
from langchain_core.messages import HumanMessage

from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    status,
    File,
    Request,
    UploadFile,
    Body,
)
from langchain_mcp_adapters.client import MultiServerMCPClient

from rag_process import pdf_text_extractor, vector_service
from schemas import RAGResponse
from upload import save_file
from graphs.graph import build_graph

SERVER_CONFIG = {
    # config for mcp by sse transportation
    "fastapi_mcp_server": {
        "transport": "sse",
        "url": "http://localhost:8001/sse",
    }
}


@asynccontextmanager  # 1. CREATES a manager for FastAPI startup/shutdown
async def lifespan(fastapi_app: FastAPI):
    mcp_client = MultiServerMCPClient(SERVER_CONFIG)
    mcp_tools = await mcp_client.get_tools()

    # Compile graph dynamically with tools and store on app state
    fastapi_app.state.agent = await build_graph(mcp_tools)

    # PAUSES here while FastAPI runs and handles requests
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/upload")
async def file_upload_controller(
        file: Annotated[UploadFile, File(description="Uploaded PDF documents")],
        bg_text_processor: BackgroundTasks,
):
    if file.content_type != "application/pdf":
        raise HTTPException(
            detail=f"Only uploading PDF documents are supported",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    try:
        filepath = await save_file(file)
        bg_text_processor.add_task(pdf_text_extractor, filepath)
        bg_text_processor.add_task(
            vector_service.store_file_content_in_db,
            filepath.replace("pdf", "txt")
        )
    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred while saving file - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {"filename": file.filename, "message": "File uploaded successfully"}


@app.post("/generate_text", response_model=RAGResponse)
async def query_by_RAG_controller(request: Request, body: RAGRequest=Body(...)) -> RAGResponse:
    try:
        message = HumanMessage(content=body.question)

        agent = request.app.state.agent
        generation = await agent.ainvoke({"messages": [message]})
        response = {"answer": generation.get("generation", "")}

        return RAGResponse(**response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


