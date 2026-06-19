from fastapi import (
    BackgroundTasks,
    FastAPI, 
    HTTPException,
    status, 
    File, 
    UploadFile, Depends
    )
from typing import Annotated
from .rag_process import pdf_text_extractor, vector_service
from .upload import save_file
from .schemas import RAGResponse
from .dependencies import get_generation, generate_rephrased_question

app = FastAPI()

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
            filepath.replace("pdf", "txt"),
            512,
            "knowledgebase",
            768,
        )
    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred while saving file - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {"filename": file.filename, "message": "File uploaded successfully"}


@app.post("/generate_text", response_model=RAGResponse)
async def query_by_RAG_controller(generation: dict = Depends(get_generation)) -> RAGResponse:
    return generation


@app.post("/rephrase_question")
async def rephrase_question_controller(rephrased_question: str = Depends(generate_rephrased_question)) -> str:
    return rephrased_question
     
   