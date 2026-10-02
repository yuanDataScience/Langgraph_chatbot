from contextlib import asynccontextmanager
from typing import Annotated
from pathlib import Path

from deepagents import create_deep_agent, FilesystemPermission
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend
from deepagents.backends.utils import create_file_data
from fastapi import (
    BackgroundTasks,
    FastAPI,
    HTTPException,
    status,
    File,
    Request,
    UploadFile
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from config import BaseConfig
from file_process import pdf_text_extractor
from schemas import ChatRequest
from upload import save_file
from graphs.cover_letter_agent import cover_letter_agent
from graphs.job_search_agent import job_search_agent

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY

PROJECT_ROOT = Path(__file__).resolve().parent
RESEARCH_DIR = PROJECT_ROOT / "research"

demo_context = {"user_id": "u_123", "workspace_id": "scidataapp"}

ROOT_INSTRUCTIONS = """
You are a career-workflow coordinator. Perform exactly one delegated task at a time.

1. Understand the user's target job title, location preferences, and skills.
2. Discover and confirm relevant current job postings.
3. Job research must complete before cover letter drafting begins. Do not run job research and cover letter drafting in parallel.
4. Treat the job-search response as intermediate data, not as the final answer.
5. The cover-letter agent must be invoked even if the selected-job result contains
fewer than five jobs, provided that the job-search agent returned successfully.
Do not silently stop after the research phase.

- Every cover letter must be based on confirmed job details.
- Do not invoke both subagents in parallel.
- Do not write files yourself.
- When invoking cover-letter-agent, include the complete candidate resume
content in the task description. Do not merely say that the resume was
provided.

Do not invoke cover-letter-agent until the task description contains:
1. The complete candidate resume.
2. The selected-job JSON.
3. The candidate's name as present in the resume.

Invoke cover-letter-agent exactly once for the entire selected-job list.
Do not invoke it once per job.
- The workflow is complete only after the cover-letter agent returns successfully.
"""


def memory_namespace(runtime):
    context = runtime.context
    return get_namespace(context)


def get_namespace(context):
    return (
        "memory",
        context["workspace_id"],
        context["user_id"],
    )

main_agent_permissions = [
    FilesystemPermission(operations=["read"], paths=["/research/**"], mode="allow"),
    FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
]



@asynccontextmanager
async def lifespan(app: FastAPI):
    checkpointer = MemorySaver()
    store = InMemoryStore()

    # initialize the store with namespace, file location and data
    store.put(
        get_namespace(demo_context),
        "/AGENTS.md",
        create_file_data("""\
    # Project Guidelines

    ## Code Style
    - All functions must have type annotations
    - Maximum line length is 80 characters
    - Use `pathlib.Path` for file operations, not `os.path`
    """),
    )

    app.state.agent = create_deep_agent(
        tools=[],  # No search tools required for writing cover letters
        system_prompt=ROOT_INSTRUCTIONS,
        subagents=[job_search_agent, cover_letter_agent],
        backend=CompositeBackend(
            default=StateBackend(),
            routes={"/memories/": StoreBackend(namespace=memory_namespace),
                    "/research/": FilesystemBackend(root_dir=RESEARCH_DIR, virtual_mode=True)}),
        model=ChatOpenAI(api_key=api_key, model="gpt-4o-mini"),
        memory=["/memories/AGENTS.md"],
        checkpointer=checkpointer,
        store=store,
    )

    yield


app = FastAPI(lifespan=lifespan)


@app.post("/chat")
async def chat(request: ChatRequest, http_request: Request):
    agent = http_request.app.state.agent

    config = {
        "configurable": {
            "thread_id": request.thread_id,
        }
    }

    context = {"user_id": request.user_id, "workspace_id": request.workspace}

    result = await agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": request.message,
                }
            ]
        },
        config=config,
        context=context
    )

    return {
        "thread_id": request.thread_id,
        "message": result["messages"][-1].content,
    }


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
    except Exception as e:
        raise HTTPException(
            detail=f"An error occurred while saving file - Error: {e}",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    return {"filename": file.filename, "message": "File uploaded successfully"}
