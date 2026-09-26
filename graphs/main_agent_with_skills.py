import asyncio
from pathlib import Path

from deepagents import (FilesystemPermission, create_deep_agent)
from deepagents.backends import FilesystemBackend
from langchain_openai import ChatOpenAI

from config import BaseConfig
from cover_letter_agent import cover_letter_agent
from job_search_agent import job_search_agent
from util import run_agent, generate_task_prompt, resume_str

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"

model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

ROOT_INSTRUCTIONS = """
You are a workflow coordinator.

For career-related requests, execute the workflow strictly in two phases.

Phase 1:
- Invoke exactly one task call for job-search-agent.
- Do not invoke cover-letter-agent in the same model response.
- Wait for the job-search-agent result.

Phase 2:
- Only after job-search-agent returns successfully, invoke call for cover-letter-agent.
- Include the candidate resume, the selected job result, and the path
  /research/sources.md in the cover-letter task description.
- After the job-search agent returns, make exactly one task call to
cover-letter-agent. Pass all confirmed jobs in that single task call. The cover-letter agent
must draft a cover letter for every job and write them to one output file.
- Do not create one task call per job.
- Do not issue multiple cover-letter task calls in the same assistant message.

The subagents do not automatically receive the complete parent conversation.
Therefore, explicitly pass all required information in each task description.

Do not respond to the user between the two phases.
Do not invoke both subagents in parallel.
Do not write files yourself.
Only report workflow completion after cover-letter-agent returns successfully.
"""

main_agent_permissions = [
    FilesystemPermission(operations=["read"], paths=["/research/**", "/skills/**"], mode="allow"),
    FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
]

agent = create_deep_agent(
    tools=[],
    system_prompt=ROOT_INSTRUCTIONS,
    subagents=[cover_letter_agent, job_search_agent],
    backend=FilesystemBackend(
        root_dir=PROJECT_ROOT,
        virtual_mode=True,
    ),
    model=model,
    permissions=main_agent_permissions,
    skills=[
        str(Path(__file__).resolve().parents[1] / "skills" / "career-workflow")
    ],
)

if __name__ == "__main__":
    target_title = "Senior Machine Learning Engineer or MLOps Architect"
    target_location = "Boston, MA (or Remote)"
    skills = ["Python", "Kubernetes", "Airflow", "MLflow", "LangGraph", "Docker"]
    initial_message = generate_task_prompt(resume_str, skills, target_title, target_location)

    asyncio.run(run_agent(agent, initial_message))
