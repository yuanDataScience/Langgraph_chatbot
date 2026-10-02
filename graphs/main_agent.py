import asyncio
from pathlib import Path

from deepagents import (FilesystemPermission, create_deep_agent)
from deepagents.backends import CompositeBackend, StateBackend, FilesystemBackend
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from config import BaseConfig
from cover_letter_agent import cover_letter_agent
from job_search_agent import job_search_agent
from util import generate_task_prompt, resume_str

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"
config = {"configurable": {"thread_id": "thread-1"}}
demo_context = {"user_id": "u_123", "workspace_id": "sci-data-app"}

model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
store = InMemoryStore()

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY


def memory_namespace(runtime):
    user_id = runtime.context["user_id"]
    workspace_id = runtime.context["workspace_id"]
    return ("memories", workspace_id, user_id)


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

main_agent_permissions = [
    FilesystemPermission(operations=["read"], paths=["/research/**"], mode="allow"),
    FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
]

agent = create_deep_agent(
    tools=[],  # No search tools required for writing cover letters
    system_prompt=ROOT_INSTRUCTIONS,
    subagents=[cover_letter_agent, job_search_agent],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/research/": FilesystemBackend(root_dir=RESEARCH_DIR, virtual_mode=True),
                }),
    model=model,
    permissions=main_agent_permissions,
    memory=["/memories/AGENTS.md"],
    checkpointer=MemorySaver(),
    store=store,
)


async def run_agent_test():
    target_title = "Senior Machine Learning Engineer or MLOps Architect"
    target_location = "Boston, MA (or Remote)"
    skills = ["Python", "Kubernetes", "Airflow", "MLflow", "LangGraph", "Docker"]
    initial_message = generate_task_prompt(resume_str, skills, target_title, target_location)
    try:
        async for step in agent.astream({"messages": [{"role": "user", "content": initial_message}]},
                                        context=demo_context,
                                        config=config):
            for node_name, output in step.items():
                print(f"--- Node: {node_name} ---")
                if output and isinstance(output, dict) and "messages" in output:
                    for msg in output["messages"]:
                        msg.pretty_print()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(run_agent_test())
