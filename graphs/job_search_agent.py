import asyncio
from pathlib import Path

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, FilesystemBackend
from langchain_openai import ChatOpenAI

from config import BaseConfig
from schemas import JobList
from tools import internet_search
from util import run_agent

# Automatically generate the schema string to inject into the prompt
schema_json_example = JobList.model_json_schema()

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY

model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"

JOB_SEARCH_PROMPT = f"""Search and select 5 real postings that match the user's target title, location, and skills.
    
How to work:
1. Use internet_search to find matching job postings based on user's target title, location and skills
2. Save the COMPLETE, verbatim output of ALL your searches to a single file:
write_file("/research/sources.md", ...). Paste the results exactly
as the tool returned them — every result's title, URL, and full content
snippet. Do NOT summarize, trim, or reformat. This one file is your raw
archive: all the bulky material stays here so it never clutters the
editor's context.
3. Only after the file is successful saved, from what you found, select the best 5 postings and output them 
   strictly follow this JSON schema:\n{schema_json_example}\n
    Output ONLY this block format with no extra text before or after :\n
    <JOBS>\n[ ... ]\n</JOBS>"""

# Researchers may write under /research/** and are denied writes elsewhere.
research_permissions = [
    FilesystemPermission(operations=["read", "write"], paths=["/research/**"], mode="allow"),
    FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
]

job_search_agent = {
    "name": "job-search-agent",
    "description": (
        "find relevant jobs"
    ),
    "system_prompt": JOB_SEARCH_PROMPT,  # its own brain — never inherited
    "tools": [internet_search],  # override — replaces the inherited set
    "model": model,  # GPT-4o-mini
    "permissions": research_permissions,  # override — scoped write access
}


async def job_search_agent_demo() -> None:
    # Setup test inputs (simulating user query state)
    initial_message = (
        "Find me 5 Machine Learning Engineer jobs in Boston or Remote. "
        "My key skills are Python, PyTorch, and Kubernetes."
    )

    # Minimal supervisor / root instructions for the standalone test runner
    root_instructions = (
        "You are the main coordinator. Delegate the job search task entirely "
        "to the job-search-agent subagent and return its final output."
    )

    # Build the test agent system using your defined subagent dictionary
    agent = create_deep_agent(
        tools=[internet_search],
        system_prompt=root_instructions,
        subagents=[job_search_agent],
        model=ChatOpenAI(model="gpt-4o-mini", api_key=BaseConfig().OPENAI_API_KEY),
        backend=CompositeBackend(
            default=StateBackend(),
            routes={"/research/": FilesystemBackend(root_dir=RESEARCH_DIR, virtual_mode=True),
                    }),
    )

    # Execute the agent asynchronously
    await run_agent(agent, initial_message)


if __name__ == "__main__":
    asyncio.run(job_search_agent_demo())
