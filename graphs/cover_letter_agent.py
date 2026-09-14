from deepagents import FilesystemPermission, create_deep_agent
from langchain_openai import ChatOpenAI
from tools import internet_search
from config import BaseConfig
import asyncio

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY

model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

COVER_LETTER_PROMPT = f""""For each job in the found list, write a subject line and a concise cover letter (≤150 words)
 that ties the user's skills/resume to the role. Append to a single file: write_file("/research/cover_letters.md", ...)
  under a heading per job. Keep writing tight and specific. Output the content of the final cover_letters.md"""

cover_letter_permissions = [
    FilesystemPermission(operations=["read", "write"], paths=["/research/**"], mode="allow"),
    FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
]

cover_letter_agent = {
    "name": "job-search-agent",
    "description": (
        "find relevant jobs"
    ),
    "system_prompt": COVER_LETTER_PROMPT,         # its own brain — never inherited
    "model": model,                        # override — the cheaper Haiku 4.5
    "permissions": cover_letter_permissions,   # override — scoped write access
}


async def cover_letter_agent_test():
    resume_str = """Name: Yuan Huang
Title: Machine Learning Architect
Current Company: Sion Power
ML architect with 10+ years of experience designing and delivering end‑to‑end AI systems for scientific cloud/on‑prem environments. I specialize in building ML platforms that integrate data engineering, model development, CI/CD, and MLOps automation into reliable, production‑ready workflows.


Background spans machine learning, deep learning, time‑series modeling, and scientific computing, with hands‑on expertise in
PyTorch, Scikit‑learn, Pandas, and modern deployment stacks such as FastAPI, React, and MongoDB. I’ve architected Kubernetes‑based ML infrastructure (RKE2/K8s), implemented reproducible pipelines with DVC and MLflow, and built CI/CD systems using Jenkins and Git‑based workflows.


Hold multiple industry certifications, including AWS Machine Learning Specialty and Solutions Architect Associate, and have a strong track record of partnering with senior leadership and cross‑functional teams to translate business objectives into impactful AI solutions.

"""
    # Provide mock job data and resume context for testing
    initial_message = (
        "Here are the jobs found:\n"
        "1. Company: Samsara | Title: Senior Machine Learning Engineer | Location: Remote | Link: https://example.com/job1\n\n"
        "Candidate Resume:\n" + resume_str
    )

    root_instructions = (
        "You are the main coordinator. Delegate the task of writing cover letters "
        "to the job-search-agent subagent and return its final output."
    )

    agent = create_deep_agent(
        tools=[],  # No search tools required for writing cover letters
        system_prompt=root_instructions,
        subagents=[cover_letter_agent],
        model=model
    )

    print("Running cover_letter_agent test...\n")

    async for step in agent.astream({"messages": [{"role": "user", "content": initial_message}]}):
        for node_name, output in step.items():
            print(f"--- Node: {node_name} ---")
            if output and isinstance(output, dict) and "messages" in output:
                for msg in output["messages"]:
                    msg.pretty_print()


if __name__ == "__main__":
    asyncio.run(cover_letter_agent_test())