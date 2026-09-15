import asyncio

from deepagents import FilesystemPermission, create_deep_agent
from langchain_openai import ChatOpenAI

from config import BaseConfig
from cover_letter_agent import cover_letter_agent
from job_search_agent import job_search_agent

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY

model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

ROOT_INSTRUCTIONS = """You are an autonomous career assistant helping users find relevant job opportunities and draft targeted cover letters.

Your Objectives:
1. Understand the user's target job title, location preferences, and skill set.
2. Coordinate the search for relevant, current job postings and save the raw findings into the research directory (e.g., /research/sources.md).
3. Generate concise, tailored cover letters (≤150 words) for the selected roles, saving them under headings in /research/cover_letters.md.

Execution & Dependency Rules:
- Execute your workflow sequentially. You MUST first discover and confirm job listings before attempting to draft cover letters.
- Do NOT invoke drafting or content-generation tools in parallel with research tools. Every cover letter requires valid job details as input.
- Always wait for the search results to be returned and saved before proceeding to the drafting phase.

Analyze the user's request and plan your steps according to these dependency rules.
You have access to specialized subagents to delegate these tasks. Analyze the user's request, plan your steps, and utilize 
your subagents appropriately to accomplish the workflow from start to finish.
"""

main_agent_permissions = [
    FilesystemPermission(operations=["read", "write"], paths=["/research/**"], mode="allow"),
    FilesystemPermission(operations=["write"], paths=["/**"], mode="deny"),
]

agent = create_deep_agent(
    tools=[],  # No search tools required for writing cover letters
    system_prompt=ROOT_INSTRUCTIONS,
    subagents=[cover_letter_agent, job_search_agent],
    model=model,
    permissions=main_agent_permissions
)

resume_str = """Name: Yuan Huang
Title: Machine Learning Architect
Current Company: Sion Power
ML architect with 10+ years of experience designing and delivering end‑to‑end AI systems for scientific cloud/on‑prem environments. I specialize in building ML platforms that integrate data engineering, model development, CI/CD, and MLOps automation into reliable, production‑ready workflows.


Background spans machine learning, deep learning, time‑series modeling, and scientific computing, with hands‑on expertise in
PyTorch, Scikit‑learn, Pandas, and modern deployment stacks such as FastAPI, React, and MongoDB. I’ve architected Kubernetes‑based ML infrastructure (RKE2/K8s), implemented reproducible pipelines with DVC and MLflow, and built CI/CD systems using Jenkins and Git‑based workflows.


Hold multiple industry certifications, including AWS Machine Learning Specialty and Solutions Architect Associate, and have a strong track record of partnering with senior leadership and cross‑functional teams to translate business objectives into impactful AI solutions.

"""


def make_task_prompt(resume_text: str, skills_hint: list[str], title: str, location: str) -> str:
    skills = "\n".join([skill.strip() for skill in skills_hint])
    skill_line = f" Prioritize these skills: {skills}." if skills else ""
    return (
        f"Target title: {title}\n"
        f"Target location(s): {location}\n"
        f"{skill_line}\n\n"
        f"RESUME RAW TEXT:\n{resume_text[:8000]}"
    )


async def run_agent_test():
    target_title = "Senior Machine Learning Engineer or MLOps Architect"
    target_location = "Boston, MA (or Remote)"
    skills = ["Python", "Kubernetes", "Airflow", "MLflow", "LangGraph", "Docker"]
    initial_message = make_task_prompt(resume_str, skills, target_title, target_location)
    try:
        async for step in agent.astream({"messages": [{"role": "user", "content": initial_message}]}):
            for node_name, output in step.items():
                print(f"--- Node: {node_name} ---")
                if output and isinstance(output, dict) and "messages" in output:
                    for msg in output["messages"]:
                        msg.pretty_print()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    asyncio.run(run_agent_test())
