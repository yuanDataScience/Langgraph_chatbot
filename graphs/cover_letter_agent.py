import asyncio
from pathlib import Path

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, FilesystemBackend
from langchain_openai import ChatOpenAI

from config import BaseConfig
from util import resume_str, run_agent

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY

model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_DIR = PROJECT_ROOT / "research"

COVER_LETTER_PROMPT = """
You are responsible for drafting tailored cover letters.

The main agent will provide:

- The user's resume
- The selected-job JSON returned by job-search-agent
- The target title and location, when available

The selected-job JSON identifies the jobs selected for drafting. It is not the
complete source of job requirements.

Before drafting, perform exactly this operation:

read_file("/research/sources.md")

The exact path is /research/sources.md. Do not use glob, ls, directory
search, or filesystem discovery to locate it.

If read_file("/research/sources.md") fails:
- Do not search for another path.
- Do not repeatedly retry the filesystem search.
- State that the source file could not be read.
- Do not invent job details.

1. Compare the verified job information with the candidate resume. Do not claim candidate 
experience that is not explicitly supported by the resume.
2. Use the detailed source material from `/research/sources.md` for the job's
   responsibilities, qualifications, technologies, and company information.
3. Compare those job details with the user's resume.
4. Use only information supported by the candidate resume and the verified job
description. Do not invent or assume technologies, cloud platforms,
projects, leadership experience, company initiatives, or responsibilities.
5. If a technology is not present in the resume, do not claim that the candidate
has experience with it. You may describe it as a skill sought by the employer,
but do not claim proficiency.   
6. use candidate's actual name in the signature.

For each selected job, write:

- A concise subject line
- A tailored cover letter with a maximum of 150 words

Write all results to a single file:

write_file("/research/cover_letters.md", ...)

Organize the file using one heading per job, for example:

# Company — Job Title

**Subject:** Application for [Job Title]

[Cover letter of no more than 150 words]

Best regards,
[the candidate's actual name]

Do not search for additional jobs.
Do not draft letters for jobs that are not present in the selected-job JSON.
If a selected job cannot be matched to detailed content in
`/research/sources.md`, do not guess. Clearly indicate that the job source
could not be located.

After writing the file, output the final contents of
`/research/cover_letters.md`.
"""

cover_letter_permissions = [
    FilesystemPermission(operations=["read", "write"], paths=["/research/**"], mode="allow"),
    FilesystemPermission(operations=["read", "write"], paths=["/**"], mode="deny"),
]

cover_letter_agent = {
    "name": "cover-letter-agent",
    "description": (
        "find relevant jobs"
    ),
    "system_prompt": COVER_LETTER_PROMPT,  # its own brain — never inherited
    "model": model,  # override — the cheaper Haiku 4.5
    "permissions": cover_letter_permissions,  # override — scoped write access
}


async def cover_letter_agent_test():
    # Provide mock job data and resume context for testing
    initial_message = (
            "Here are the jobs found:\n"
            "1. Company: Chewy | Title: Senior Machine Learning Engineer | Location: Boston, MA or Remote \n\n"
            "Candidate Resume:\n" + resume_str
    )

    root_instructions = (
        "You are the main coordinator. Delegate the task of writing cover letters "
        "to the job-search-agent subagent and return its final output."
        "Do not read or write files yourself."
    )

    agent = create_deep_agent(
        tools=[],  # No search tools required for writing cover letters
        system_prompt=root_instructions,
        subagents=[cover_letter_agent],
        model=model,
        backend=CompositeBackend(
            default=StateBackend(),
            routes={"/research/": FilesystemBackend(root_dir=RESEARCH_DIR, virtual_mode=True),
                    }),
    )

    print("Running cover_letter_agent test...\n")
    await run_agent(agent, initial_message)


if __name__ == "__main__":
    asyncio.run(cover_letter_agent_test())
