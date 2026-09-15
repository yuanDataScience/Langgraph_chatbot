from typing import List

from pydantic import (
    BaseModel,
    Field,
)


class JobPosting(BaseModel):
    company: str = Field(description="Name of the hiring company.")
    title: str = Field(description="Job title of the position.")
    location: str = Field(description="Location or remote status.")
    link: str = Field(description="Direct URL to the job posting.")
    good_match: str = Field(description="One sentence explaining why this is a good match.")


class JobList(BaseModel):
    jobs: List[JobPosting]
