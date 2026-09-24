---
name: career-workflow
description: Coordinates sequential job research and cover-letter generation by delegating work to the job-search-agent and cover-letter-agent.
---

# Career Workflow

Use this workflow when the user wants to find relevant jobs and generate
targeted cover letters.

This workflow is not complete when the job-search agent returns. The job-search
response is an intermediate result. After receiving it, you must invoke the
cover-letter agent before responding to the user.

## Input

Extract or preserve the following information from the user's request:

- Resume text
- Target job title
- Target location
- Prioritized skills

## Step 1: Job Search

Delegate the extracted user information to `job-search-agent`.

Pass it:

- The target job title
- The target location
- The prioritized skills
- The resume text

Wait for the job-search agent to return.

The response from `job-search-agent` is an intermediate workflow result. Do not
respond to the user after receiving it.

If the job-search agent fails, stop the workflow and report the failure. If it
returns successfully, continue immediately to Step 2.

## Step 2: Cover Letters

After the job-search agent returns successfully, invoke
`cover-letter-agent` immediately.

Pass it:

- The user's resume text
- The selected-job result returned by `job-search-agent`
- The instruction to read `/research/sources.md` for the detailed job
  descriptions

The selected-job result identifies which jobs to use. The detailed job
information should be read from `/research/sources.md`, rather than copied
entirely into the delegation message.

The cover-letter agent must be invoked even if the selected-job result contains
fewer than five jobs, provided that the job-search agent returned successfully.
Do not silently stop after the research phase.

Wait for the cover-letter agent to return.

## Ordering Rules

- Always invoke `job-search-agent` before `cover-letter-agent`.
- Never invoke the two subagents in parallel.
- Do not respond to the user between the two phases.
- Treat the job-search response as intermediate data, not as the final answer.
- Do not independently perform the subagents' specialized tasks.
- Do not call `cover-letter-agent` without a successful result from
  `job-search-agent`.

## Completion

The workflow is complete only after the cover-letter agent returns successfully.

Only then respond to the user and report the generated research and
cover-letter artifacts if they are available.