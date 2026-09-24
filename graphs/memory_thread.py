from pathlib import Path

from deepagents import (create_deep_agent)
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore

from config import BaseConfig
import asyncio

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = PROJECT_ROOT / "memories"
config_blue = {"configurable": {"thread_id": "thread-1"}}
config_red = {"configurable": {"thread_id": "thread-2"}}

model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")
demo_context = {"user_id": "u_123", "workspace_id": "acme"}
store = InMemoryStore()


def memory_namespace(runtime):
    context = runtime.context
    return get_namespace(context)


def get_namespace(context):
    return (
        "memory",
        context["workspace_id"],
        context["user_id"],
    )


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

ROOT_INSTRUCTIONS = """
you are a coding assistant. 
Persistent memories must be stored only under /memories/.
Never write persistent memories to the root directory or to paths such as
/user_preferences.txt.
Read /memories/AGENTS.md and follow its coding guidelines.

The checkpointer stores the conversation history. Do not write ordinary
conversation messages, user statements, questions, or answers to AGENTS.md.

You may modify /memories/AGENTS.md only when the user explicitly uses the
phrase "save to memories"
If the user does not explicitly say "save to memories", do not call
write_file or edit_file for memory storage.
Only tell the user that a memories was saved if the write operation succeeds.
"""


agent = create_deep_agent(
    tools=[],  # No search tools required for writing cover letters
    system_prompt=ROOT_INSTRUCTIONS,
    subagents=[],
    backend=CompositeBackend(
        default=StateBackend(),
        routes={"/memories/": StoreBackend(namespace=memory_namespace)}),
    model=model,
    memory=["/memories/AGENTS.md"],
    checkpointer=MemorySaver(),
    store=store,
)


def checkpointer_demo_config_blue() -> None:
    agent.invoke(
        {"messages": [{"role": "user", "content": "my favorite colour is blue."}]},
        config=config_blue,
        context=demo_context,
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my favorite colour?"}]},
        config=config_blue,
        context=demo_context,
    )

    print(result["messages"][-1].content)

    history = list(agent.get_state_history(config_blue))
    if history:
        latest_snapshot = history[0]  # Usually the newest snapshot
        for message in latest_snapshot.values.get("messages", []):
            message.pretty_print()

    print("\n---------------------start to test config_red-----------------------\n")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my favorite colour?"}]},
        config=config_red,
        context=demo_context,
    )

    print(result["messages"][-1].content)


async def memory_demo() -> None:
    initial_message = "save to memories that we change the maximum line length to 88 characters"
    try:
        async for step in agent.astream({"messages": [{"role": "user", "content": initial_message}]},
                                        context={"user_id": "u_123", "workspace_id": "acme"},
                                        config=config_blue):
            for node_name, output in step.items():
                print(f"--- Node: {node_name} ---")
                if output and isinstance(output, dict) and "messages" in output:
                    for msg in output["messages"]:
                        msg.pretty_print()
    except Exception as e:
        print(f"Error: {e}")


    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the maximum line length of our code style?"}]},
        config=config_blue,
        context=demo_context,
    )

    print(result["messages"][-1].content)


if __name__ == "__main__":
    # checkpointer_demo_config_blue()
    asyncio.run(memory_demo())


