import asyncio

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_supervisor import create_supervisor

from config import BaseConfig
from sub_agents import research_agent, analyst_agent
from utils import agent_run, print_agent

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

checkpointer = InMemorySaver()

# Create the supervisor multi-agent graph and compile it
supervisor = create_supervisor(
    model=llm,
    agents=[research_agent, analyst_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- a research agent. Assign research and data collection tasks to this agent\n"
        "- an analyst agent. Assign the creation of visualizations via code to this agent\n"
        "Assign work to one agent at a time, do not call agents in parallel.\n"
        "Do not do any work yourself."
    ),
    add_handoff_back_messages=True,
    # output_mode="full_history",
    output_mode="last_message"
).compile(checkpointer=checkpointer)

if __name__ == "__main__":
    print_agent(supervisor, "supervisor_agent.png")
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}

    query = """Plot a chart of Meta's share price over the last month"""
    asyncio.run(agent_run(supervisor, query, config))
    # asyncio.run(supervisor.ainvoke({"messages": [{"role": "user",
    #                        "content": query}]}, config
    # ))
