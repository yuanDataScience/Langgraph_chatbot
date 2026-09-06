import asyncio

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_swarm import create_handoff_tool, create_swarm

from config import BaseConfig
from sub_agents import research_agent, analyst_agent
from utils import print_agent, agent_run

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

# Create a handoff tool for analyst -> researcher
transfer_to_researcher = create_handoff_tool(
    agent_name="researcher",
    description="Transfer user to the researcher assistant, who can retrieve Wikipedia summaries or load stock performance data.",
)

# Create a handoff tool for researcher -> analyst
transfer_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description="Transfer user to the analyst assistant, who can create visualizations of provided data.",
)

checkpointer = InMemorySaver()

swarm_agent = create_swarm(
    agents=[research_agent, analyst_agent],
    default_active_agent="researcher"
).compile(checkpointer=checkpointer)

if __name__ == "__main__":
    print_agent(swarm_agent, "swarm_agent.png")

    config = {"configurable": {"thread_id": "1", "user_id": "1"}}
    query = """Plot a chart of Meta's share price over the last month"""
    asyncio.run(agent_run(swarm_agent, query, config))
