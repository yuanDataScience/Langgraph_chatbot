import asyncio

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_swarm import create_handoff_tool, create_swarm
from langchain.agents import create_agent
from tools import wikipedia_tool, stock_data_tool, python_repl_tool

from config import BaseConfig
from utils import print_agent, agent_run

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

# Create a handoff tool for analyst -> researcher
transfer_to_researcher = create_handoff_tool(
    agent_name="researcher",
    description="Transfer user to the researcher assistant, who can retrieve Wikipedia summaries or query stock performance data.",
)

# Create a handoff tool for researcher -> analyst
transfer_to_analyst = create_handoff_tool(
    agent_name="analyst",
    description="Transfer user to the analyst assistant, who can create visualizations for provided data.",
)

research_agent = create_agent(
    llm,
    tools=[wikipedia_tool, stock_data_tool, transfer_to_analyst],
    system_prompt="You provide summaries from Wikipedia, and can query raw, numerical stock performance data.",
    name="researcher"
)

# Create a analyst agent with access to one tool + the handoff tool
analyst_agent = create_agent(
    llm,
    [python_repl_tool, transfer_to_researcher],
    system_prompt="""You generate plots of stock performance data provided by another assistant.
    When generating stock performance visualizations,ALWAYS ALWAYS use `plt.savefig()` to save generated figure
    and call `plt.close()`. DO NOT use `plt.show()`.""",
    name="analyst"
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
