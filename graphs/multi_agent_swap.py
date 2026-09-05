from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tools import wikipedia_tool, stock_data_tool, python_repl_tool
from utils import pretty_print_messages
from config import BaseConfig
from langgraph_swarm import create_handoff_tool, create_swarm
from langgraph.checkpoint.memory import InMemorySaver
import asyncio

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY

# Add three tools to the list: wikipedia_tool, stock_data_tool, and python_repl_tool


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

# Create a researcher agent with access to two tools + the handoff tool
research_agent = create_agent(
    llm,
    tools=[wikipedia_tool, stock_data_tool, transfer_to_analyst],
    system_prompt="You provide summaries from Wikipedia, and can query load raw, numerical stock performance data.",
    name="researcher"
)

# Create a analyst agent with access to one tool + the handoff tool
analyst_agent = create_agent(
    llm,
    [python_repl_tool, transfer_to_researcher],
    # system_prompt="""You generate plots of stock performance data provided by another assistant.
    # When generating stock performance visualizations, ALWAYS save the figure to a file using `plt.savefig('plot.png')`
    # and call `plt.close()`. DO NOT use `plt.show()`.""",
    system_prompt = """You generate plots of stock performance data provided by another assistant.""",
    name="analyst"
)

config = {"configurable": {"thread_id": "1", "user_id": "1"}}
checkpointer = InMemorySaver()

# Create the swarm multi-agent graph and compile it



async def agent_run(agent, query: str, config=None):
    async for chunk in agent.astream(
            {"messages": [{"role": "user",
                           "content": query}]}, config
    ):
        pretty_print_messages(chunk)

def print_agent(agent):
    png_bytes = agent.get_graph().draw_mermaid_png()

    with open("swarm_agent_graph.png", "wb") as f:
        f.write(png_bytes)


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}
    checkpointer = InMemorySaver()

    swarm_agent = create_swarm(
        agents=[research_agent, analyst_agent],
        default_active_agent="researcher"
    ).compile(checkpointer=checkpointer)

    # print_agent(swarm_agent)

    query = """Plot a chart of Meta's share price over the last month"""
    # asyncio.run(agent_run(query))
    asyncio.run(agent_run(swarm_agent, query, config))