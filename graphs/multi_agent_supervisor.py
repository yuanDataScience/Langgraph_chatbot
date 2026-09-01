from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tools import wikipedia_tool, stock_data_tool, python_repl_tool
from utils import pretty_print_messages
from config import BaseConfig
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import InMemorySaver
import asyncio

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY

# Add three tools to the list: wikipedia_tool, stock_data_tool, and python_repl_tool


llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")


# Create a researcher agent with access to two tools + the handoff tool
research_agent = create_agent(
    llm,
    tools=[wikipedia_tool, stock_data_tool],
    system_prompt="You provide summaries from Wikipedia, and can query load raw, numerical stock performance data.",
    name="researcher"
)

# Create a analyst agent with access to one tool + the handoff tool
analyst_agent = create_agent(
    llm,
    [python_repl_tool],
    system_prompt="""You generate plots of stock performance data provided by another assistant. 
    When generating stock performance visualizations, ALWAYS save the figure to a file using `plt.savefig('plot.png')` 
    and call `plt.close()`. DO NOT use `plt.show()`.""",
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

    # print_agent(swarm_agent)

    query = """Plot a chart of Meta's share price over the last month"""
    # asyncio.run(agent_run(query))
    asyncio.run(agent_run(supervisor, query, config))