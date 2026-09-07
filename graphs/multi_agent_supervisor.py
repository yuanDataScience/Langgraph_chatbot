import asyncio

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph_supervisor import create_supervisor
from tools import wikipedia_tool, stock_data_tool, python_repl_tool
from langchain.agents import create_agent

from config import BaseConfig
from utils import agent_run, print_agent

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

checkpointer = InMemorySaver()

research_agent = create_agent(
    llm,
    tools=[wikipedia_tool, stock_data_tool],
    system_prompt="You provide summaries from Wikipedia, and can query raw, numerical stock performance data.",
    name="researcher"
)

# Create a analyst agent with access to one tool + the handoff tool
analyst_agent = create_agent(
    llm,
    tools=[python_repl_tool],
    system_prompt="""You generate plots of stock performance data provided by another assistant.
    When generating stock performance visualizations,ALWAYS use `plt.savefig()` to save generated figure
    and call `plt.close()`. DO NOT use `plt.show()`.""",
    name="analyst"
)


# Create the supervisor multi-agent graph and compile it
supervisor = create_supervisor(
    model=llm,
    agents=[research_agent, analyst_agent],
    prompt=(
        "You are a supervisor managing two agents:\n"
        "- research agent: assign research and data query and collection tasks to this agent\n"
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
    config = {"configurable": {"thread_id": "2", "user_id": "1"}}

    query = """Plot a chart of Meta's share price over the last month"""
    asyncio.run(agent_run(supervisor, query, config))

