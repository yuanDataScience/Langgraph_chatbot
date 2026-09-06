from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import BaseConfig
from tools import wikipedia_tool, stock_data_tool, python_repl_tool

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY

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
    When generating stock performance visualizations,ALWAYS ALWAYS use `plt.savefig()` to save generated figure
    and call `plt.close()`. DO NOT use `plt.show()`.""",
    name="analyst"
)
