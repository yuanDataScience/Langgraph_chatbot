from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from tools import wikipedia_tool, stock_data_tool, python_repl_tool
from utils import pretty_print_messages
from config import BaseConfig
import asyncio

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY

# Add three tools to the list: wikipedia_tool, stock_data_tool, and python_repl_tool
tools = [wikipedia_tool, stock_data_tool, python_repl_tool]

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini")

# Create an agent using the create_react_agent function
prompt = """
You are an assistant for research and analysis of Fortune 500 companies. You have access to three tools:
- A Wikipedia tool for retrieving factual summary information about companies
- A stock performance data tool for retrieving stock price information from yfinance package
- A Python tool for executing Python code, which is to be used for creating stock performance visualizations. 
  When generating stock performance visualizations, ALWAYS save the figure to a file using `plt.savefig('plot.png')` 
  and call `plt.close()`. DO NOT use `plt.show()`.
"""

# Create an agent using the create_react_agent function


def print_agent(agent):
    png_bytes = agent.get_graph().draw_mermaid_png()

    with open("agent_graph.png", "wb") as f:
        f.write(png_bytes)

async def agent_run(agent, query: str):
    async for chunk in agent.astream(
            {"messages": [{"role": "user",
                           "content": query}]}
    ):
        pretty_print_messages(chunk)


if __name__ == "__main__":
    agent = create_agent(model=llm, system_prompt=prompt, tools=tools, name="financial_assistance")

    # print_agent(agent)
    query = """Tell me Tesla's current CEO, their latest stock price, 
    and generate a plot of the closing price with the most up-to-date data you have available."""
    asyncio.run(agent_run(agent, query))


