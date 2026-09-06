import asyncio

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from config import BaseConfig
from tools import wikipedia_tool, stock_data_tool, python_repl_tool
from utils import print_agent, agent_run

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

agent = create_agent(model=llm, system_prompt=prompt, tools=tools, name="financial_assistance")

if __name__ == "__main__":
    print_agent(agent, "agent.png")

    query = """Tell me Tesla's current CEO, their latest stock price, 
    and generate a plot of the closing price with the most up-to-date data you have available."""
    asyncio.run(agent_run(agent, query))
