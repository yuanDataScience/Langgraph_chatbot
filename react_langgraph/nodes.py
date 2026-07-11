from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch

from tools.weather import get_weather
from utils import print_message
from config import BaseConfig
from state import ReactState


settings = BaseConfig()

api_key = settings.OPENAI_API_KEY
TAVILY_API_KEY = settings.TAVILY_API_KEY

tools = [TavilySearch(max_results=3, tavily_api_key=TAVILY_API_KEY), get_weather]

SYSTEM_PROMPT = """
You are a helpful assistant that can use tools to answer questions.
"""
system_message = SystemMessage(content=SYSTEM_PROMPT)

llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


async def run_agent_reasoning(state: ReactState) -> dict:
    """
    run agent reasoning node
    :param state: ReactState, manage graph state
    :return: dict containing new message
    """

    print("----AGENT REASONING------")
    messages = [system_message] + state["messages"]

    if isinstance(messages[-1], ToolMessage):
        print_message(messages[-1])

    response = await llm_with_tools.ainvoke(messages)
    print_message(response)

    return {"messages": [response]}


tool_node = ToolNode(tools)
