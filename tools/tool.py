from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from config import BaseConfig

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY
TAVILY_API_KEY = settings.TAVILY_API_KEY

tools = [TavilySearch(max_results=1), triple]
