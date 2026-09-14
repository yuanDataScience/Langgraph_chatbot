import asyncio
from typing import List, Dict, Any, Literal

from langchain_core.tools import tool
from langchain_tavily import TavilySearch

from config import BaseConfig

# Ensure your TAVILY_API_KEY environment variable is set
settings = BaseConfig()
api_key = settings.OPENAI_API_KEY
TAVILY_API_KEY = settings.TAVILY_API_KEY

# Initialize the TavilySearch tool instance
tavily_tool = TavilySearch(max_results=3, include_raw_content=False, tavily_api_key=TAVILY_API_KEY)


@tool
async def internet_search(
        query: str,
        topic: Literal["general", "news", "finance"] = "general"
) -> List[Dict[str, Any]]:
    """Search the web using Tavily."""
    # TavilySearch accepts a dictionary input or string query depending on usage;
    # invoking it with the query string and arguments maintains compatibility.
    result = await tavily_tool.ainvoke({
        "query": query,
        "topic": topic
    })

    # Depending on the version, TavilySearch might return a string or list/dict.
    # If it returns a string or list of results, parse or return accordingly.
    return result


async def main(query: str) -> str:
    response = await internet_search.ainvoke(query)
    print(response["results"][0]["content"])


if __name__ == "__main__":
    query = "what is the temperature in Boston today?"
    asyncio.run(main(query))
