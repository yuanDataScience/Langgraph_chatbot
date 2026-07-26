import httpx
from langchain_tavily import TavilySearch
from mcp.server.fastmcp import FastMCP

from config import BaseConfig
from rag_process.service import vector_service

# mcp = FastMCP("Search Server")

mcp = FastMCP(
    name="My MCP Server",
    host="0.0.0.0",
    port=8001
)

settings = BaseConfig()
TAVILY_API_KEY = settings.TAVILY_API_KEY

tavily = TavilySearch(max_results=3, tavily_api_key=TAVILY_API_KEY)


@mcp.tool()
async def get_weather(city: str, date: str = None) -> str:
    """Get weather information for a city.

    Args:
        city: Name of the city
        date: Date in YYYY-MM-DD format (optional, defaults to today)

    Returns:
        Weather information including temperature and conditions
    """
    geocoding_url = "https://geocoding-api.open-meteo.com/v1/search"
    weather_url = f"https://api.open-meteo.com/v1/forecast"

    try:
        # Use Open-Meteo free weather API (no key required)
        # First get coordinates for the city

        async with httpx.AsyncClient() as client:
            geo_response = await client.get(
                geocoding_url,
                params={"name": city, "count": 1}
            )

            geo_response.raise_for_status()

            geo_data = geo_response.json()
            if not geo_data.get("results"):
                return f"Error: City '{city}' not found"

            # Get weather data
            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            weather_dict = {
                "latitude": lat, "longitude": lon,
                "current_weather": "true",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto"
            }

            if date:
                weather_dict["start_date"] = date
                weather_dict["end_date"] = date

            weather_response = await client.get(weather_url,
                                                params=weather_dict
                                                )

            weather_response.raise_for_status()

            data = weather_response.json()
            current = data.get("current_weather", {})
            daily = data.get("daily", {})

        result = f"Weather in {city}"
        if date and daily.get("time"):
            result += f" on {date}:\n"
            result += f"High: {daily['temperature_2m_max'][0]}°C\n"
            result += f"Low: {daily['temperature_2m_min'][0]}°C\n"
            result += f"Precipitation: {daily['precipitation_sum'][0]}mm"
        else:
            result += f" (current):\n"
            result += f"Temperature: {current.get('temperature', 'N/A')}°C\n"
            result += f"Wind Speed: {current.get('windspeed', 'N/A')} km/h"

        return result
    except Exception as e:
        return f"Error: Failed to get weather - {str(e)}"


@mcp.tool()
async def search_agentic_docs(query: str) -> str:
    """
    Good for questions about generative agents, prompt engineering, and adversarial attacks.
    Pass a natural language search query to retrieve context from the agentic ai database.
    """
    # Simply invoke your existing vector service retriever
    docs = await vector_service.retriever.ainvoke(query)

    # Flatten the document content into a single string for the LLM
    return "\n\n".join([d.page_content for d in docs])


if __name__ == "__main__":
    # mcp.run(transport="sse")
    mcp.run(transport="stdio")

