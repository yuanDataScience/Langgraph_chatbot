from langchain.tools import tool
import httpx

@tool
async def get_weather(city: str):
    async with httpx.AsyncClient() as client:
        geo = await client.get(
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        )
        data = geo.json()
        return data
