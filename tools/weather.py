from langchain.tools import tool
import httpx
from datetime import datetime, timedelta
import asyncio


@tool
async def get_weather(city: str, date: str = None) -> str:
    """Get weather information for a city.

    Args:
        city: Name of the city
        date: Date in YYYY-MM-DD format (optional, defaults to today)

    Returns:
        Weather information including temperature and conditions
    """
    try:
        # Use Open-Meteo free weather API (no key required)
        # First get coordinates for the city
        geocoding_url = (
            f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        )
        async with httpx.AsyncClient() as client:
            geo_response = await client.get(geocoding_url)

        geo_response.raise_for_status()

        geo_data = geo_response.json()
        if not geo_data.get("results"):
            return f"Error: City '{city}' not found"

        # Get weather data
        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]

        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"

        if date:
            weather_url += f"&start_date={date}&end_date={date}"

        async with httpx.AsyncClient() as client:
            weather_response = await client.get(weather_url)

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

async def main():
    next_week = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
    weather_forcast = await get_weather.ainvoke({"city": "Boston", "date": next_week})
    print(weather_forcast)
    weather_today = await get_weather.ainvoke({"city": "Boston"})
    print(weather_today)


if __name__ == "__main__":
    asyncio.run(main())


