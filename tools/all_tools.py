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

from langchain.tools import tool
import pandas as pd
import httpx
import zoneinfo
from datetime import datetime

# Load once at startup
parks_df = pd.read_csv("national_parks.csv")


def parse_coord(coord: str) -> float:
    """
    Converts coordinates like '44.35°N' or '68.21°W' into numeric floats.
    """
    coord = coord.strip().replace("°", "")
    value = float(coord[:-1])
    direction = coord[-1].upper()

    if direction in ["S", "W"]:
        return -value
    return value


def lookup_park(park_name: str):
    row = parks_df.loc[parks_df["name"].str.lower() == park_name.lower()]
    if row.empty:
        return None

    lat = parse_coord(row.iloc[0]["latitude"])
    lon = parse_coord(row.iloc[0]["longitude"])

    return {
        "name": park_name,
        "latitude": lat,
        "longitude": lon
    }


@tool
async def get_park_info(park_name: str):
    info = lookup_park(park_name)
    if not info:
        return {"error": "Park not found"}

    lat, lon = info["latitude"], info["longitude"]

    # Get timezone from Open-Meteo
    async with httpx.AsyncClient() as client:
        tz_resp = await client.get(
            "https://api.open-meteo.com/v1/timezone",
            params={"latitude": lat, "longitude": lon}
        )
        tz_data = tz_resp.json()
        timezone = tz_data.get("timezone")

    # Convert current time to park local time
    tz = zoneinfo.ZoneInfo(timezone)
    local_time = datetime.now(tz).isoformat()

    return {
        "park": info["name"],
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "local_time": local_time
    }



@tool
async def get_weather(park_name: str):
    """
    Returns 7-day weather forecast for a national park.
    """
    info = lookup_park(park_name)
    if not info:
        return {"error": "Park not found"}

    lat, lon = info["latitude"], info["longitude"]

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": info["timezone"]
            }
        )
        data = resp.json()

    return {
        "park": park_name,
        "timezone": info["timezone"],
        "forecast": data.get("daily", {})
    }

OPENTRIPMAP_API_KEY = "<your key>"

@tool
async def get_attractions(park_name: str):
    """
    Returns nearby attractions, trails, and POIs for a national park.
    """
    info = lookup_park(park_name)
    if not info:
        return {"error": "Park not found"}

    lat, lon = info["latitude"], info["longitude"]

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.opentripmap.com/0.1/en/places/radius",
            params={
                "radius": 20000,  # 20 km
                "lon": lon,
                "lat": lat,
                "apikey": OPENTRIPMAP_API_KEY,
                "limit": 30
            }
        )
        data = resp.json()

    return {
        "park": park_name,
        "results": data.get("features", [])
    }

ORS_API_KEY = "<your key>"

@tool
async def get_route(park_name: str, dest_lat: float, dest_lon: float):
    """
    Computes driving route from park center to a destination coordinate.
    """
    info = lookup_park(park_name)
    if not info:
        return {"error": "Park not found"}

    start = [info["longitude"], info["latitude"]]
    end = [dest_lon, dest_lat]

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.openrouteservice.org/v2/directions/driving-car",
            headers={"Authorization": ORS_API_KEY},
            json={"coordinates": [start, end]}
        )
        data = resp.json()

    return {
        "park": park_name,
        "distance_m": data["routes"][0]["summary"]["distance"],
        "duration_s": data["routes"][0]["summary"]["duration"],
        "geometry": data["routes"][0]["geometry"]
    }

GEOAPIFY_API_KEY = "<your key>"

@tool
async def get_hotels(park_name: str):
    """
    Returns hotels and lodging options near a national park.
    """
    info = lookup_park(park_name)
    if not info:
        return {"error": "Park not found"}

    lat, lon = info["latitude"], info["longitude"]

    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.geoapify.com/v2/places",
            params={
                "categories": "accommodation.hotel",
                "filter": f"circle:{lon},{lat},20000",
                "limit": 20,
                "apiKey": GEOAPIFY_API_KEY
            }
        )
        data = resp.json()

    return {
        "park": park_name,
        "hotels": data.get("features", [])
    }

@tool
async def get_local_time(park_name: str):
    """
    Converts current UTC time to the park's local timezone.
    """
    info = lookup_park(park_name)
    if not info:
        return {"error": "Park not found"}

    tz = info["timezone"]

    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://worldtimeapi.org/api/timezone/{tz}")
        data = resp.json()

    return {
        "park": park_name,
        "timezone": tz,
        "local_time": data.get("datetime")
    }

SENDGRID_API_KEY = "<your key>"

@tool
async def send_itinerary_email(email: str, subject: str, content: str):
    """
    Sends itinerary summary to the user via SendGrid.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "https://api.sendgrid.com/v3/mail/send",
            headers={
                "Authorization": f"Bearer {SENDGRID_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "personalizations": [{"to": [{"email": email}]}],
                "from": {"email": "trip-planner@example.com"},
                "subject": subject,
                "content": [{"type": "text/plain", "value": content}]
            }
        )

    return {"status": resp.status_code}