import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


SERVER_CONFIG = {
    # config for mcp_service by sse transportation
    "fastapi_mcp_server": {
        "transport": "sse",
        "url": "http://localhost:8001/sse",
    }
}


def get_mcp_client() :
    """Works for stdio, sse, or a mix of both!"""
    client = MultiServerMCPClient(SERVER_CONFIG)
    return client



async def main():
    client = get_mcp_client()
    tools = await client.get_tools()
    print(tools)


if __name__ == "__main__":
    asyncio.run(main())