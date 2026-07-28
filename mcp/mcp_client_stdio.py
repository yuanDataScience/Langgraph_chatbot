import asyncio
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

BASE_DIR = Path(__file__).parent
SERVER_SCRIPT = BASE_DIR / "server_stdio.py"

SERVER_CONFIG = {
    # Server 1: Local server running directly via Python over stdio
    "local_search_server": {
        "transport": "stdio",
        "command": "python",
        "args": [str(SERVER_SCRIPT)],
    },
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
