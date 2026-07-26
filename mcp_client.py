import asyncio
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

BASE_DIR = Path(__file__).parent
SERVER_SCRIPT = BASE_DIR / "server.py"

SERVER_CONFIG = {
    # Server 1: Local server running directly via Python over stdio
    "local_search_server": {
        "transport": "stdio",
        "command": "python",
        "args": [str(SERVER_SCRIPT)],
    },
}


async def get_mcp_tools() :
    """Works for stdio, sse, or a mix of both!"""
    client = MultiServerMCPClient(SERVER_CONFIG)
    tools = await client.get_tools()
    return tools



async def main():
    tools = await get_mcp_tools()
    print(tools)


if __name__ == "__main__":
    asyncio.run(main())
