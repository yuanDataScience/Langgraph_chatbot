import asyncio
from typing import Annotated

import httpx
from langchain_core.tools import tool


@tool()
async def wikipedia_tool(
        query: Annotated[str, "The Wikipedia search to execute to find key summary information."],
) -> str:
    """Use this to search Wikipedia for factual information asynchronously."""
    headers = {"User-Agent": "MyLangChainAgent/1.0 (contact@example.com)"}

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        try:
            # Step 1: Search Wikipedia for matching pages
            search_url = "https://en.wikipedia.org/w/api.php"
            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "format": "json"
            }
            response = await client.get(search_url, params=search_params)
            response.raise_for_status()
            search_data = response.json()

            results = search_data.get("query", {}).get("search", [])
            if not results:
                return "No results found on Wikipedia."

            title = results[0]["title"]

            # Step 2: Fetch the page extract/summary
            summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
            summary_res = await client.get(summary_url)
            summary_res.raise_for_status()
            summary_data = summary_res.json()

            summary = summary_data.get("extract", "No summary available.")

        except Exception as e:
            return f"Failed to execute. Error: {repr(e)}"

    return f"Successfully executed:\nWikipedia summary: {summary}"


async def main(company_name: str) -> str:
    wiki_summary = await wikipedia_tool.ainvoke(f"{company_name}")
    print(wiki_summary)


if __name__ == "__main__":
    company = "Apple Inc."
    asyncio.run(main(company))
