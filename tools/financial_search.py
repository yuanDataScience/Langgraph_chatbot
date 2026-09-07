import asyncio
from typing import Annotated

import pandas as pd
import yfinance as yf
from langchain_core.tools import tool


def _fetch_stock_sync(symbol: str, period: str = "30d") -> pd.DataFrame:
    """Synchronous helper function to fetch stock data via yfinance."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period)

    if df.empty:
        return pd.DataFrame()

    # Format to match your CSV schema: Date, Close/Last, Volume, Open, High, Low
    df = df.reset_index()
    df = df[['Date', 'Close', 'Volume', 'Open', 'High', 'Low']]
    df.columns = ['Date', 'Close/Last', 'Volume', 'Open', 'High', 'Low']
    return df


@tool
async def stock_data_tool(
        company_ticker: Annotated[str, "The ticker symbol of the company to retrieve stock performance data."],
        num_days: Annotated[int, "The number of business days of stock data."]
) -> str:
    """
    Use this to look up stock performance data for companies.
    Please check if you need to convert company names into ticker symbols in order to call this function,
    for example, you need to convert Apple Inc. to its ticker symbol: AAPL,
    Convert weeks, months, and years, into days if needed.
    """
    try:
        # Runs _fetch_stock_sync in a background ThreadPoolExecutor thread
        df = await asyncio.to_thread(_fetch_stock_sync, company_ticker, f"{num_days}d")
        return (f"Successfully executed the stock performance data retrieval tool to retrieve "
                f"the last *{num_days} days* of data for company **{company_ticker}**:\n\n{df.to_markdown()}")

    except Exception as e:
        print(f"Error fetching data for {company_ticker}: {e}")


async def main():
    # Fetch data concurrently for multiple stock tickers
    symbols = ["AAPL", "MSFT", "GOOGL"]

    # Concurrently launch tasks across the thread pool
    tasks = [
        stock_data_tool.ainvoke({"company_ticker": symbol, "num_days": 5})
        for symbol in symbols
    ]
    results = await asyncio.gather(*tasks)

    for symbol, tool_output in zip(symbols, results):
        print(f"\n--- {symbol} Result ---")
        print(tool_output)


if __name__ == "__main__":
    asyncio.run(main())
