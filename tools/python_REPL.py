import asyncio
from typing import Annotated
from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import asyncio
import textwrap
import matplotlib
matplotlib.use('Agg')  # Force headless non-GUI backend before importing pyplot!


repl = PythonREPL()


@tool
async def python_repl_tool(
        code: Annotated[str, "The python code to execute to generate your chart."],
) -> str:
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user. The chart should be displayed using `plt.show()`."""
    try:
        # Offload CPU-bound exec() to a background thread to keep event loop responsive
        clean_code = textwrap.dedent(code).strip()
        result = await asyncio.to_thread(repl.run, clean_code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"

    return (
        f"Successfully executed the Python REPL tool.\n\n"
        f"Python code executed:\n```python\n{code}\n```\n\n"
        f"Code output:\n```\n{result}```"
    )


async def main():
    code = f"""
    import numpy as np

    arr = np.arange(0, 9)
    print(arr)
    print(2 * arr)
    """

    result = await python_repl_tool.ainvoke({"code": code})
    print(result)


if __name__ == "__main__":
    asyncio.run(main())

