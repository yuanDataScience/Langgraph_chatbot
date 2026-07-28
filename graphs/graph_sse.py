import asyncio
from functools import partial

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode


from config import BaseConfig
from graphs.consts import GENERATE, ACT, GRADE
from graphs.nodes import generate_answer, grade_answer, route_after_grading
from graphs.state import GraphState
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch


settings = BaseConfig()

api_key = settings.OPENAI_API_KEY
TAVILY_API_KEY = settings.TAVILY_API_KEY


async def build_graph(mcp_tools: list):
    """
    build StateGraph
    :return:
    """
    llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
    tools = [TavilySearch(max_results=3, tavily_api_key=TAVILY_API_KEY)] + mcp_tools
    print(tools)

    tool_node = ToolNode(tools)
    workflow = StateGraph(GraphState)
    llm_with_tools = llm.bind_tools(tools)

    workflow.add_node(GENERATE, partial(generate_answer, llm_with_tools=llm_with_tools))
    workflow.add_node(ACT, tool_node)
    workflow.add_node(GRADE, grade_answer)

    workflow.add_edge(START, GENERATE)
    workflow.add_conditional_edges(
        GENERATE,
        tools_condition,
        {"tools": ACT, END: GRADE}
    )
    workflow.add_edge(ACT, GENERATE)

    workflow.add_conditional_edges(
        GRADE,
        route_after_grading,
        {END: END, "generate": GENERATE}
    )
    app = workflow.compile()

    return app


question_1 = "what is prompt engineer ?"
question_2 = "what is generative agents?"
question_3 = "What is the temperature in Tokyo?"
question_4 = "how to make a pizza?"
question_5 = "what is the stock price of GOOGLE today?"


async def main():
    print("hello ReAct agent by Langraph")
    message = HumanMessage(content=question_1)
    app = await build_graph()
    res = await app.ainvoke({"messages": [message]})


if __name__ == "__main__":
    # app.get_graph().draw_mermaid_png(output_file_path="agent_workflow.png")
    asyncio.run(main())
