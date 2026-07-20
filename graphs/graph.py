import asyncio

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from graphs.consts import GENERATE, ACT, GRADE, ROUTE

from graphs.nodes import generate_answer, tool_node, grade_answer, route_after_grading
from graphs.state import GraphState

workflow = StateGraph(GraphState)

workflow.add_node(GENERATE, generate_answer)
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
    {END: END, "generate": GENERATE }
)
app = workflow.compile()

question_1 = "what is prompt engineer ?"
question_2 = "what is generative agents?"
question_3 = "What is the temperature in Tokyo?"
question_4 = "how to make a pizza?"
question_5 = "what is the stock price of GOOGLE today?"

async def main():
    print("hello ReAct agent by Langraph")
    message = HumanMessage(content=question_1)
    res = await app.ainvoke({"messages": [message]})


if __name__ == "__main__":
    # app.get_graph().draw_mermaid_png(output_file_path="agent_workflow.png")
    asyncio.run(main())
