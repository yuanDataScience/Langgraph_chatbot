import asyncio

from langchain_core.messages import HumanMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import tools_condition

from consts import AGENT_REASON, ACT
from nodes import (run_agent_reasoning, tool_node,
                   )
from state import ReactState

workflow = StateGraph(ReactState)

# workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(AGENT_REASON, run_agent_reasoning)
workflow.add_node(ACT, tool_node)

workflow.add_edge(START, AGENT_REASON)
workflow.add_edge(ACT, AGENT_REASON)

workflow.add_conditional_edges(
    AGENT_REASON,
    tools_condition,
    {"tools": ACT, END: END}
)

react_agent = workflow.compile()

react_agent.get_graph().draw_mermaid_png(output_file_path="graphs.png")

question = """
           tell me the first hotel you find in Boston if 
            its temperature is lower than 30 degree."""


async def main():
    print("hello ReAct agent by Langraph")
    message = HumanMessage(content=question)
    res = await react_agent.ainvoke({"messages": [message]})


if __name__ == "__main__":
    asyncio.run(main())
