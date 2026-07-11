from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_core.messages import HumanMessage

from nodes import run_agent_reasoning, tool_node
from consts import AGENT_REASON, ACT
from state import ReactState
import asyncio

workflow = StateGraph(ReactState)

workflow.add_node(AGENT_REASON, run_agent_reasoning)
workflow.add_node(ACT, tool_node)

workflow.add_edge(START, AGENT_REASON)
workflow.add_conditional_edges(
    AGENT_REASON,
    tools_condition,
    {"tools": ACT, END:END}
)
workflow.add_edge(ACT, AGENT_REASON)
react_agent = workflow.compile()

async def main():
    print("hello ReAct agent by Langraph")
    message = HumanMessage(content="What is the temperature in Tokyo?")
    res = await react_agent.ainvoke({"messages": [message]})
    print(res['messages'][-1].content)

if __name__ == "__main__":
    # react_agent.get_graph().draw_mermaid_png(output_file_path="react_agent.png")
    asyncio.run(main())


