from langgraph.graph import END, StateGraph

from graphs.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graphs.nodes import (generate, retrieve, search_web, grade_documents, route_question,
                          decide_to_generate, grade_generation,
                          )
from graphs.state import GraphState
import asyncio

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, search_web)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    }
)

# workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    }
)

workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graphs.png")

test_question = "what is prompt engineer ?"
test_question_1 = "what is generative agents?"
async def main():
    result = await app.ainvoke({"question": test_question})
    print(result)

if __name__ == "__main__":
    asyncio.run(main())

