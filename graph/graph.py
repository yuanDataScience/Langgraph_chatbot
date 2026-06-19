from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import END, StateGraph

from .constants import RETRIEVE, GENERATE, WEB_SEARCH, MIN_DOCUMENTS
from .nodes import generate, retrieve, search_web
from .state import GraphState
from .chains import answer_grader, hallucination_grader

async def grade_generation(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state.get("documents",[])
    generation = state["generation"]
    searched_web = state.get("searched_web", False)

    hallucination_grade = await hallucination_grader(documents, generation)
    relavant_grade = await answer_grader(question, generation)


    if hallucination_grade:
        print("---DECISION: GENERATION IS GROUNDED IN DOUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        
        if relavant_grade:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "end"
        if not searched_web:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
        else:
            print("---DECISION: CAN NOT FIND DOCUMENTS TO CORRECT HALLUCINATION")
            return "end"

    else:
        print("---DECISION: GENERATION IS NOT GOOUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    

def decide_to_generate(state):
    print("---CHECK IF NEED WEB SEARCH OR GENERATE RESULTS")

    documents = state.get("documents", [])
    searched_web = state.get("searched_web", False)

    if searched_web or len(documents) >= MIN_DOCUMENTS:
        return GENERATE
    else:
        return WEB_SEARCH   
   

workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
# workflow.add_node(GRADE_DOCUMENTS, evaluate_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, search_web)

workflow.set_entry_point(RETRIEVE)
# workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    RETRIEVE,
    decide_to_generate,
    {
        WEB_SEARCH: WEB_SEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation,
    {
        "not supported": GENERATE,
        "end": END,
        "not useful": WEB_SEARCH,
    }
)

workflow.add_edge(WEB_SEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

graph_app = workflow.compile()

if __name__ == "__main__":
    graph_app.get_graph().draw_mermaid_png(output_file_path="graph.png")      