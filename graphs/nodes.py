from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END
from pydantic import BaseModel, Field

from config import BaseConfig
from graphs.state import GraphState
from utils import print_message, extract_documents, extract_question

settings = BaseConfig()

api_key = settings.OPENAI_API_KEY

GENERATE_SYSTEM_PROMPT = f"""You are an assistant for question-answering tasks with access to multiple tools.
    use these tools to retrieve context to answer the question. If you don't know the answer,
    just say that you don't know. Use three sentences maximum and keep the answer concise.

    Tool Use Instructions:

    - Use tools as much as possible.
    - Chain tools when needed (e.g., search for info, then use that to search
      for relevant hotels)
    - Be conversational and helpful
    - If you choose to answer a question using your internal knowledge instead of a tool, 
      you must be completely certain of the facts. If there is any ambiguity, 
      or if the question requires up-to-date or highly specific details, you MUST use a tool to verify.
      
    """

GRADER_SYSTEM_PROMPT = """""You are a grader assessing whether an LLM generation addresses/resolves a 
      question using the context. Generate critique and recommendations for LLM that generated the answer.
      always provide detailed recommendations, including requests for search more relevant context, generate
      answer based on context to reduce hallucination, and generate answer relevant to the question.

      The LLM generating this answer has the access to the following tools to obtain context:
      **get_weather**: search for weather information given a city name
      **search_agentic_docs**:  Good for questions about generative agents, prompt engineering, and adversarial attacks
      **TavilySearch**: General search tool from the internet.

      INSTRUCTIONS:
      (a) Evaluate if the context is relevant to the question
      Provide recommendation if context is not relevant to the question. You can suggest the LLM to use an 
      appropriate tool. If there is no context, you must be completely certain of the facts. If the question is
      covered by provided tools, or there is any ambiguity, 
      or if the question requires up-to-date or highly specific details, you MUST suggest to use a tool to verify answer.

      (b) Evaluate if the answer is based on the context
      if the answer is not based on the context, it is a hallucination. You should suggest to regenerate the answer
      based on the context.
      (c) Evaluate if the answer resolves the question 
      if the answer does not resolves the question, suggest to regenerate the answer

      The LLM generated this answer is strictly instructed to keep its answer CONCISE (maximum of 3 sentences). 
      Do NOT penalize the answer for omitting minor details if the core question is answered accurately.
      If all the evaluations are passed, no suggestions should be provided. 
      
      CRITERIA FOR ACCEPTABLE (True):
      - If the answer is accurate, addresses the user's question using the context, and does not hallucinate, 
        you MUST set acceptable to True, even if it missed some details from the context.
      - Only set acceptable to False if there is a fundamental flaw (hallucination, wrong information, 
        or completely missing the point).
      

      OUTPUT REQUIREMENTS:

        You must return a structured output with the following fields:

        - acceptable (boolean): True if the answer passed all evaluation,
          False if any of the evaluation fails and recommendations are provided
        - recommendations (string): recommendations you provide        

        INPUTS:

        question: {question}
        answer: {answer}
        context: {context} 
    """


async def generate_answer(state: GraphState, llm_with_tools: ChatOpenAI) -> dict:
    """
        Run agent reasoning node, synchronize context documents,
        and track web search tool usage safely.
        :param state: GraphState, manage graph state
        :return: dict containing new messages and state synchronization fields
        """

    print("----AGENT REASONING------")

    system_message = SystemMessage(content=GENERATE_SYSTEM_PROMPT)

    messages = [system_message] + state["messages"]

    if isinstance(messages[-1], ToolMessage):
        print_message(messages[-1])

    response = await llm_with_tools.ainvoke(messages)
    print_message(response)

    return {
        "messages": [response],
        "generation": response.content
    }


class GradeAnswer(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    acceptable: bool = Field(
        description="answer resolves question with no hallucination"
    )
    recommendations: str = Field(
        description="recommendations to improve answer"
    )


async def grade_answer(state: GraphState):
    """

    :param state:
    :return:
    """

    print("----GRADE ANSWER------")

    messages = state["messages"]
    answer = state["generation"]
    loop_count = state.get("loop_count", 0)
    question = extract_question(messages)
    context_list = extract_documents(messages)
    context_str = "\n\n".join(context_list) if context_list else "No context available (Model internal knowledge used)."

    eval_model = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
    structured_llm_grader = eval_model.with_structured_output(GradeAnswer)

    system_prompt = GRADER_SYSTEM_PROMPT.format(
        context=context_str,
        answer=answer,
        question=question
    )

    response = await structured_llm_grader.ainvoke([SystemMessage(content=system_prompt)])
    print(response)
    critique_message = f"""[SYSTEM EVALUATION & CRITIQUE]\n{response.recommendations}\n\n
                       Please regenerate your response addressing the critique above."""
    return {
        "acceptable": response.acceptable,
        "messages": [HumanMessage(content=critique_message)],
        "loop_count": loop_count + 1
    }


def route_after_grading(state: GraphState):
    if state.get("acceptable", False):
        return END

    # Circuit Breaker: Max 3 reflection attempts
    if state.get("loop_count", 0) >= 3:
        print("Max reflection loops reached. Forcing exit.")
        return END

    return "generate"
