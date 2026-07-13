from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.messages.base import get_msg_title_repr


def print_message(message):
    if isinstance(message, AIMessage) and isinstance(message.content, list):
        title = get_msg_title_repr(message.type.title() + " Message")
        msg_repr = f"{title}\n\n{message.text}"
    else:
        msg_repr = message.pretty_repr()
    print(msg_repr)


def format_conversation_history(messages: list) -> str:
    if not messages:
        return "No conversation history yet."
    formatted = "=== CONVERSATION HISTORY ===\n"
    for msg in messages:
        if isinstance(msg, HumanMessage):
            formatted += f"USER: {msg.text}\n"
        elif isinstance(msg, AIMessage):
            formatted += f"ASSISTANT: {msg.text}\n"
    return formatted


def extract_documents(messages: list) -> list[str]:
    """
    extract document list from tool messages and if web search tool has been called
    :param messages:
    :return:
    """

    doc_sources = ["TavilySearch", "tavily_search_results_json", "search_agentic_docs", "get_weather"]

    # 1. Look back through history to extract context from any ToolMessages
    # If no tools have run yet, this naturally results in an empty list []
    tool_messages = [msg for msg in messages if isinstance(msg, ToolMessage)]
    extracted_docs = [msg.content for msg in tool_messages if msg.name in doc_sources]

    return extracted_docs


def extract_question(messages: list) -> str:

    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""