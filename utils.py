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
    extracted_docs = []

    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name in doc_sources:
            content = msg.content

            # 1. If content is a list (MCP text blocks or Tavily results)
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        # Extract 'text' from MCP content block, or 'content'/'snippet' from Tavily/Search dicts
                        text = item.get("text") or item.get("content") or item.get("snippet") or str(item)
                        extracted_docs.append(str(text))
                    else:
                        extracted_docs.append(str(item))

            # 2. If content is a dict
            elif isinstance(content, dict):
                text = content.get("text") or content.get("content") or content.get("snippet") or str(content)
                extracted_docs.append(str(text))

            # 3. If content is already a string
            elif content:
                extracted_docs.append(str(content))

    return extracted_docs


def extract_question(messages: list) -> str:
    for msg in messages:
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""
