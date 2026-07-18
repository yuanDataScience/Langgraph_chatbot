from langchain_core.messages import AIMessage
from langchain_core.messages.base import get_msg_title_repr


def print_message(message):
    if isinstance(message, AIMessage) and isinstance(message.content, list):
        title = get_msg_title_repr(message.type.title() + " Message")
        msg_repr = f"{title}\n\n{message.text}"
    else:
        msg_repr = message.pretty_repr()
    print(msg_repr)
