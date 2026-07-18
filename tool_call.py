import asyncio

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from config import BaseConfig
from tools.weather import get_weather

settings = BaseConfig()
api_key = settings.OPENAI_API_KEY
TAVILY_API_KEY = settings.TAVILY_API_KEY


class WeatherAssistant:
    def __init__(self):

        # initialize llm
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)

        # create tool dictionary and assign a name to each function
        self.tools = {"get_weather": get_weather,
                      "tavily_search": TavilySearch(max_results=3, tavily_api_key=TAVILY_API_KEY)}

        # bind LangChain tools to llm
        self.llm_with_tools = self.llm.bind_tools(list(self.tools.values()))

        # initialize messages to store message list
        self.messages = []

        # System prompt
        self.system_prompt = f"""You are a helpful assistant for question-answering tasks. 
        When users ask about weather, use the get_weather tool to get weather. For other questions,
        use web_search. If you don't know the answer, just say that you don't know.
        Be conversational and helpful in your responses."""

        # append SystemMessage to message list
        self.messages.append(SystemMessage(content=self.system_prompt))

    async def chat(self, message: str):
        # Wrap User message in a HumanMessage and add it to message list
        self.messages.append(HumanMessage(content=message))

        # Get AI response (it may or may not contain tool calls)
        response = await self.llm_with_tools.ainvoke(self.messages)
        self.messages.append(response)

        # If there is any tool calls in the AI response
        if response.tool_calls:

            # process tool calls
            for tool_call in response.tool_calls:

                # retrieve name of the function and invoke it,
                # append resulting tool message to message list
                tool = self.tools[tool_call["name"]]
                tool_result = await tool.ainvoke(tool_call)
                self.messages.append(tool_result)

            # Get final response after tool execution
            final_response = await self.llm_with_tools.ainvoke(self.messages)
            self.messages.append(final_response)


async def main():
    print("hello tool calling!")
    assistant = WeatherAssistant()

    message = "What is the temperature in Tokyo?"
    await assistant.chat(message)

    for msg in assistant.messages:
        msg.pretty_print()


if __name__ == "__main__":
    asyncio.run(main())