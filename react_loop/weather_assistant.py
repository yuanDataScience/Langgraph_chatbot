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
        self.llm = ChatOpenAI(api_key=api_key, model="gpt-4o-mini", temperature=0)
        self.tools = {"get_weather": get_weather,
                      "web_search": TavilySearch(max_results=3, tavily_api_key=TAVILY_API_KEY)}
        self.llm_with_tools = self.llm.bind_tools(list(self.tools.values()))
        self.messages = []

        # System prompt
        self.system_prompt = f"""You are a helpful assistant for question-answering tasks. 
        When users ask about weather, use the get_weather tool to get weather. For other questions,
        use web_search. If you don't know the answer, just say that you don't know.
        Be conversational and helpful in your responses."""

        self.messages.append(SystemMessage(content=self.system_prompt))

    async def chat(self, message: str):
        # Add user message
        self.messages.append(HumanMessage(content=message))

        # loop to handle serial tool calls
        while True:

            # Get AI response with potential tool calls
            response = await self.llm_with_tools.ainvoke(self.messages)
            self.messages.append(response)

            # Check if there is any tools to call
            if not response.tool_calls:
                # no tools to call
                break
                # Execute each tool call

            # process tool calls
            # for tool_call in response.tool_calls:
            #     tool = self.tools[tool_call["name"]]
            #     tool_result = await tool.ainvoke(tool_call)
            #     self.messages.append(tool_result)

            # process tool calls asynchronously
            results = await asyncio.gather(*[self.tools[tc["name"]].ainvoke(tc) for tc in response.tool_calls])
            self.messages.extend(results)


async def main():
    print("hello Vanilla ReAct Loop")
    assistant = WeatherAssistant()

    message = "What is the temperature in Tokyo?"
    await assistant.chat(message)

    for msg in assistant.messages:
        msg.pretty_print()


if __name__ == "__main__":
    # react_agent.get_graph().draw_mermaid_png(output_file_path="react_agent.png")
    asyncio.run(main())
