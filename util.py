async def run_agents(agent, user_message: str) -> None:
    try:
        async for step in agent.astream({"messages": [{"role": "user", "content": user_message}]}):
            for node_name, output in step.items():
                print(f"--- Node: {node_name} ---")
                if output and isinstance(output, dict) and "messages" in output:
                    for msg in output["messages"]:
                        msg.pretty_print()
    except Exception as e:
        print(f"Error: {e}")