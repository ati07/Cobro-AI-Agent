# app/services/agent_service.py
from app.agents.graph import agent
from app.schemas.response import AgentResponse
from langchain.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage


async def invoke_agent(messages: list, llm_calls_limit: int = 20) -> AgentResponse:
     # ✅ Convert dict messages to LangChain message objects
    lc_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(msg)  # Already a LangChain message object

    result = agent.invoke({   # ✅ Use sync invoke, not ainvoke
        "messages": lc_messages,
        "llm_calls": 0
    })
    # result = await agent.ainvoke({
    #     "messages": messages,
    #     "llm_calls": 0
    # })
    
    def extract_text(content):
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        return str(content)

    # Extract final human-readable response
    final_response = ""
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content:
            final_response = msg.content
            break

    return AgentResponse(
        messages=[msg.dict() if hasattr(msg, "dict") else {"content": str(msg)} for msg in result["messages"]],
        llm_calls=result.get("llm_calls", 0),
        final_response=extract_text(final_response)
    )