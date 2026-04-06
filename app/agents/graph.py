from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from typing import Annotated, Literal
from operator import add
from typing_extensions import TypedDict

from app.core.config import settings
from app.agents.tools import tools, tools_by_name
from app.agents.prompt import make_system_prompt

class MessagesState(TypedDict):
    messages: Annotated[list, add]
    llm_calls: int

# model = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     api_key=settings.google_api_key,
#     temperature=0,
# )
model = ChatGoogleGenerativeAI(
    model=settings.gemini_model,        # Now comes from .env
    api_key=settings.google_api_key,
    temperature=0,
)
model_with_tools = model.bind_tools(tools)

def llm_call(state: MessagesState):
    system_msg = make_system_prompt()
    response = model_with_tools.invoke([{"role": "system", "content": system_msg}] + state["messages"])
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

tool_node = ToolNode(tools)  # Use prebuilt ToolNode for cleaner code

def should_continue(state: MessagesState) -> Literal["tools", END]:
    return "tools" if state["messages"][-1].tool_calls else END

def create_agent_graph():
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("agent", llm_call)
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", should_continue, ["tools", END])
    graph_builder.add_edge("tools", "agent")
    return graph_builder.compile()

agent = create_agent_graph()