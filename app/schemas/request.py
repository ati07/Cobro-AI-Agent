# app/schemas/request.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from typing import Union, List, Dict, Any
# class AgentInvokeRequest(BaseModel):
#     # messages: List[Dict[str, Any]] = Field(..., description="List of messages (role/content or full LangChain messages)")
#     messages: str
#     thread_id: Optional[str] = None  # For future state persistence
#     llm_calls_limit: Optional[int] = 20
class AgentInvokeRequest(BaseModel):
    messages: Union[str, List[Dict[str, Any]]]
    thread_id: Optional[str] = None
    llm_calls_limit: Optional[int] = 20

class AgentMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str