# app/schemas/response.py
from pydantic import BaseModel
from typing import List, Dict, Any

class AgentResponse(BaseModel):
    messages: List[Dict[str, Any]]
    llm_calls: int
    final_response: str  # Extracted last AIMessage content for convenience