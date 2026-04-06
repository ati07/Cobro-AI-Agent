from fastapi import APIRouter, HTTPException, Depends
from app.schemas.request import AgentInvokeRequest
from app.schemas.response import AgentResponse
from app.services.agent_service import invoke_agent

router = APIRouter(prefix="/agent", tags=["agent"])

@router.post("/invoke", response_model=AgentResponse)
async def invoke_agent_endpoint(request: AgentInvokeRequest):
    try:
        # Convert simple messages if needed
        messages = request.messages
        # ✅ Normalize string → list format
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
            
        result = await invoke_agent(messages, request.llm_calls_limit)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))