from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers.agent import router as agent_router
from app.core.config import settings
from app.core.database import get_db

db = get_db()

app = FastAPI(
    title="Cobros y Ventas AI Agent API",
    description="FastAPI server exposing a LangGraph-powered real estate CRM assistant.",
    version="1.0.0"
)

# CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agent_router)

@app.get("/health")
async def health_check():
    return {"status": "healthy", "database": "connected" if db.command("ping") else "error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)