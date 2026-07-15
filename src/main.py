"""
Law Guide: FastAPI Backend
Serves the LangGraph legal advisor + premium web frontend.

Usage: python src/main.py
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn

from graph import run_query

# ── App Setup ──────────────────────────────────────────────────
app = FastAPI(
    title="Law Guide",
    description="AI Legal Advisor for Pakistani Law",
    version="1.0.0",
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "static")

if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ── Models ─────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:      str
    chat_history: Optional[List[str]] = []   # ← receives history from frontend


class ChatResponse(BaseModel):
    response:   str
    category:   str
    is_vague:   bool
    is_draft:   bool = False
    retries:    int  = 0
    confidence: str  = "Low"


# ── Routes ─────────────────────────────────────────────────────
@app.get("/")
async def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Law Guide API running. No frontend found at /static/index.html"}


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "Law Guide", "model": "llama-3.3-70b-versatile"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a legal query through the LangGraph pipeline."""
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        result = await run_query(
            query        = request.message.strip(),
            chat_history = request.chat_history or [],   # ← pass history to graph
        )
        return ChatResponse(
            response   = result["response"],
            category   = result["category"],
            is_vague   = result["is_vague"],
            is_draft   = result.get("is_draft", False),
            retries    = result.get("retries", 0),
            confidence = result.get("confidence", "Low"),
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# ── Entry Point ────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    is_development = os.environ.get("ENVIRONMENT", "development") == "development"
    
    print(f"⚖️  Starting Law Guide Server (env: {os.environ.get('ENVIRONMENT', 'development')})...")
    print(f"   Open: http://localhost:{port}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=is_development,
        reload_dirs=[os.path.dirname(os.path.abspath(__file__))] if is_development else None,
    )