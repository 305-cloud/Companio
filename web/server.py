"""
web/server.py - a thin FastAPI wrapper around the existing Companion
class. No agent logic lives here -- every endpoint is a direct call
into agent.py, the same object the CLI (main.py) drives. One
Companion instance per domain, shared across users; each user is kept
separate purely by the user_id already threaded through every
Companion method.

Run:
    pip install fastapi uvicorn
    python web/server.py
    open http://localhost:8000
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

from agent import Companion
from domains.general import GENERAL_DOMAIN
from domains.study import STUDY_DOMAIN
from llm.stub import StubBackend

DOMAINS = {"general": GENERAL_DOMAIN, "study": STUDY_DOMAIN}
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "companion.db")


def _build_llm(instruction: str):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return StubBackend()
    try:
        from llm.adk_backend import ADKBackend
        return ADKBackend(instruction=instruction)
    except Exception:
        pass
    try:
        from llm.gemini import GeminiBackend
        return GeminiBackend()
    except Exception:
        return StubBackend()


# one Companion per domain, lazily built and reused across requests --
# mirrors how main.py builds exactly one Companion per REPL invocation.
_companions: Dict[str, Companion] = {}


def get_companion(domain_name: str) -> Companion:
    if domain_name not in DOMAINS:
        raise HTTPException(status_code=404, detail=f"unknown domain: {domain_name}")
    if domain_name not in _companions:
        domain = DOMAINS[domain_name]
        llm = _build_llm(domain.system_prompt or domain.purpose)
        _companions[domain_name] = Companion(domain=domain, llm=llm, db_path=DB_PATH)
    return _companions[domain_name]


app = FastAPI(title="Companio")


class TurnRequest(BaseModel):
    user_id: str
    domain: str = "general"
    text: str


class FeedbackRequest(BaseModel):
    user_id: str
    domain: str = "general"
    rating: str  # "up" | "down" | "correction"
    fact: Optional[Dict[str, Any]] = None


class ResolveRequest(BaseModel):
    user_id: str
    domain: str = "general"
    key: str
    accept_update: bool


@app.get("/api/domains")
def list_domains() -> List[Dict[str, str]]:
    return [{"id": k, "name": d.name, "purpose": d.purpose} for k, d in DOMAINS.items()]


@app.post("/api/session")
def start_session(user_id: str, domain: str = "general") -> Dict[str, str]:
    companion = get_companion(domain)
    companion.new_session(user_id)
    return {"status": "ok"}


@app.post("/api/turn")
def turn(req: TurnRequest) -> Dict[str, Any]:
    companion = get_companion(req.domain)
    result = companion.turn(req.user_id, req.text)
    return {
        "response": result.response,
        "asked_clarifying": result.asked_clarifying,
        "used_profile": result.used_profile,
        "confidence": round(result.confidence, 3),
        "pending_confirmations": result.pending_confirmations,
    }


@app.post("/api/feedback")
def feedback(req: FeedbackRequest) -> Dict[str, str]:
    companion = get_companion(req.domain)
    companion.give_feedback(req.user_id, req.rating, req.fact)
    if req.rating == "correction":
        companion.consolidate(req.user_id)
    return {"status": "ok"}


@app.post("/api/resolve_pending")
def resolve_pending(req: ResolveRequest) -> Dict[str, str]:
    companion = get_companion(req.domain)
    companion.resolve_pending(req.user_id, req.key, req.accept_update)
    return {"status": "ok"}


@app.get("/api/profile")
def profile(user_id: str, domain: str = "general") -> List[Dict[str, Any]]:
    return get_companion(domain).profile(user_id)


@app.delete("/api/profile/{key}")
def forget(key: str, user_id: str, domain: str = "general") -> Dict[str, bool]:
    ok = get_companion(domain).forget(user_id, key)
    return {"forgotten": ok}


@app.get("/api/feed")
def feed(user_id: str, domain: str = "general", n: int = 12) -> List[Dict[str, Any]]:
    return get_companion(domain).live_feed(user_id, n)


@app.get("/api/metrics")
def metrics(user_id: str, domain: str = "general") -> List[Dict[str, Any]]:
    return get_companion(domain).adaptation_metrics(user_id)


static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
