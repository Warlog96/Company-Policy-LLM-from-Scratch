"""
REST API for the knowledge base model.

  uvicorn app.api:app --port 8000

  POST /ask   {"question": "..."}  ->  {"answer": "...", "source": "...", "latency_ms": 123.4}
  GET  /info  model details for the UI sidebar
  GET  /health
  GET  /       minimal chat page (app/static/index.html)

The model and tokenizer are loaded ONCE when the server starts.
Set COMPANY_AI_CKPT to use another checkpoint (e.g. checkpoints/quick/qa.pt).
"""
import csv
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from model.generate import KnowledgeBot  # noqa: E402

state = {}


def stage_val_losses(ckpt_path):
    """Last logged val loss of each training stage (read from logs/<stage>_log.csv)."""
    log_dir = os.path.join(ROOT, "logs", "quick" if "quick" in ckpt_path else "")
    out = {}
    for stage in ["pretrain", "knowledge", "qa"]:
        path = os.path.join(log_dir, stage + "_log.csv")
        if os.path.exists(path):
            with open(path) as f:
                vals = [r["val_loss"] for r in csv.DictReader(f) if r["val_loss"]]
            if vals:
                out[stage] = float(vals[-1])
    return out


@asynccontextmanager
async def lifespan(app):
    bot = KnowledgeBot()
    state["bot"] = bot
    state["info"] = dict(bot.info, stage_val_losses=stage_val_losses(bot.ckpt_path),
                         training_stages=["1. pretrain (English stories)",
                                          "2. knowledge (policy statements)",
                                          "3. qa (question -> answer + source)"])
    print("Model loaded:", bot.info)
    yield
    state.clear()


app = FastAPI(title="Arcvale Knowledge Base AI", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    source: str
    latency_ms: float


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    t0 = time.perf_counter()
    out = state["bot"].answer(question[:500])
    return AskResponse(answer=out["answer"], source=out["source"],
                       latency_ms=round((time.perf_counter() - t0) * 1000, 1))


@app.get("/info")
def info():
    return state["info"]


@app.get("/", include_in_schema=False)
def chat_page():
    return FileResponse(os.path.join(ROOT, "app", "static", "index.html"))


@app.get("/health")
def health():
    return {"status": "ok"}
