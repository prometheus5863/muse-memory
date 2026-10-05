"""uvicorn muse.api:app  — POST /query, POST /chat, GET /calls, GET /stats."""
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

from muse import retrieve
from muse.graph import Memory
from muse.providers import Router

state: dict = {}
lock = threading.Lock()  # Memory.reload() swaps nodes/vecs/G; don't read mid-swap

SYSTEM = """You are a helpful assistant with access to the user's notes and past conversations.
Use the numbered context when relevant and cite it like [2]. If it does not contain the answer, say so
and answer from general knowledge."""


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["memory"] = Memory()
    state["router"] = Router(store=state["memory"].store)
    yield


app = FastAPI(title="muse", lifespan=lifespan)


class QueryIn(BaseModel):
    query: str
    k: int = 5
    budget: int | None = None
    mode: str = "graph"  # graph | flat


class ChatIn(BaseModel):
    message: str
    k: int = 5
    budget: int | None = None
    remember: bool = True


def _retrieve(q: str, k: int, budget: int | None, mode: str) -> list[dict]:
    with lock:
        return retrieve.flat(state["memory"], q, k) if mode == "flat" else retrieve.graph(state["memory"], q, k, budget)


@app.post("/query")
def query(body: QueryIn):
    return {"query": body.query, "mode": body.mode, "results": _retrieve(body.query, body.k, body.budget, body.mode)}


@app.post("/chat")
def chat(body: ChatIn, response: Response):
    ctx = _retrieve(body.message, body.k, body.budget, "graph")
    context = "\n\n".join(f"[{i}] ({c['source']})\n{c['text']}" for i, c in enumerate(ctx, 1))
    system = f"{SYSTEM}\n\n<context>\n{context or '(memory is empty)'}\n</context>"
    try:
        completion, provider, log = state["router"].complete(system, [{"role": "user", "content": body.message}])
    except RuntimeError as e:
        raise HTTPException(502, str(e))
    rid = log[0]["request_id"]
    remembered = []
    if body.remember:  # the exchange becomes memory: two nodes, linked like any other chunk
        with lock:
            remembered = state["memory"].add([f"User asked: {body.message}", f"Assistant answered: {completion.text}"],
                                             [f"chat:{rid}#user", f"chat:{rid}#assistant"])
    response.headers["X-Muse-Provider"] = provider
    return {
        "answer": completion.text, "provider": provider, "request_id": rid,
        "attempts": [{k: a.get(k) for k in ("provider", "attempt", "outcome", "latency_ms")} for a in log],
        "context": [{k: c[k] for k in ("id", "source", "reason", "via", "score")} for c in ctx],
        "remembered_ids": remembered,
    }


@app.get("/calls")
def calls(limit: int = 20):
    return state["memory"].store.recent_calls(limit)


@app.get("/stats")
def stats():
    with lock:
        return state["memory"].stats()
