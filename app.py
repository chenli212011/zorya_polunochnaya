import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import persistence
import translation
from config import settings
from agents.router import router as agents_router
from threads import router as threads_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    await persistence.init()
    yield
    await persistence.close()


app = FastAPI(title="Zorya Polunochnaya", lifespan=lifespan)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(agents_router)
app.include_router(threads_router)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("zorya")

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Zorya Polunochnaya</title>
  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
  <main class="container">
    <div class="panel">
      <span class="pill">running</span>
      <h1>Zorya Polunochnaya</h1>
      <p class="muted">Server running at <code>http://localhost:8070</code></p>

      <h2>Endpoints</h2>
      <ul class="endpoint-list">
        <li><code>GET /</code><span class="desc">this page</span></li>
        <li><code>GET <a href="/chat">/chat</a></code><span class="desc">chat UI (gpt-4o-mini)</span></li>
        <li><code>GET /api/health</code><span class="desc">health check</span></li>
        <li><code>GET /api/echo?msg=hello</code><span class="desc">echoes a query parameter</span></li>
        <li><code>GET /api/agents</code><span class="desc">list registered LangGraph agents</span></li>
        <li><code>POST /api/agents/{name}/run</code><span class="desc">invoke an agent (sync)</span></li>
        <li><code>POST /api/agents/{name}/stream</code><span class="desc">invoke an agent (SSE stream)</span></li>
        <li><code>GET /api/translate/languages</code><span class="desc">supported translation languages</span></li>
        <li><code>POST /api/translate</code><span class="desc">translate text to a target language</span></li>
        <li><code>GET /api/threads</code><span class="desc">list persisted conversation threads</span></li>
        <li><code>GET /api/threads/{id}</code><span class="desc">load a thread's full message history</span></li>
        <li><code>DELETE /api/threads/{id}</code><span class="desc">delete a thread</span></li>
        <li><code>GET /docs</code><span class="desc">interactive OpenAPI docs</span></li>
      </ul>
    </div>
  </main>
</body>
</html>
"""


@app.middleware("http")
async def log_requests(request: Request, call_next):
    client = request.client.host if request.client else "-"
    logger.info("%s %s from %s", request.method, request.url.path, client)
    return await call_next(request)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return INDEX_HTML


@app.get("/chat", response_class=HTMLResponse)
async def chat_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}


@app.get("/api/echo")
async def echo(msg: str = "") -> dict:
    return {"msg": msg}


# Heuristic filter for chat-capable OpenAI models. The /v1/models endpoint
# does not advertise capabilities, so we exclude obvious non-chat families
# and include the known chat-capable prefixes.
_OPENAI_CHAT_PREFIXES = ("gpt-", "chatgpt-", "o1", "o3", "o4")
_OPENAI_NON_CHAT = (
    "instruct", "realtime", "audio", "tts", "transcribe",
    "embedding", "whisper", "dall", "moderation", "search-",
    "image", "babbage", "davinci", "curie", "ada",
)


def _is_chat_model(model_id: str) -> bool:
    if any(token in model_id for token in _OPENAI_NON_CHAT):
        return False
    return model_id.startswith(_OPENAI_CHAT_PREFIXES)


class TranslateRequest(BaseModel):
    text: str = Field(..., description="Text to translate.")
    target: str = Field(..., min_length=2, description="Target language code (e.g. 'fr').")
    source: str | None = Field(default=None, description="Source language code; auto-detect if omitted.")


@app.get("/api/translate/languages")
def list_translate_languages() -> dict:
    return {"languages": translation.supported_languages()}


@app.post("/api/translate")
def api_translate(body: TranslateRequest) -> dict:
    try:
        out = translation.translate(body.text, body.target, body.source or "auto")
    except Exception as exc:
        logger.exception("translation failed")
        raise HTTPException(status_code=502, detail=f"translate error: {exc}")
    return {"translated": out, "target": body.target}


@app.get("/api/openai/models")
async def list_openai_models() -> dict:
    if not settings.has_openai:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set")

    # Lazy import so app boots without the openai package on the path
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    try:
        page = await client.models.list()
    except Exception as exc:
        logger.exception("openai models list failed")
        raise HTTPException(status_code=502, detail=f"openai error: {exc}")

    models = [
        {"id": m.id, "created": m.created}
        for m in page.data
        if _is_chat_model(m.id)
    ]
    models.sort(key=lambda m: m["id"])
    return {"models": models, "default": settings.default_openai_model}
