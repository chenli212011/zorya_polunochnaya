"""Standalone MCP server for claude.ai integration.

Runs alongside (NOT inside) the main FastAPI app. Exposes a small set of
story-writing tools backed by the main app's HTTP API on localhost:8070, and
implements an OAuth 2.1 provider with Dynamic Client Registration so claude.ai's
"Add custom connector" flow can self-register.

Design constraints:
- Zero changes to the main app — purely additive
- Read-only access to the main app (HTTP calls, not direct DB)
- Separate venv (.venv-mcp), separate port (8071), separate SQLite DB
  (data/mcp_oauth.sqlite — distinct from data/checkpoints.db)
"""

from __future__ import annotations

import base64
import hashlib
import html
import json
import logging
import os
import secrets
import sqlite3
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env.mcp", override=False)
load_dotenv(ROOT / ".env", override=False)

# --- Configuration -----------------------------------------------------------

MCP_PORT = int(os.getenv("MCP_PORT", "8071"))
MCP_PUBLIC_URL = os.getenv("MCP_PUBLIC_URL", f"http://localhost:{MCP_PORT}").rstrip("/")
APP_BASE_URL = os.getenv("APP_BASE_URL", "http://localhost:8070").rstrip("/")
DB_PATH = Path(os.getenv("MCP_DB_PATH", str(ROOT / "data" / "mcp_oauth.sqlite")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

ACCESS_TOKEN_TTL = 3600                 # 1 hour
REFRESH_TOKEN_TTL = 30 * 24 * 3600      # 30 days
AUTH_CODE_TTL = 600                     # 10 minutes
SUPPORTED_SCOPES = ["mcp"]

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "zorya-polunochnaya-mcp", "version": "0.1.0"}

# --- Logging -----------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mcp")


# --- SQLite OAuth state ------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS clients (
    client_id TEXT PRIMARY KEY,
    client_secret TEXT,
    client_name TEXT,
    redirect_uris TEXT NOT NULL,
    token_endpoint_auth_method TEXT,
    created_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_codes (
    code TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    redirect_uri TEXT NOT NULL,
    scope TEXT,
    code_challenge TEXT,
    code_challenge_method TEXT,
    expires_at INTEGER NOT NULL,
    used INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS access_tokens (
    token TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    scope TEXT,
    expires_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS refresh_tokens (
    token TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    scope TEXT,
    expires_at INTEGER NOT NULL
);
"""


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def now() -> int:
    return int(time.time())


# --- OAuth helpers -----------------------------------------------------------

def gen_token(prefix: str = "") -> str:
    return f"{prefix}{secrets.token_urlsafe(32)}"


def verify_pkce(code_verifier: str, code_challenge: str, method: str) -> bool:
    if method == "S256":
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        expected = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
        return secrets.compare_digest(expected, code_challenge)
    if method == "plain":
        return secrets.compare_digest(code_verifier, code_challenge)
    return False


def validate_bearer(request: Request) -> dict:
    auth = request.headers.get("authorization", "")
    challenge = f'Bearer resource_metadata="{MCP_PUBLIC_URL}/.well-known/oauth-protected-resource"'
    if not auth.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token",
                            headers={"WWW-Authenticate": challenge})
    token = auth[7:].strip()
    with db() as conn:
        row = conn.execute(
            "SELECT client_id, scope, expires_at FROM access_tokens WHERE token = ?",
            (token,),
        ).fetchone()
    if row is None or row["expires_at"] < now():
        raise HTTPException(status_code=401, detail="invalid or expired token",
                            headers={"WWW-Authenticate": challenge})
    return {"client_id": row["client_id"], "scope": row["scope"]}


# --- Tools (HTTP wrappers over the main app's API) --------------------------

_http: httpx.AsyncClient | None = None


def http_client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(base_url=APP_BASE_URL, timeout=20.0)
    return _http


TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "list_threads",
        "description": (
            "List all persisted conversation threads in the local Zorya story-writing app. "
            "Each thread is a separate conversation that may hold story drafts, world-building "
            "notes, or research. Returns thread_id, a short title derived from the first user "
            "message, and message count. Call this first to discover what exists."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_thread",
        "description": (
            "Fetch the full message history of a single conversation thread by its thread_id. "
            "Returns a list of user/assistant messages with role and content. Use to bring "
            "an existing story or conversation into context before continuing it."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "thread_id": {
                    "type": "string",
                    "description": "The thread_id to load (obtain from list_threads).",
                },
            },
            "required": ["thread_id"],
        },
    },
    {
        "name": "count_threads",
        "description": "Return the total number of persisted threads.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


async def tool_list_threads(_args: dict) -> dict:
    r = await http_client().get("/api/threads")
    r.raise_for_status()
    return r.json()


async def tool_get_thread(args: dict) -> dict:
    thread_id = args.get("thread_id")
    if not thread_id:
        raise ValueError("thread_id is required")
    r = await http_client().get(f"/api/threads/{thread_id}")
    if r.status_code == 404:
        return {"error": f"thread not found: {thread_id}"}
    r.raise_for_status()
    return r.json()


async def tool_count_threads(_args: dict) -> dict:
    r = await http_client().get("/api/threads")
    r.raise_for_status()
    return {"count": len(r.json().get("threads", []))}


TOOL_FUNCS = {
    "list_threads": tool_list_threads,
    "get_thread": tool_get_thread,
    "count_threads": tool_count_threads,
}


# --- MCP protocol dispatcher (JSON-RPC over HTTP) ---------------------------

async def handle_mcp(request: dict, ctx: dict) -> dict | None:
    method = request.get("method")
    rid = request.get("id")
    params = request.get("params") or {}
    is_notification = "id" not in request

    def ok(result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    def err(code, message):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

    if method == "initialize":
        return ok({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": SERVER_INFO,
        })

    if method in ("notifications/initialized", "notifications/cancelled"):
        return None

    if method == "tools/list":
        return ok({"tools": TOOL_DEFINITIONS})

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        func = TOOL_FUNCS.get(name)
        if func is None:
            return err(-32601, f"unknown tool: {name}")
        try:
            result = await func(args)
        except Exception as exc:
            log.exception("tool %s failed", name)
            return ok({
                "isError": True,
                "content": [{"type": "text", "text": f"tool error: {exc}"}],
            })
        return ok({
            "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
        })

    if method == "ping":
        return ok({})

    if is_notification:
        return None
    return err(-32601, f"method not found: {method}")


# --- FastAPI app -------------------------------------------------------------

@asynccontextmanager
async def lifespan(_app: FastAPI):
    with db() as _:
        pass  # ensure schema exists
    log.info("MCP server starting — port=%s public=%s app=%s db=%s",
             MCP_PORT, MCP_PUBLIC_URL, APP_BASE_URL, DB_PATH)
    yield
    if _http is not None:
        await _http.aclose()


app = FastAPI(title="Zorya MCP Server", lifespan=lifespan)


# --- OAuth metadata ---------------------------------------------------------

@app.get("/.well-known/oauth-authorization-server")
def authorization_server_metadata() -> JSONResponse:
    return JSONResponse({
        "issuer": MCP_PUBLIC_URL,
        "authorization_endpoint": f"{MCP_PUBLIC_URL}/oauth/authorize",
        "token_endpoint": f"{MCP_PUBLIC_URL}/oauth/token",
        "registration_endpoint": f"{MCP_PUBLIC_URL}/oauth/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post", "client_secret_basic"],
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": SUPPORTED_SCOPES,
    })


@app.get("/.well-known/oauth-protected-resource")
def protected_resource_metadata() -> JSONResponse:
    return JSONResponse({
        "resource": MCP_PUBLIC_URL,
        "authorization_servers": [MCP_PUBLIC_URL],
        "scopes_supported": SUPPORTED_SCOPES,
        "bearer_methods_supported": ["header"],
    })


# --- Dynamic Client Registration --------------------------------------------

class RegisterRequest(BaseModel):
    client_name: str | None = None
    redirect_uris: list[str] = Field(default_factory=list)
    token_endpoint_auth_method: str | None = None
    grant_types: list[str] | None = None
    response_types: list[str] | None = None
    scope: str | None = None


@app.post("/oauth/register")
async def oauth_register(req: RegisterRequest) -> dict:
    if not req.redirect_uris:
        raise HTTPException(status_code=400, detail="redirect_uris is required")
    client_id = gen_token("client_")
    auth_method = req.token_endpoint_auth_method or "none"
    client_secret = None if auth_method == "none" else gen_token("secret_")

    with db() as conn:
        conn.execute(
            "INSERT INTO clients(client_id, client_secret, client_name, redirect_uris, "
            "token_endpoint_auth_method, created_at) VALUES (?,?,?,?,?,?)",
            (client_id, client_secret, req.client_name or "(unnamed)",
             json.dumps(req.redirect_uris), auth_method, now()),
        )

    response: dict[str, Any] = {
        "client_id": client_id,
        "client_name": req.client_name,
        "redirect_uris": req.redirect_uris,
        "token_endpoint_auth_method": auth_method,
        "grant_types": req.grant_types or ["authorization_code", "refresh_token"],
        "response_types": req.response_types or ["code"],
    }
    if client_secret is not None:
        response["client_secret"] = client_secret
    log.info("Registered client %s (%s)", client_id, req.client_name)
    return response


# --- Authorization endpoint -------------------------------------------------

def render_consent(*, client_name: str, redirect_uri: str, client_id: str,
                   response_type: str, state: str, scope: str,
                   code_challenge: str, code_challenge_method: str) -> str:
    e = html.escape
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Authorize {e(client_name)} — Zorya MCP</title>
  <style>
    body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 480px;
            margin: 4rem auto; padding: 2rem; background: #07060d; color: #fff;
            border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; }}
    h1 {{ font-size: 1.25rem; margin: 0 0 1rem; }}
    p  {{ color: rgba(255,255,255,0.7); line-height: 1.5; }}
    .actions {{ display: flex; gap: 0.75rem; margin-top: 1.5rem; }}
    button {{ flex: 1; padding: 0.75rem; border-radius: 8px;
              border: 1px solid rgba(255,255,255,0.15); font: inherit; cursor: pointer; }}
    .allow {{ background: #a78bfa; color: #07060d; border: none; font-weight: 600; }}
    .deny  {{ background: transparent; color: #fff; }}
    code   {{ font-family: ui-monospace, Menlo, monospace;
              background: rgba(255,255,255,0.07); padding: 1px 6px;
              border-radius: 4px; font-size: 0.85em; word-break: break-all; }}
  </style>
</head>
<body>
  <h1>Authorize <code>{e(client_name)}</code></h1>
  <p>This client is asking permission to access your Zorya story-writing tools
     (list threads, read thread contents).</p>
  <p><strong>Redirect URI:</strong><br><code>{e(redirect_uri)}</code></p>
  <form method="POST" action="/oauth/authorize">
    <input type="hidden" name="client_id" value="{e(client_id)}">
    <input type="hidden" name="redirect_uri" value="{e(redirect_uri)}">
    <input type="hidden" name="response_type" value="{e(response_type)}">
    <input type="hidden" name="state" value="{e(state)}">
    <input type="hidden" name="scope" value="{e(scope)}">
    <input type="hidden" name="code_challenge" value="{e(code_challenge)}">
    <input type="hidden" name="code_challenge_method" value="{e(code_challenge_method)}">
    <div class="actions">
      <button type="submit" name="decision" value="deny" class="deny">Deny</button>
      <button type="submit" name="decision" value="allow" class="allow">Allow</button>
    </div>
  </form>
</body>
</html>
"""


@app.get("/oauth/authorize")
def oauth_authorize_get(
    client_id: str = Query(...),
    redirect_uri: str = Query(...),
    response_type: str = Query("code"),
    state: str = Query(""),
    scope: str = Query(""),
    code_challenge: str = Query(""),
    code_challenge_method: str = Query("S256"),
) -> HTMLResponse:
    if response_type != "code":
        raise HTTPException(status_code=400, detail="only response_type=code is supported")

    with db() as conn:
        client = conn.execute(
            "SELECT client_id, client_name, redirect_uris FROM clients WHERE client_id = ?",
            (client_id,),
        ).fetchone()
    if client is None:
        raise HTTPException(status_code=400, detail="unknown client_id")

    allowed = json.loads(client["redirect_uris"])
    if redirect_uri not in allowed:
        raise HTTPException(status_code=400, detail="redirect_uri not in registered list")

    if code_challenge and code_challenge_method not in ("S256", "plain"):
        raise HTTPException(status_code=400, detail="unsupported code_challenge_method")

    return HTMLResponse(render_consent(
        client_name=client["client_name"] or client_id,
        redirect_uri=redirect_uri,
        client_id=client_id,
        response_type=response_type,
        state=state,
        scope=scope,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
    ))


@app.post("/oauth/authorize")
async def oauth_authorize_post(request: Request) -> RedirectResponse:
    form = await request.form()
    decision = form.get("decision")
    client_id = form.get("client_id", "")
    redirect_uri = form.get("redirect_uri", "")
    state = form.get("state", "")
    scope = form.get("scope", "")
    code_challenge = form.get("code_challenge", "")
    code_challenge_method = form.get("code_challenge_method", "S256")

    if decision != "allow":
        qs = urlencode({"error": "access_denied", "state": state})
        return RedirectResponse(f"{redirect_uri}?{qs}", status_code=303)

    code = gen_token("code_")
    with db() as conn:
        conn.execute(
            "INSERT INTO auth_codes(code, client_id, redirect_uri, scope, "
            "code_challenge, code_challenge_method, expires_at, used) "
            "VALUES (?,?,?,?,?,?,?,0)",
            (code, client_id, redirect_uri, scope, code_challenge,
             code_challenge_method, now() + AUTH_CODE_TTL),
        )
    log.info("Issued auth code for client %s", client_id)
    qs = urlencode({"code": code, "state": state})
    return RedirectResponse(f"{redirect_uri}?{qs}", status_code=303)


# --- Token endpoint ---------------------------------------------------------

def _extract_client_creds(request: Request, form_client_id: str, form_secret: str | None) -> tuple[str, str | None]:
    """Pull client_id / secret from form fields or HTTP Basic auth."""
    if form_client_id:
        return form_client_id, form_secret
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("basic "):
        try:
            decoded = base64.b64decode(auth[6:]).decode("utf-8")
            cid, _, csec = decoded.partition(":")
            return cid, csec or None
        except Exception:
            pass
    return "", None


@app.post("/oauth/token")
async def oauth_token(request: Request) -> dict:
    form = await request.form()
    grant_type = form.get("grant_type")

    if grant_type == "authorization_code":
        code = form.get("code", "")
        redirect_uri = form.get("redirect_uri", "")
        code_verifier = form.get("code_verifier", "")
        client_id, client_secret = _extract_client_creds(
            request, form.get("client_id", ""), form.get("client_secret"),
        )

        with db() as conn:
            row = conn.execute(
                "SELECT code, client_id, redirect_uri, scope, code_challenge, "
                "code_challenge_method, expires_at, used FROM auth_codes WHERE code = ?",
                (code,),
            ).fetchone()
            if row is None or row["used"] or row["expires_at"] < now():
                raise HTTPException(status_code=400,
                                    detail={"error": "invalid_grant",
                                            "error_description": "auth code invalid or expired"})
            if row["client_id"] != client_id:
                raise HTTPException(status_code=400,
                                    detail={"error": "invalid_grant",
                                            "error_description": "client mismatch"})
            if row["redirect_uri"] != redirect_uri:
                raise HTTPException(status_code=400,
                                    detail={"error": "invalid_grant",
                                            "error_description": "redirect_uri mismatch"})
            if row["code_challenge"]:
                if not code_verifier or not verify_pkce(
                    code_verifier, row["code_challenge"], row["code_challenge_method"] or "S256",
                ):
                    raise HTTPException(status_code=400,
                                        detail={"error": "invalid_grant",
                                                "error_description": "PKCE verification failed"})

            client = conn.execute(
                "SELECT client_id, client_secret, token_endpoint_auth_method "
                "FROM clients WHERE client_id = ?",
                (client_id,),
            ).fetchone()
            if client is None:
                raise HTTPException(status_code=400, detail={"error": "invalid_client"})
            if client["token_endpoint_auth_method"] != "none":
                if not client_secret or not secrets.compare_digest(
                    client["client_secret"] or "", client_secret,
                ):
                    raise HTTPException(status_code=401, detail={"error": "invalid_client"})

            conn.execute("UPDATE auth_codes SET used = 1 WHERE code = ?", (code,))

            access = gen_token("at_")
            refresh = gen_token("rt_")
            scope = row["scope"] or ""
            conn.execute(
                "INSERT INTO access_tokens(token, client_id, scope, expires_at) VALUES (?,?,?,?)",
                (access, client_id, scope, now() + ACCESS_TOKEN_TTL),
            )
            conn.execute(
                "INSERT INTO refresh_tokens(token, client_id, scope, expires_at) VALUES (?,?,?,?)",
                (refresh, client_id, scope, now() + REFRESH_TOKEN_TTL),
            )

        log.info("Issued access+refresh tokens to client %s", client_id)
        return {
            "access_token": access,
            "token_type": "Bearer",
            "expires_in": ACCESS_TOKEN_TTL,
            "refresh_token": refresh,
            "scope": scope,
        }

    if grant_type == "refresh_token":
        token = form.get("refresh_token", "")
        with db() as conn:
            row = conn.execute(
                "SELECT token, client_id, scope, expires_at FROM refresh_tokens WHERE token = ?",
                (token,),
            ).fetchone()
            if row is None or row["expires_at"] < now():
                raise HTTPException(status_code=400, detail={"error": "invalid_grant"})
            new_refresh = gen_token("rt_")
            access = gen_token("at_")
            conn.execute("DELETE FROM refresh_tokens WHERE token = ?", (token,))
            conn.execute(
                "INSERT INTO refresh_tokens(token, client_id, scope, expires_at) VALUES (?,?,?,?)",
                (new_refresh, row["client_id"], row["scope"], now() + REFRESH_TOKEN_TTL),
            )
            conn.execute(
                "INSERT INTO access_tokens(token, client_id, scope, expires_at) VALUES (?,?,?,?)",
                (access, row["client_id"], row["scope"], now() + ACCESS_TOKEN_TTL),
            )
        return {
            "access_token": access,
            "token_type": "Bearer",
            "expires_in": ACCESS_TOKEN_TTL,
            "refresh_token": new_refresh,
            "scope": row["scope"] or "",
        }

    raise HTTPException(status_code=400, detail={"error": "unsupported_grant_type"})


# --- MCP endpoint -----------------------------------------------------------

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    ctx = validate_bearer(request)
    body = await request.json()

    if isinstance(body, list):
        responses = []
        for item in body:
            r = await handle_mcp(item, ctx)
            if r is not None:
                responses.append(r)
        return JSONResponse(responses or [])

    response = await handle_mcp(body, ctx)
    if response is None:
        return JSONResponse({}, status_code=202)
    return JSONResponse(response)


@app.get("/mcp")
def mcp_get():
    raise HTTPException(status_code=405, detail="GET not supported on /mcp; POST only in v1")


# --- Health / debug ---------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "public_url": MCP_PUBLIC_URL, "app_base_url": APP_BASE_URL}


# --- Entrypoint -------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=MCP_PORT, log_level="info")
