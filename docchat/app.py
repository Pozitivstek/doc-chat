"""FastAPI-приложение: загрузка документов, поиск, чат со стримингом ответа (SSE)."""
import asyncio
import json
import logging
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import anthropic
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints
from starlette.concurrency import run_in_threadpool

from .config import Settings, load_settings
from .llm import AnswerError, Answerer, ClaudeAnswerer, Delta, Done, ExtractiveAnswerer, Source
from .pdf import DocumentError, extract_pages, extract_text_file
from .ratelimit import RateLimiter
from .search import Bm25Index
from .store import LimitError, Store
from .text import tokenize

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
COOKIE = "dc_sid"
SID_RE = re.compile(r"^[A-Za-z0-9_-]{20,64}$")
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}
UPLOAD_OVERHEAD = 64 * 1024          # multipart-обвязка сверх размера файла
SHORT_QUESTION_TOKENS = 3            # короче — добавляем контекст прошлого вопроса
PURGE_INTERVAL = 15 * 60
NOT_FOUND_TEXT = ("В загруженных документах ничего не нашлось по этому вопросу. "
                  "Попробуйте переформулировать или уточнить, о чём спрашиваете.")

SECURITY_HEADERS = {
    b"x-content-type-options": b"nosniff",
    b"x-frame-options": b"DENY",
    b"referrer-policy": b"no-referrer",
    b"content-security-policy": (
        b"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        b"connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"),
}


class SecurityHeaders:
    """ASGI-прослойка: добавляет защитные заголовки, не буферизуя поток (важно для SSE)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() not in SECURITY_HEADERS]
                headers.extend(SECURITY_HEADERS.items())
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_headers)


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: Annotated[str, StringConstraints(max_length=4000)]


class ChatRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    doc_ids: list[str] | None = Field(default=None, max_length=20)
    history: list[HistoryItem] = Field(default_factory=list, max_length=20)


def error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse({"code": code, "message": message}, status_code=status)


def safe_name(raw: str | None) -> str:
    name = Path((raw or "").replace("\\", "/")).name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "", name).strip()[:120]
    return name or "document"


async def read_limited(file: UploadFile, limit: int) -> bytes | None:
    """Читает файл кусками; None — размер превысил лимит."""
    body = bytearray()
    while chunk := await file.read(64 * 1024):
        body.extend(chunk)
        if len(body) > limit:
            return None
    return bytes(body)


def sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_answerer(settings: Settings) -> Answerer:
    if not settings.has_llm:
        return ExtractiveAnswerer()
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    return ClaudeAnswerer(client, settings.model, effort=settings.effort, fallbacks=settings.fallbacks)


def create_app(settings: Settings | None = None, *, store: Store | None = None,
               answerer: Answerer | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = store or Store(settings.db_path)
    answerer = answerer or build_answerer(settings)
    limiter = RateLimiter(settings.chat_rate_limit, window=60.0)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async def purge_loop():
            while True:
                await asyncio.sleep(PURGE_INTERVAL)
                removed = await run_in_threadpool(store.purge_older_than, settings.ttl_hours * 3600)
                if removed:
                    log.info("удалено просроченных документов: %s", removed)

        task = asyncio.create_task(purge_loop())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title="DocChat", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(SecurityHeaders)
    app.state.store = store

    def current_sid(request: Request) -> str | None:
        sid = request.cookies.get(COOKIE)
        return sid if sid and SID_RE.match(sid) else None

    def with_session(request: Request, response: JSONResponse) -> tuple[str, JSONResponse]:
        """Возвращает sid и ставит cookie, если сессия новая."""
        sid = current_sid(request)
        if sid is None:
            sid = secrets.token_urlsafe(24)
            response.set_cookie(COOKIE, sid, max_age=int(settings.ttl_hours * 3600), httponly=True,
                                samesite="lax", secure=settings.cookie_secure, path="/")
        return sid, response

    def meta_json(meta) -> dict:
        return {"id": meta.id, "name": meta.name, "pages": meta.pages, "chunks": meta.chunks}

    @app.get("/api/config")
    async def get_config():
        return {
            "mode": getattr(answerer, "mode", "retrieval"),
            "model": settings.model if getattr(answerer, "mode", "") == "claude" else None,
            "max_upload_mb": round(settings.max_upload_bytes / (1024 * 1024), 1),
            "max_pages": settings.max_pages,
            "max_docs": 5,
            "ttl_hours": settings.ttl_hours,
        }

    @app.get("/api/docs")
    async def list_docs(request: Request):
        sid = current_sid(request)
        docs = await run_in_threadpool(store.list_documents, sid) if sid else []
        return {"documents": [meta_json(d) for d in docs]}

    @app.post("/api/docs")
    async def upload(request: Request, file: UploadFile = File(...)):
        sid, fresh = with_session(request, JSONResponse({}))

        def reply(status: int, code: str, message: str) -> JSONResponse:
            response = error(status, code, message)
            if COOKIE in fresh.headers.get("set-cookie", ""):
                response.headers["set-cookie"] = fresh.headers["set-cookie"]
            return response

        name = safe_name(file.filename)
        extension = Path(name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            return reply(415, "unsupported_type", "Поддерживаются PDF, TXT и MD.")
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_upload_bytes + UPLOAD_OVERHEAD:
            return reply(413, "too_large", f"Файл больше {settings.max_upload_bytes // (1024 * 1024)} МБ.")
        data = await read_limited(file, settings.max_upload_bytes)
        if data is None:
            return reply(413, "too_large", f"Файл больше {settings.max_upload_bytes // (1024 * 1024)} МБ.")

        try:
            if extension == ".pdf":
                pages = await run_in_threadpool(extract_pages, data, max_pages=settings.max_pages)
            else:
                pages = await run_in_threadpool(extract_text_file, data)
            meta = await run_in_threadpool(store.add_document, sid, name, pages)
        except DocumentError as exc:
            return reply(422, exc.code, exc.message)
        except LimitError as exc:
            return reply(409, "limit", str(exc))

        response = JSONResponse(meta_json(meta))
        if COOKIE in fresh.headers.get("set-cookie", ""):
            response.headers["set-cookie"] = fresh.headers["set-cookie"]
        return response

    @app.delete("/api/docs/{doc_id}")
    async def delete_doc(request: Request, doc_id: str):
        sid = current_sid(request)
        if not sid or not await run_in_threadpool(store.delete_document, sid, doc_id):
            return error(404, "not_found", "Документ не найден.")
        return {"ok": True}

    @app.post("/api/chat")
    async def chat(request: Request, body: ChatRequest):
        sid = current_sid(request)
        if sid is None:
            return error(400, "no_documents", "Сначала загрузите документ.")
        if not limiter.allow(sid):
            return error(429, "rate_limited", "Слишком много вопросов подряд. Подождите минуту.")

        names = await run_in_threadpool(store.document_names, sid)
        selected = [d for d in (body.doc_ids or []) if d in names] or None   # чужие id отбрасываются
        chunks = await run_in_threadpool(store.chunks_for, sid, selected)
        if not chunks:
            return error(400, "no_documents", "Сначала загрузите документ.")

        history = [item.model_dump() for item in body.history]
        query = body.question
        if len(tokenize(query)) < SHORT_QUESTION_TOKENS:
            previous = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
            query = f"{previous} {query}".strip()

        def retrieve():
            return Bm25Index(chunks).search(query, k=settings.top_k)

        hits = await run_in_threadpool(retrieve)
        sources = [Source(n, h.chunk.doc_id, names.get(h.chunk.doc_id, "document"), h.chunk.page,
                          h.chunk.text, round(h.score, 3)) for n, h in enumerate(hits, start=1)]

        async def events():
            yield sse("sources", {"sources": [
                {"n": s.n, "doc_id": s.doc_id, "doc_name": s.doc_name, "page": s.page,
                 "text": s.text, "score": s.score} for s in sources]})
            if not sources:
                yield sse("delta", {"text": NOT_FOUND_TEXT})
                yield sse("done", {"usage": None, "model": None})
                return
            try:
                async for event in answerer.stream_answer(body.question, history, sources):
                    if isinstance(event, Delta):
                        yield sse("delta", {"text": event.text})
                    elif isinstance(event, Done):
                        yield sse("done", {"usage": event.usage, "model": event.model})
            except AnswerError as exc:
                yield sse("error", {"message": str(exc)})
            except Exception:
                log.exception("сбой при генерации ответа")
                yield sse("error", {"message": "Не удалось получить ответ. Попробуйте ещё раз."})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/")
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app
