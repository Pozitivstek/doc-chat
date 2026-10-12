"""SQLite-хранилище: документы и фрагменты, изолированные по сессии."""
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_pages

MAX_DOCS_PER_SESSION = 5

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
    id      TEXT PRIMARY KEY,
    sid     TEXT NOT NULL,
    name    TEXT NOT NULL,
    pages   INTEGER NOT NULL,
    chunks  INTEGER NOT NULL,
    created REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_docs_sid ON docs(sid, created);
CREATE TABLE IF NOT EXISTS chunks (
    doc_id TEXT NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    idx    INTEGER NOT NULL,
    page   INTEGER NOT NULL,
    text   TEXT NOT NULL,
    PRIMARY KEY (doc_id, idx)
);
"""


class LimitError(Exception):
    """У сессии уже максимум документов."""


@dataclass(frozen=True)
class DocMeta:
    id: str
    name: str
    pages: int
    chunks: int
    created: float


class Store:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def add_document(self, sid: str, name: str, pages: list[str]) -> DocMeta:
        doc_id = uuid.uuid4().hex[:12]
        chunks = chunk_pages(doc_id, pages)
        with self._lock:
            count = self._db.execute("SELECT COUNT(*) FROM docs WHERE sid = ?", (sid,)).fetchone()[0]
            if count >= MAX_DOCS_PER_SESSION:
                raise LimitError(f"Не больше {MAX_DOCS_PER_SESSION} документов в одной сессии.")
            created = time.time()
            self._db.execute(
                "INSERT INTO docs(id, sid, name, pages, chunks, created) VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, sid, name, len(pages), len(chunks), created))
            self._db.executemany(
                "INSERT INTO chunks(doc_id, idx, page, text) VALUES (?, ?, ?, ?)",
                [(c.doc_id, c.index, c.page, c.text) for c in chunks])
            self._db.commit()
        return DocMeta(doc_id, name, len(pages), len(chunks), created)

    def list_documents(self, sid: str) -> list[DocMeta]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, name, pages, chunks, created FROM docs WHERE sid = ? ORDER BY created, id",
                (sid,)).fetchall()
        return [DocMeta(r["id"], r["name"], r["pages"], r["chunks"], r["created"]) for r in rows]

    def document_names(self, sid: str) -> dict[str, str]:
        return {d.id: d.name for d in self.list_documents(sid)}

    def delete_document(self, sid: str, doc_id: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM docs WHERE id = ? AND sid = ?", (doc_id, sid))
            self._db.commit()
            return cur.rowcount == 1

    def chunks_for(self, sid: str, doc_ids: list[str] | None) -> list[Chunk]:
        """Фрагменты документов сессии; чужие id молча игнорируются."""
        with self._lock:
            rows = self._db.execute(
                "SELECT c.doc_id, c.idx, c.page, c.text FROM chunks c JOIN docs d ON d.id = c.doc_id "
                "WHERE d.sid = ? ORDER BY d.created, d.id, c.idx", (sid,)).fetchall()
        wanted = set(doc_ids) if doc_ids is not None else None
        return [Chunk(r["doc_id"], r["idx"], r["page"], r["text"]) for r in rows
                if wanted is None or r["doc_id"] in wanted]

    def purge_older_than(self, seconds: float, now_offset: float = 0.0) -> int:
        cutoff = time.time() + now_offset - seconds
        with self._lock:
            cur = self._db.execute("DELETE FROM docs WHERE created < ?", (cutoff,))
            self._db.commit()
            return cur.rowcount
