from __future__ import annotations

from array import array
from dataclasses import asdict
from pathlib import Path
import json
import sqlite3
from datetime import UTC, datetime

from .chunking import Chunk, chunk_document
from .embeddings import MixedbreadClient, env_embedding_config
from .sources import SourceDocument, load_manifest, sync_sources


BUILD_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def build_index(
    *,
    workspace_root: Path,
    index_dir: Path,
    manifest_path: Path | None = None,
    refresh_sources: bool = False,
    build_dense: bool = True,
    embed_batch_size: int = 32,
) -> dict[str, object]:
    index_dir = index_dir.resolve()
    cache_dir = index_dir / "cache"
    db_path = index_dir / "index.sqlite3"
    manifest = load_manifest(manifest_path, workspace_root)
    documents = sync_sources(manifest, cache_dir=cache_dir, refresh=refresh_sources)
    chunks = _collect_chunks(documents)

    index_dir.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    try:
        _init_schema(conn)
        _insert_documents(conn, documents)
        _insert_chunks(conn, chunks)
        dense_ready = False
        embed_model = None
        if build_dense:
            client = MixedbreadClient(env_embedding_config())
            if client.available and chunks:
                vectors = client.embed_texts([chunk.text for chunk in chunks], is_query=False, batch_size=embed_batch_size)
                _insert_embeddings(conn, chunks, vectors, model=client.config.model)
                dense_ready = True
                embed_model = client.config.model
        conn.commit()
    finally:
        conn.close()

    build_summary = {
        "built_at": datetime.now(UTC).strftime(BUILD_TIMESTAMP_FORMAT),
        "documents": len(documents),
        "chunks": len(chunks),
        "dense_ready": dense_ready,
        "embedding_model": embed_model,
        "index_dir": str(index_dir),
    }
    (index_dir / "build_summary.json").write_text(
        json.dumps(build_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (index_dir / "manifest.lock.json").write_text(
        json.dumps({"sources": [asdict(spec) for spec in manifest]}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return build_summary


def open_index(index_dir: Path) -> sqlite3.Connection:
    db_path = index_dir / "index.sqlite3"
    if not db_path.exists():
        raise FileNotFoundError(f"index database not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def index_summary(index_dir: Path) -> dict[str, object]:
    summary_path = index_dir / "build_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["index_dir"] = str(index_dir.resolve())
        return payload
    with open_index(index_dir) as conn:
        document_count = conn.execute("SELECT COUNT(*) AS count FROM documents").fetchone()["count"]
        chunk_count = conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()["count"]
        dense_count = conn.execute("SELECT COUNT(*) AS count FROM chunk_embeddings").fetchone()["count"]
    return {
        "documents": int(document_count),
        "chunks": int(chunk_count),
        "dense_ready": bool(dense_count),
        "index_dir": str(index_dir.resolve()),
    }


def load_dense_vectors(index_dir: Path) -> tuple[list[str], list[list[float]], str | None]:
    with open_index(index_dir) as conn:
        rows = conn.execute(
            "SELECT chunk_id, vector, dims, model FROM chunk_embeddings ORDER BY chunk_id"
        ).fetchall()
    chunk_ids: list[str] = []
    vectors: list[list[float]] = []
    model: str | None = None
    for row in rows:
        vec = array("f")
        vec.frombytes(row["vector"])
        if row["dims"] != len(vec):
            raise RuntimeError(f"corrupt vector for chunk {row['chunk_id']}: dims mismatch")
        chunk_ids.append(str(row["chunk_id"]))
        vectors.append(list(vec))
        if model is None:
            model = str(row["model"]) if row["model"] else None
    return chunk_ids, vectors, model


def _collect_chunks(documents: list[SourceDocument]) -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for document in documents:
        for chunk in chunk_document(document):
            if chunk.chunk_id in seen:
                continue
            seen.add(chunk.chunk_id)
            chunks.append(chunk)
    return chunks


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;

        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            canonical_uri TEXT NOT NULL,
            display_path TEXT NOT NULL,
            file_path TEXT NOT NULL,
            git_commit TEXT,
            content_hash TEXT NOT NULL
        );

        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            canonical_uri TEXT NOT NULL,
            display_path TEXT NOT NULL,
            file_path TEXT NOT NULL,
            symbol TEXT,
            chunk_kind TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL,
            git_commit TEXT,
            content_hash TEXT NOT NULL,
            text TEXT NOT NULL,
            search_text TEXT NOT NULL,
            exact_tokens_json TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,
            search_text
        );

        CREATE TABLE chunk_embeddings (
            chunk_id TEXT PRIMARY KEY,
            dims INTEGER NOT NULL,
            vector BLOB NOT NULL,
            model TEXT NOT NULL
        );

        CREATE INDEX idx_chunks_doc_id ON chunks(doc_id);
        CREATE INDEX idx_chunks_file_path ON chunks(file_path);
        CREATE INDEX idx_chunks_symbol ON chunks(symbol);
        """
    )


def _insert_documents(conn: sqlite3.Connection, documents: list[SourceDocument]) -> None:
    conn.executemany(
        """
        INSERT INTO documents (
            doc_id, source_id, kind, title, canonical_uri, display_path,
            file_path, git_commit, content_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                document.doc_id,
                document.source_id,
                document.kind,
                document.title,
                document.canonical_uri,
                document.display_path,
                document.file_path,
                document.commit,
                document.content_hash,
            )
            for document in documents
        ],
    )


def _insert_chunks(conn: sqlite3.Connection, chunks: list[Chunk]) -> None:
    conn.executemany(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, source_id, canonical_uri, display_path, file_path,
            symbol, chunk_kind, start_line, end_line, git_commit, content_hash,
            text, search_text, exact_tokens_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                chunk.chunk_id,
                chunk.doc_id,
                chunk.source_id,
                chunk.canonical_uri,
                chunk.display_path,
                chunk.file_path,
                chunk.symbol,
                chunk.chunk_kind,
                chunk.start_line,
                chunk.end_line,
                chunk.commit,
                chunk.content_hash,
                chunk.text,
                chunk.search_text,
                json.dumps(chunk.exact_tokens),
            )
            for chunk in chunks
        ],
    )
    conn.executemany(
        "INSERT INTO chunks_fts (chunk_id, search_text) VALUES (?, ?)",
        [(chunk.chunk_id, chunk.search_text) for chunk in chunks],
    )


def _insert_embeddings(
    conn: sqlite3.Connection,
    chunks: list[Chunk],
    vectors: list[list[float]],
    *,
    model: str,
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("chunk/vector count mismatch")
    rows = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        packed = array("f", vector).tobytes()
        rows.append((chunk.chunk_id, len(vector), packed, model))
    conn.executemany(
        "INSERT INTO chunk_embeddings (chunk_id, dims, vector, model) VALUES (?, ?, ?, ?)",
        rows,
    )
