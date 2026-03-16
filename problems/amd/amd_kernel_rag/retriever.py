from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import sqlite3

from .chunking import normalized_query_terms, query_exact_tokens
from .embeddings import MixedbreadClient, env_embedding_config
from .index import load_dense_vectors, open_index

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional acceleration
    np = None


RRF_K = 60


@dataclass(frozen=True)
class SearchResult:
    rank: int
    chunk_id: str
    source_id: str
    canonical_uri: str
    display_path: str
    file_path: str
    symbol: str | None
    start_line: int
    end_line: int
    text: str
    exact_tokens: list[str]
    sparse_rank: int | None
    dense_rank: int | None
    exact_boost: float
    fused_score: float

    @property
    def citation(self) -> str:
        symbol = f"::{self.symbol}" if self.symbol else ""
        return f"{self.display_path}{symbol}:{self.start_line}-{self.end_line}"


class HybridRetriever:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir.resolve()
        self._conn: sqlite3.Connection | None = None
        self._dense_cache: tuple[list[str], list[list[float]], str | None] | None = None
        self._client = MixedbreadClient(env_embedding_config())

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        sparse_k: int = 64,
        dense_k: int = 64,
    ) -> list[SearchResult]:
        sparse_rows = self._sparse_search(query, limit=sparse_k)
        dense_rows = self._dense_search(query, limit=dense_k)
        query_tokens = query_exact_tokens(query)

        fused: dict[str, dict[str, object]] = {}
        for rank, row in enumerate(sparse_rows, start=1):
            entry = fused.setdefault(row["chunk_id"], {"row": row, "sparse_rank": None, "dense_rank": None})
            entry["sparse_rank"] = rank
        for rank, row in enumerate(dense_rows, start=1):
            entry = fused.setdefault(row["chunk_id"], {"row": row, "sparse_rank": None, "dense_rank": None})
            entry["dense_rank"] = rank
            if entry["row"] is None:
                entry["row"] = row

        ranked: list[SearchResult] = []
        for chunk_id, entry in fused.items():
            row = entry["row"]
            sparse_rank = entry["sparse_rank"]
            dense_rank = entry["dense_rank"]
            exact_tokens = json.loads(row["exact_tokens_json"])
            exact_boost = _exact_token_boost(query_tokens, exact_tokens)
            fused_score = _rrf_score(sparse_rank) + _rrf_score(dense_rank) + exact_boost
            ranked.append(
                SearchResult(
                    rank=0,
                    chunk_id=chunk_id,
                    source_id=str(row["source_id"]),
                    canonical_uri=str(row["canonical_uri"]),
                    display_path=str(row["display_path"]),
                    file_path=str(row["file_path"]),
                    symbol=str(row["symbol"]) if row["symbol"] else None,
                    start_line=int(row["start_line"]),
                    end_line=int(row["end_line"]),
                    text=str(row["text"]),
                    exact_tokens=list(exact_tokens),
                    sparse_rank=int(sparse_rank) if sparse_rank else None,
                    dense_rank=int(dense_rank) if dense_rank else None,
                    exact_boost=exact_boost,
                    fused_score=fused_score,
                )
            )
        ranked.sort(key=lambda item: item.fused_score, reverse=True)
        final: list[SearchResult] = []
        for rank, item in enumerate(ranked[:top_k], start=1):
            final.append(
                SearchResult(
                    rank=rank,
                    chunk_id=item.chunk_id,
                    source_id=item.source_id,
                    canonical_uri=item.canonical_uri,
                    display_path=item.display_path,
                    file_path=item.file_path,
                    symbol=item.symbol,
                    start_line=item.start_line,
                    end_line=item.end_line,
                    text=item.text,
                    exact_tokens=item.exact_tokens,
                    sparse_rank=item.sparse_rank,
                    dense_rank=item.dense_rank,
                    exact_boost=item.exact_boost,
                    fused_score=item.fused_score,
                )
            )
        return final

    def grounded_answer(self, query: str, *, top_k: int = 5) -> str:
        results = self.search(query, top_k=top_k)
        if not results:
            return "No matching chunks found."
        snippets: list[str] = []
        keywords = set(normalized_query_terms(query))
        for result in results:
            lines = _salient_lines(result.text, keywords)
            snippets.append(
                "\n".join(
                    [
                        f"[{result.rank}] {result.citation}",
                        *lines,
                        f"Source: {result.canonical_uri}",
                    ]
                ).strip()
            )
        return "\n\n".join(snippets)

    def _sparse_search(self, query: str, *, limit: int) -> list[sqlite3.Row]:
        terms = normalized_query_terms(query)
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms[:24])
        conn = self._conn_or_open()
        rows = conn.execute(
            """
            SELECT
                c.chunk_id,
                c.source_id,
                c.canonical_uri,
                c.display_path,
                c.file_path,
                c.symbol,
                c.start_line,
                c.end_line,
                c.text,
                c.exact_tokens_json,
                bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks AS c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score ASC
            LIMIT ?
            """,
            (fts_query, limit),
        ).fetchall()
        return rows

    def _dense_search(self, query: str, *, limit: int) -> list[sqlite3.Row]:
        if not self._client.available:
            return []
        chunk_ids, vectors, _ = self._load_dense()
        if not chunk_ids:
            return []
        query_vector = self._client.embed_texts([query], is_query=True, batch_size=1)[0]
        if np is not None and not isinstance(vectors, list):
            query_array = np.asarray(query_vector, dtype=np.float32)
            scores = vectors @ query_array
            top_count = min(limit, int(scores.shape[0]))
            if top_count <= 0:
                return []
            top_indices = np.argpartition(scores, -top_count)[-top_count:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
            chosen = [chunk_ids[int(index)] for index in top_indices.tolist()]
        else:
            scored: list[tuple[float, str]] = []
            for chunk_id, vector in zip(chunk_ids, vectors, strict=True):
                scored.append((_dot(query_vector, vector), chunk_id))
            scored.sort(key=lambda item: item[0], reverse=True)
            chosen = [chunk_id for _, chunk_id in scored[:limit]]
        if not chosen:
            return []
        conn = self._conn_or_open()
        placeholders = ",".join("?" for _ in chosen)
        rows = conn.execute(
            f"""
            SELECT
                chunk_id,
                source_id,
                canonical_uri,
                display_path,
                file_path,
                symbol,
                start_line,
                end_line,
                text,
                exact_tokens_json
            FROM chunks
            WHERE chunk_id IN ({placeholders})
            """,
            chosen,
        ).fetchall()
        by_id = {str(row["chunk_id"]): row for row in rows}
        return [by_id[chunk_id] for chunk_id in chosen if chunk_id in by_id]

    def _load_dense(self) -> tuple[list[str], object, str | None]:
        if self._dense_cache is None:
            chunk_ids, vectors, model = load_dense_vectors(self.index_dir)
            if np is not None and vectors:
                dense_matrix = np.asarray(vectors, dtype=np.float32)
                self._dense_cache = (chunk_ids, dense_matrix, model)
            else:
                self._dense_cache = (chunk_ids, vectors, model)
        return self._dense_cache

    def _conn_or_open(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = open_index(self.index_dir)
        return self._conn


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _rrf_score(rank: int | None) -> float:
    if rank is None:
        return 0.0
    return 1.0 / (RRF_K + rank)


def _exact_token_boost(query_tokens: list[str], chunk_tokens: list[str]) -> float:
    if not query_tokens or not chunk_tokens:
        return 0.0
    chunk_set = set(chunk_tokens)
    boost = 0.0
    for token in query_tokens:
        if token not in chunk_set:
            continue
        if token.startswith("__builtin_amdgcn_") or token.startswith("llvm.amdgcn."):
            boost += 0.08
        elif "mfma" in token:
            boost += 0.06
        elif token.startswith(("v_", "s_", "buffer_", "flat_", "global_", "ds_")):
            boost += 0.04
        else:
            boost += 0.02
    return boost


def _salient_lines(text: str, keywords: set[str]) -> list[str]:
    picked: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        lowered = stripped.lower()
        if keywords and any(keyword in lowered for keyword in keywords):
            picked.append(stripped)
        if len(picked) == 4:
            break
    if not picked:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                picked.append(stripped)
            if len(picked) == 4:
                break
    return picked
