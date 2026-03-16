from __future__ import annotations

from pathlib import Path
import json
import os

from .__init__ import DEFAULT_INDEX_RELATIVE_DIR
from .chunking import extract_exact_tokens
from .retriever import HybridRetriever


PROBLEM_HINTS = {
    "mxfp4_mm": ["mxfp4", "scaled mfma", "operand order", "lds", "barrier", "memory fault", "cdna4"],
    "moe_mxfp4": ["mxfp4", "moe", "routing", "swiglu", "expert", "lds", "cdna4"],
    "mixed_mla": ["decode", "latency", "kv cache", "mqa", "fp8", "mxfp4", "memory fault"],
}


def default_index_dir(workspace_root: Path) -> Path:
    override = os.environ.get("AMD_KERNEL_RAG_INDEX_DIR")
    if override:
        return Path(override).expanduser().resolve()
    direct = (workspace_root / DEFAULT_INDEX_RELATIVE_DIR).resolve()
    sibling = (workspace_root.parent / DEFAULT_INDEX_RELATIVE_DIR).resolve()
    if direct.exists():
        return direct
    if sibling.exists():
        return sibling
    return direct


def build_agent_query(
    *,
    problem_key: str,
    parent_source: str,
    history: list[dict[str, object]],
    hypothesis: str | None,
) -> str:
    exact_tokens = extract_exact_tokens(parent_source)[:12]
    recent_summaries: list[str] = []
    for item in reversed(history[-4:]):
        critique = item.get("critique")
        if isinstance(critique, dict):
            summary = critique.get("summary")
            if isinstance(summary, str) and summary:
                recent_summaries.append(summary)
    hints = PROBLEM_HINTS.get(problem_key, [])
    parts = [
        f"Problem: {problem_key}",
        f"Hypothesis: {hypothesis or ''}",
        f"Failure context: {' | '.join(recent_summaries)}",
        f"AMD focus terms: {' '.join(hints)}",
        f"Exact tokens: {' '.join(exact_tokens)}",
    ]
    return "\n".join(part for part in parts if part.strip())


def maybe_write_prompt_context(
    *,
    workspace_root: Path,
    candidate_dir: Path,
    problem_key: str,
    parent_source: str,
    history: list[dict[str, object]],
    hypothesis: str | None,
    top_k: int = 6,
) -> Path | None:
    index_dir = default_index_dir(workspace_root)
    db_path = index_dir / "index.sqlite3"
    if not db_path.exists():
        return None

    query = build_agent_query(
        problem_key=problem_key,
        parent_source=parent_source,
        history=history,
        hypothesis=hypothesis,
    )
    retriever = HybridRetriever(index_dir)
    try:
        results = retriever.search(query, top_k=top_k)
    finally:
        retriever.close()
    if not results:
        return None

    payload = {
        "query": query,
        "results": [
            {
                "rank": result.rank,
                "citation": result.citation,
                "canonical_uri": result.canonical_uri,
                "symbol": result.symbol,
                "score": result.fused_score,
                "text": result.text,
            }
            for result in results
        ],
    }
    json_path = candidate_dir / "rag_context.json"
    text_path = candidate_dir / "rag_context.txt"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    text_path.write_text(_render_prompt_context(payload), encoding="utf-8")
    return text_path


def _render_prompt_context(payload: dict[str, object]) -> str:
    lines = [
        f"Retrieval query: {payload['query']}",
        "",
        "Top grounded references:",
        "",
    ]
    for result in payload["results"]:
        snippet = _compact_text(str(result["text"]))
        lines.extend(
            [
                f"[{result['rank']}] {result['citation']}",
                f"Source: {result['canonical_uri']}",
                snippet,
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _compact_text(text: str, limit_lines: int = 12, limit_chars: int = 1200) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    compact = "\n".join(lines[:limit_lines])
    if len(compact) > limit_chars:
        compact = compact[: limit_chars - 3].rstrip() + "..."
    return compact
