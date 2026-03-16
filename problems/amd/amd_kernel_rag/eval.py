from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

from .retriever import HybridRetriever, SearchResult


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    query: str
    relevant_any_of: list[dict[str, object]]
    why_hard: str


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []
    for item in raw:
        cases.append(
            BenchmarkCase(
                id=str(item["id"]),
                query=str(item["query"]),
                relevant_any_of=list(item["relevant_any_of"]),
                why_hard=str(item.get("why_hard", "")),
            )
        )
    return cases


def run_benchmark(
    *,
    index_dir: Path,
    benchmark_path: Path,
    top_k: int = 20,
) -> dict[str, object]:
    retriever = HybridRetriever(index_dir)
    cases = load_benchmark(benchmark_path)
    per_query: list[dict[str, object]] = []
    try:
        for case in cases:
            results = retriever.search(case.query, top_k=top_k)
            first_relevant_rank = _first_relevant_rank(results, case.relevant_any_of)
            per_query.append(
                {
                    "id": case.id,
                    "query": case.query,
                    "why_hard": case.why_hard,
                    "first_relevant_rank": first_relevant_rank,
                    "recall_at_5": bool(first_relevant_rank and first_relevant_rank <= 5),
                    "recall_at_20": bool(first_relevant_rank and first_relevant_rank <= 20),
                    "mrr_at_10": (1.0 / first_relevant_rank) if first_relevant_rank and first_relevant_rank <= 10 else 0.0,
                    "top_results": [_result_brief(result) for result in results[:5]],
                }
            )
    finally:
        retriever.close()

    total = len(per_query) or 1
    recall_at_5 = sum(1 for item in per_query if item["recall_at_5"]) / total
    recall_at_20 = sum(1 for item in per_query if item["recall_at_20"]) / total
    mrr_at_10 = sum(float(item["mrr_at_10"]) for item in per_query) / total
    misses = [
        item
        for item in sorted(
            per_query,
            key=lambda item: item["first_relevant_rank"] if item["first_relevant_rank"] is not None else 10_000,
        )
        if not item["recall_at_20"]
    ]
    return {
        "summary": {
            "queries": len(per_query),
            "recall_at_5": round(recall_at_5, 4),
            "recall_at_20": round(recall_at_20, 4),
            "mrr_at_10": round(mrr_at_10, 4),
        },
        "top_misses": misses[:10],
        "per_query": per_query,
    }


def write_report(report: dict[str, object], path: Path) -> Path:
    summary = report["summary"]
    misses = report["top_misses"]
    lines = [
        "# AMD Kernel RAG Evaluation",
        "",
        f"- Queries: {summary['queries']}",
        f"- Recall@5: {summary['recall_at_5']}",
        f"- Recall@20: {summary['recall_at_20']}",
        f"- MRR@10: {summary['mrr_at_10']}",
        "",
        "## Top Misses",
        "",
    ]
    if not misses:
        lines.append("- No misses at Recall@20 in this run.")
    for miss in misses:
        lines.extend(
            [
                f"### {miss['id']}",
                "",
                f"- Query: `{miss['query']}`",
                f"- Why hard: {miss['why_hard']}",
                "- Top retrieved chunks:",
                *[f"  - {item['citation']} ({item['source']})" for item in miss["top_results"]],
                "",
            ]
        )
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return path


def _first_relevant_rank(results: list[SearchResult], rules: list[dict[str, object]]) -> int | None:
    for result in results:
        if _matches_any_rule(result, rules):
            return result.rank
    return None


def _matches_any_rule(result: SearchResult, rules: list[dict[str, object]]) -> bool:
    for rule in rules:
        path_contains = str(rule.get("path_contains", "")).lower()
        symbol_contains = str(rule.get("symbol_contains", "")).lower()
        text_contains = str(rule.get("text_contains", "")).lower()
        source_contains = str(rule.get("source_contains", "")).lower()
        if path_contains and path_contains not in result.display_path.lower() and path_contains not in result.canonical_uri.lower():
            continue
        if symbol_contains and symbol_contains not in (result.symbol or "").lower():
            continue
        if text_contains and text_contains not in result.text.lower():
            continue
        if source_contains and source_contains not in result.source_id.lower():
            continue
        return True
    return False


def _result_brief(result: SearchResult) -> dict[str, object]:
    return {
        "rank": result.rank,
        "citation": result.citation,
        "source": result.canonical_uri,
    }

