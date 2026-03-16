from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .__init__ import DEFAULT_INDEX_RELATIVE_DIR
from .eval import run_benchmark, write_report
from .index import build_index, index_summary
from .retriever import HybridRetriever
from .sources import write_default_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amd-kernel-rag")
    parser.add_argument(
        "--workspace-root",
        default=".",
        help="AMD workspace root. Defaults to the current directory.",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Index directory. Defaults to <workspace-root>/" + str(DEFAULT_INDEX_RELATIVE_DIR),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("write-default-manifest", help="Write the default source manifest JSON")
    manifest.add_argument("path", help="Where to write the manifest JSON")

    build = sub.add_parser("build", help="Fetch sources, chunk them, and build the local hybrid index")
    build.add_argument("--manifest", help="Optional source manifest JSON")
    build.add_argument("--refresh-sources", action="store_true")
    build.add_argument("--skip-dense", action="store_true")
    build.add_argument("--embed-batch-size", type=int, default=32)

    summary = sub.add_parser("summary", help="Show index stats")

    query = sub.add_parser("query", help="Run hybrid retrieval and print the top chunks")
    query.add_argument("--query", required=True)
    query.add_argument("--top-k", type=int, default=8)
    query.add_argument("--json", action="store_true")

    answer = sub.add_parser("answer", help="Print a grounded extractive answer with citations")
    answer.add_argument("--query", required=True)
    answer.add_argument("--top-k", type=int, default=5)

    evaluate = sub.add_parser("eval", help="Run the benchmark set and optionally write a report")
    evaluate.add_argument("--benchmark", required=True)
    evaluate.add_argument("--top-k", type=int, default=20)
    evaluate.add_argument("--report-out")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    default_base = workspace_root / ".agent-loop" if (workspace_root / ".agent-loop").exists() else workspace_root
    index_dir = (
        Path(args.index_dir).expanduser().resolve()
        if args.index_dir
        else (default_base / DEFAULT_INDEX_RELATIVE_DIR).resolve()
    )

    if args.command == "write-default-manifest":
        path = write_default_manifest(Path(args.path).expanduser(), workspace_root)
        print(str(path))
        return 0

    if args.command == "build":
        summary = build_index(
            workspace_root=workspace_root,
            index_dir=index_dir,
            manifest_path=Path(args.manifest).expanduser().resolve() if args.manifest else None,
            refresh_sources=bool(args.refresh_sources),
            build_dense=not args.skip_dense,
            embed_batch_size=args.embed_batch_size,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "summary":
        print(json.dumps(index_summary(index_dir), indent=2, sort_keys=True))
        return 0

    if args.command == "query":
        retriever = HybridRetriever(index_dir)
        try:
            results = retriever.search(args.query, top_k=args.top_k)
        finally:
            retriever.close()
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "rank": result.rank,
                            "citation": result.citation,
                            "source": result.canonical_uri,
                            "score": result.fused_score,
                            "text": result.text,
                        }
                        for result in results
                    ],
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        for result in results:
            print(f"[{result.rank}] {result.citation}")
            print(f"Source: {result.canonical_uri}")
            print(result.text)
            print()
        return 0

    if args.command == "answer":
        retriever = HybridRetriever(index_dir)
        try:
            print(retriever.grounded_answer(args.query, top_k=args.top_k))
        finally:
            retriever.close()
        return 0

    if args.command == "eval":
        report = run_benchmark(
            index_dir=index_dir,
            benchmark_path=Path(args.benchmark).expanduser().resolve(),
            top_k=args.top_k,
        )
        print(json.dumps(report["summary"], indent=2, sort_keys=True))
        if args.report_out:
            path = write_report(report, Path(args.report_out).expanduser().resolve())
            print(str(path))
        return 0

    parser.error(f"unknown command: {args.command}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
