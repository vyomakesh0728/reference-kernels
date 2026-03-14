from __future__ import annotations

import argparse
from pathlib import Path
import sys

import duckdb


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one KernelBot data query against local parquet files.")
    parser.add_argument("--data-dir", required=True, help="Directory containing kernelbot-data parquet files.")
    parser.add_argument("--query", required=True, help="Path to a .sql file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir).expanduser().resolve()
    query_path = Path(args.query).expanduser().resolve()
    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")
    if not query_path.exists():
        raise SystemExit(f"query file not found: {query_path}")

    con = duckdb.connect()
    views = {
        "submissions": data_dir / "submissions.parquet",
        "successful_submissions": data_dir / "successful_submissions.parquet",
        "deduplicated_submissions": data_dir / "deduplicated_submissions.parquet",
        "deduplicated_successful_submissions": data_dir / "deduplicated_successful_submissions.parquet",
        "leaderboards": data_dir / "leaderboards.parquet",
    }

    for name, path in views.items():
        if path.exists():
            con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path.as_posix()}')")

    sql = query_path.read_text(encoding="utf-8")
    rows = con.execute(sql).fetchall()
    columns = [item[0] for item in con.description]

    print("\t".join(columns))
    for row in rows:
        print("\t".join("" if value is None else str(value) for value in row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
