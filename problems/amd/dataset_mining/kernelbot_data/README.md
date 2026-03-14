# KernelBot Data Query Pack

This folder contains a small DuckDB-oriented query pack for mining the
`GPUMODE/kernelbot-data` dataset.

Current intent:
- mine AMD benchmark-only failures
- find repeated runtime-error signatures
- retrieve nearby successful kernels without guessing blindly

Important note:
- the current `kernelbot-data` snapshot is the older AMD competition set
- it contains:
  - `398 amd-identity`
  - `399 amd-fp8-mm`
  - `430 amd-mixture-of-experts`
  - `463 amd-mla-decode`
  - `563 amd-all2all`
  - `564 amd-gemm-rs`
  - `565 amd-ag-gemm`
- it does not currently include the newer MI355X `amd-mxfp4-mm` leaderboard id
  we are actively competing on

So this query pack is best used as:
- a failure-pattern memory
- a harness prior source
- a code retrieval source

It is not a direct live-reference oracle for the current `mxfp4_mm` bug.

## Files

- `run_duckdb_query.py`
  - mounts the parquet files as views and runs one SQL file
- `queries/leaderboard_inventory.sql`
  - lists the leaderboards present in the snapshot
- `queries/amd_failure_signature_counts.sql`
  - counts benchmark/runtime failure signatures across AMD leaderboards
- `queries/amd_timeout_failures.sql`
  - isolates timeout-like failures
- `queries/amd_memory_fault_failures.sql`
  - isolates memory-fault / illegal-access style failures
- `queries/amd_benchmark_only_runtime_failures.sql`
  - finds code ids that passed `test` but failed `benchmark`
- `queries/amd_benchmark_memory_fault_examples.sql`
  - fetches benchmark-only memory-fault examples with code prefixes
- `queries/amd_success_lookup.sql`
  - samples successful code rows for a chosen AMD leaderboard

## Usage

```bash
python3 dataset_mining/kernelbot_data/run_duckdb_query.py \
  --data-dir /sandbox-workspace/amd-kernel-miner/data/kernelbot-data \
  --query dataset_mining/kernelbot_data/queries/leaderboard_inventory.sql
```

You can also point it at a local mirror on macOS if the parquet files exist
there.
