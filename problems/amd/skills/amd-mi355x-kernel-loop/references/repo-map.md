# Repo Map

## Live Problem Directories

- `mxfp4_mm`: [/Users/v/reference-kernels/problems/amd/fp8-mm](/Users/v/reference-kernels/problems/amd/fp8-mm)
- `moe_mxfp4`: [/Users/v/reference-kernels/problems/amd/moe](/Users/v/reference-kernels/problems/amd/moe)
- `mixed_mla`: [/Users/v/reference-kernels/problems/amd/mla-decode](/Users/v/reference-kernels/problems/amd/mla-decode)
- `identity`: [/Users/v/reference-kernels/problems/amd/identity](/Users/v/reference-kernels/problems/amd/identity)

If a prompt references `problems/amd_202602/...`, treat it as stale and map it to the live directories above.

## Experiment State

- Manual candidates: [/Users/v/reference-kernels/problems/amd/.agent-loop/manual](/Users/v/reference-kernels/problems/amd/.agent-loop/manual)
- Harness runs: [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs)
- Handrolled experiments: [/Users/v/reference-kernels/problems/amd/.agent-loop/handrolled](/Users/v/reference-kernels/problems/amd/.agent-loop/handrolled)
- Closed-loop ledger for `mxfp4_mm`: [/Users/v/reference-kernels/problems/amd/.agent-loop/closed_loop/mxfp4_mm/experiment_ledger.jsonl](/Users/v/reference-kernels/problems/amd/.agent-loop/closed_loop/mxfp4_mm/experiment_ledger.jsonl)
- Team snapshots: [/Users/v/reference-kernels/problems/amd/team_results](/Users/v/reference-kernels/problems/amd/team_results)

Use `.agent-loop/` for transient local state. Use `team_results/` for committed summaries teammates should read.

## Retrieval Stack

- Package: [/Users/v/reference-kernels/problems/amd/amd_kernel_rag](/Users/v/reference-kernels/problems/amd/amd_kernel_rag)
- Main guide: [/Users/v/reference-kernels/problems/amd/amd_kernel_rag/README.md](/Users/v/reference-kernels/problems/amd/amd_kernel_rag/README.md)
- Source extension guide: [/Users/v/reference-kernels/problems/amd/amd_kernel_rag/ADDING_SOURCES.md](/Users/v/reference-kernels/problems/amd/amd_kernel_rag/ADDING_SOURCES.md)
- Default index path: [/Users/v/reference-kernels/problems/amd/.agent-loop/retrieval/amd-kernel-rag](/Users/v/reference-kernels/problems/amd/.agent-loop/retrieval/amd-kernel-rag)

Useful commands:

```bash
python3 -m amd_kernel_rag.cli build
python3 -m amd_kernel_rag.cli summary
python3 -m amd_kernel_rag.cli query --query "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4 operand order" --top-k 5
python3 -m amd_kernel_rag.cli answer --query "What local notes mention GPU memory faults in scaled MFMA experiments?"
python3 -m amd_kernel_rag.cli eval --benchmark amd_kernel_rag/benchmarks/kernel_queries.json --report-out amd_kernel_rag/reports/latest.md
```

Use retrieval for:
- AMDGPU intrinsics and builtins
- ISA operand contracts
- MFMA opcode details
- LLVM/Clang source-backed answers

Do not use retrieval as a substitute for live correctness evidence.

## Dataset Mining

- Query pack: [/Users/v/reference-kernels/problems/amd/dataset_mining/kernelbot_data](/Users/v/reference-kernels/problems/amd/dataset_mining/kernelbot_data)
- Guide: [/Users/v/reference-kernels/problems/amd/dataset_mining/kernelbot_data/README.md](/Users/v/reference-kernels/problems/amd/dataset_mining/kernelbot_data/README.md)

Use dataset mining for:
- repeated failure signatures
- benchmark-only runtime failures
- older AMD code examples

Do not use it as the live `mxfp4_mm` truth source. The snapshot does not include the newer MI355X `amd-mxfp4-mm` leaderboard.

## Remote Evaluation

- Agent loop guide: [/Users/v/reference-kernels/problems/amd/agent_loop/README.md](/Users/v/reference-kernels/problems/amd/agent_loop/README.md)
- Closed-loop coordinator: [/Users/v/reference-kernels/problems/amd/agent_loop/mxfp4_closed_loop.py](/Users/v/reference-kernels/problems/amd/agent_loop/mxfp4_closed_loop.py)

Current `mxfp4_mm` flow:
- static local preflight
- remote `test`
- remote `benchmark`
- remote `leaderboard`

## Repo-Local Companion Skills

- Correctness-first: [/Users/v/reference-kernels/problems/amd/skills/amd-live-reference-correctness/SKILL.md](/Users/v/reference-kernels/problems/amd/skills/amd-live-reference-correctness/SKILL.md)
- Green-to-fast optimization: [/Users/v/reference-kernels/problems/amd/skills/optimization-skill/SKILL.md](/Users/v/reference-kernels/problems/amd/skills/optimization-skill/SKILL.md)
- Older speedrun guidance: [/Users/v/reference-kernels/problems/amd/skills/amd-kernel-speedrun/SKILL.md](/Users/v/reference-kernels/problems/amd/skills/amd-kernel-speedrun/SKILL.md)

## User-Level Companion Skills

- Commit or push when asked: [$commit-push](/Users/v/.codex/skills/commit-push/SKILL.md)
- Plan only when asked: [$create-plan](/Users/v/.codex/skills/create-plan/SKILL.md)
