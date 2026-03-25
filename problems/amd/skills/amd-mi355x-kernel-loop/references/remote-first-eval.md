# Remote-First Evaluation

## Default Policy

Use the remote cluster as the real validator. Use local checks only for:

- syntax
- purity scan
- shape-gate sanity
- lightweight file inspection

Do not block on local Docker parity unless the user explicitly asks for it.

## Current `mxfp4_mm` Coordinator

Main entrypoint:

```bash
python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop status --report
```

Useful commands:

```bash
python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop preflight --variant <name> --source <submission.py> --lane <A|B|A+B> --hypothesis "<one line>" --expected-gain "<one line>" --next-patch "<one line>" --runtime none
python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop submit --variant <name> --source <submission.py> --lane <A|B|A+B> --stage test
python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop submit --variant <name> --source <submission.py> --lane <A|B|A+B> --stage benchmark
python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop submit --variant <name> --source <submission.py> --lane <A|B|A+B> --stage profile_rocprof
python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop submit --variant <name> --source <submission.py> --lane <A|B|A+B> --stage leaderboard
```

Quota watcher helper:

```bash
python3 /Users/v/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/scripts/quota_watch_resume.py \
  --repo /Users/v/reference-kernels/problems/amd \
  --variant <name> \
  --source <submission.py> \
  --lane <A|B|A+B> \
  --stage <test|benchmark|profile_rocprof|leaderboard>
```

Current coordinator behavior from code:

- local preflight is static-only by default
- `test` and `benchmark` are treated as a shared bucket by the local governor
- `profile_rocprof` currently shares that same remote-spend bucket conservatively
- `leaderboard` is gated before UTC minute `45`
- for `mxfp4_mm`, leaderboard uses a different seeded input population than `test` and `benchmark`, so treat ranked as a separate cross-seed gate
- `profile_rocprof` uses `popcorn-cli --mode profile` plus `POPCORN_PROFILE_BACKEND=rocprofv3` and materializes:
  - `profile/profile_summary.json`
  - `profile/candidate_cards.json`
  - `profile/raw/...` decoded CSV artifacts when the custom payload lands
  - the downloaded kernelbot `profile_*.zip` artifact is also part of the contract and must be mined when kernelbot returns its built-in rocPROF trace output instead of the custom PMC payload
- the current exact-shape trunk also exposes ROCTx ranges per shape so profile traces stay separable without reopening shared-path instrumentation work
- the ledger lives at [/Users/v/reference-kernels/problems/amd/.agent-loop/closed_loop/mxfp4_mm/experiment_ledger.jsonl](/Users/v/reference-kernels/problems/amd/.agent-loop/closed_loop/mxfp4_mm/experiment_ledger.jsonl)

## Quota Exhaustion Workflow

When the coordinator says quota is exhausted:

1. Record the pending branch in the frontier note before moving on.
2. Start the quota watcher helper in the background.
3. Let the watcher sleep until the exact next eligible slot and auto-submit the pending stage.
4. The watcher writes a PID file and metadata under:
   [/Users/v/reference-kernels/problems/amd/.agent-loop/quota_watch](/Users/v/reference-kernels/problems/amd/.agent-loop/quota_watch)

Operational rule:

- do not manually busy-poll quota after a clean watcher is running
- use the freed time for retrieval, canon updates, and next-branch design
- prefer one watcher per pending variant/stage rather than stacking duplicate submitters
- treat `profile_rocprof` the same way if a benchmark winner is waiting for hardware-counter confirmation

## Profile Zip Mining

Current kernelbot behavior for `--mode profile` can return a built-in rocPROF trace zip instead of the custom `rocprofv3` PMC payload. Treat that as the live profile source, not as a failed run.

Operational rule:

1. read `profile_*.zip`
2. mine `kernel_trace.csv`, `hip_api_trace.csv`, and `agent_info.csv`
3. let the fallback parser write:
   - `profile/profile_summary.json`
   - `profile/candidate_cards.json`
4. only then open the next branch

For `mxfp4_mm`, the current fallback parser already extracts:

- per-shape cost buckets from kernelbot profile markdown
- trace metadata like `VGPR`, `SGPR`, workgroup size, grid size, and LDS block size from the zip
- zip-derived Candidate Cards that obey the exact-shape cost-center gate

## Broader Harness Commands

Use these when the closed-loop coordinator is not the right tool:

```bash
python3 -m agent_loop harness-run --problem <problem> --source <submission.py> --label <label> --stages test
python3 -m agent_loop harness-run --problem <problem> --source <submission.py> --label <label> --stages benchmark
python3 -m agent_loop harness-summary --problem <problem>
python3 -m agent_loop harness-resume --problem <problem>
```

## Promotion Rules

- `test` first
- `benchmark` second
- `profile_rocprof` third for benchmark winners that need counter evidence
- `leaderboard` only for a candidate that already has a measured reason to win

Do not spend leaderboard slots on:
- contract guesses
- wrapper-only reshapes
- broad rewrites
- candidates that did not already beat the current measured base

## Result Locations

- transient staged output:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs)
- shareable summaries:
  [/Users/v/reference-kernels/problems/amd/team_results](/Users/v/reference-kernels/problems/amd/team_results)

## Team Logging

When you get a real win:

1. keep the source candidate under `.agent-loop/manual/`
2. write or update a `team_results/.../summary.md`
3. use [$commit-push](/Users/v/.codex/skills/commit-push/SKILL.md) only if the user explicitly asks to commit or push
