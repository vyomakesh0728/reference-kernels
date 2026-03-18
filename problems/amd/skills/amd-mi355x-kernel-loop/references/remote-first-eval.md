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
python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop submit --variant <name> --source <submission.py> --lane <A|B|A+B> --stage leaderboard
```

Current coordinator behavior from code:

- local preflight is static-only by default
- `test` and `benchmark` are treated as a shared bucket by the local governor
- `leaderboard` is gated before UTC minute `45`
- the ledger lives at [/Users/v/reference-kernels/problems/amd/.agent-loop/closed_loop/mxfp4_mm/experiment_ledger.jsonl](/Users/v/reference-kernels/problems/amd/.agent-loop/closed_loop/mxfp4_mm/experiment_ledger.jsonl)

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
