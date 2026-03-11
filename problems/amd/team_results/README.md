# Team Results

This folder is the shareable snapshot for the team.

It is intentionally separate from `.agent-loop/`:
- `.agent-loop/` is local experiment state, logs, SQLite data, and transient artifacts
- `team_results/` is the small committed subset we want teammates to read and diff

Current contents:
- `ranked/2026-03-10/summary.md`: human-readable snapshot of the first successful ranked submissions
- `ranked/2026-03-10/*.txt`: raw leaderboard-mode outputs copied from the local agent workspace

Important note:
- The currently promoted branch submissions are contract-faithful AITER-backed anchors
- Kernel exploration code exists in `agent_loop/kernel_mutator.py`, but the current ranked-success submissions are not pure HIP kernels yet
