# Transfer To MOE And MLA

## Keep The Same Meta-Workflow

Reuse the `mxfp4_mm` playbook in this order:

1. Read `task.py`, `reference.py`, `task.yml`, and the current `submission.py`.
2. Build a correctness-first real candidate before chasing speed.
3. Keep one hypothesis per run.
4. Use retrieval for low-level contract questions.
5. Use dataset mining for failure priors and older code patterns.
6. Use remote `test` before `benchmark`.

Do not copy `mxfp4` kernels blindly. Transfer the workflow, not the exact code shape.

## MOE (`/Users/v/reference-kernels/problems/amd/moe`)

Treat MOE as at least three separate levers:

1. routing and expert selection
2. expert GEMM data movement and quant/dequant contract
3. activation / shared-expert / reduction fusion

Recommended decomposition:

- Keep routing correctness isolated first.
- Keep expert GEMM contract questions separate from activation fusion.
- Split small-expert or sparse-load regimes from heavy expert-throughput regimes.
- Use retrieval when the expert GEMM path hits AMD ISA or MFMA contract questions.
- Use dataset mining when you need older AMD MoE failure signatures or successful code priors.

Start files:
- [/Users/v/reference-kernels/problems/amd/moe/task.py](/Users/v/reference-kernels/problems/amd/moe/task.py)
- [/Users/v/reference-kernels/problems/amd/moe/reference.py](/Users/v/reference-kernels/problems/amd/moe/reference.py)
- [/Users/v/reference-kernels/problems/amd/moe/task.yml](/Users/v/reference-kernels/problems/amd/moe/task.yml)
- [/Users/v/reference-kernels/problems/amd/moe/submission.py](/Users/v/reference-kernels/problems/amd/moe/submission.py)

## MLA (`/Users/v/reference-kernels/problems/amd/mla-decode`)

Treat MLA as a decode-latency and bandwidth problem first:

1. KV load and format/dequant path
2. attention core math
3. split-K / MQA reduction and writeback

Recommended decomposition:

- Fix contract and data format first.
- Then shrink data movement.
- Then tune the attention core.
- Only then revisit broader scheduling ideas.

For MLA, prioritize:
- bandwidth
- decode-latency
- split-K discipline
- KV format correctness

Start files:
- [/Users/v/reference-kernels/problems/amd/mla-decode/task.py](/Users/v/reference-kernels/problems/amd/mla-decode/task.py)
- [/Users/v/reference-kernels/problems/amd/mla-decode/reference.py](/Users/v/reference-kernels/problems/amd/mla-decode/reference.py)
- [/Users/v/reference-kernels/problems/amd/mla-decode/task.yml](/Users/v/reference-kernels/problems/amd/mla-decode/task.yml)
- [/Users/v/reference-kernels/problems/amd/mla-decode/README.md](/Users/v/reference-kernels/problems/amd/mla-decode/README.md)
- [/Users/v/reference-kernels/problems/amd/mla-decode/submission.py](/Users/v/reference-kernels/problems/amd/mla-decode/submission.py)

## Identity

Use `/Users/v/reference-kernels/problems/amd/identity` as a cheap harness sanity problem when you need to verify loop behavior, parsing, or promotion mechanics without burning time on the harder kernels.

## Transfer Rules

- Preserve lane or regime separation whenever the problem naturally has thin vs wide, sparse vs dense, or latency vs throughput regimes.
- Keep correctness probes and performance probes separate.
- Do not assume square-GEMM ladder steps transfer directly to decode-style shapes.
- Reuse the remote-first budget discipline and team-results logging even when the kernel internals change completely.
