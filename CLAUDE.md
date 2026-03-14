# AMD Kernel Hackathon -- reference-kernels

## What This Is
AMD $100K kernel competition via GPU MODE. Team: code1 (vyom, elliot, pritam).
Target GPU: MI355X. Deadline: March 30, 2026. Phase 2 if selected: mid-April.

## How It Works
- You write `submission.py` with a `custom_kernel(data)` function
- Submit via `popcorn-cli submit` which runs on remote MI355X cluster
- Scored by geometric mean of benchmark kernel times (lower = better)
- No local MI355X -- all eval is remote

## Three Problems (all in `problems/amd_202602/`)

### 1. mxfp4-mm (leaderboard: amd-mxfp4-mm)
MXFP4 matrix multiply. bf16 A -> quantize to MXFP4 -> gemm_a4w4 -> bf16 C.
Input: `(A, B, B_q, B_shuffle, B_scale_sh)` -- B is pre-quantized.
Current baseline: ~12-27us using `aiter.gemm_a4w4`.

### 2. moe-mxfp4 (leaderboard: amd-moe-mxfp4)
DeepSeek-R1 style MoE. 256 routed + 1 shared expert, top-8+1, SwiGLU, MXFP4 weights.
Input: 12-tuple (hidden_states, weights, scales, shuffled variants, topk_weights/ids, config).
Current baseline: ~95-352us using `aiter.fused_moe`.

### 3. mixed-mla (leaderboard: amd-mixed-mla)
MLA decode attention. 16 query heads, 1 KV head, kv_lora_rank=512, fp8 KV cache.
Input: `(q, kv_data_dict, seq_lens, page_table, config)`.
Current baseline: ~152-389us using `aiter.mla.mla_decode_fwd`.

## Agent Loop (`problems/amd/agent_loop/`)
Automated kernel search loop:
- `agent_loop.toml` -- config (API keys, problem defs, mutator commands)
- `triton_mutator.py` -- template-based Triton kernel variants
- `llm_mutator.py` -- LLM-driven mutation (OpenAI/Anthropic/OpenRouter/Codex)
- `runner.py` -- closed loop: mutate -> submit -> evaluate -> promote if better
- `evaluator.py` -- wraps `popcorn-cli submit`, parses results
- `critic.py` -- generates structured feedback from eval results

Run: `cd problems/amd && python3 -m agent_loop <command>`
Commands: healthcheck, baseline, loop, swarm, status, promote

## Key Libraries
- `aiter` -- AMD's inference library (provides gemm_a4w4, fused_moe, mla_decode_fwd)
- `triton` -- write custom GPU kernels in Python
- Both are pre-installed on the MI355X eval environment

## Build/Run
No local build needed. All eval is remote via popcorn-cli.
To test agent loop locally: `cd problems/amd && python3 -m agent_loop healthcheck`

## Submission Contract
Every submission.py MUST:
- Define `custom_kernel(data: input_t) -> output_t`
- Match the exact input/output types from `task.py`
- Pass correctness checks (rtol/atol tolerances in reference.py)
- Include `#!POPCORN` header lines if using agent loop

## Current State (2026-03-10)
All three problems have working ranked baselines using aiter library calls.
No pure Triton submission has beaten the aiter anchors yet.
The agent loop + LLM mutator infrastructure is ready for iteration.
