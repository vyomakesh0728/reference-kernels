---
name: amd-live-reference-correctness
description: Recover live KernelBot reference correctness for amd-mxfp4-mm HIP submission.py candidates in /Users/v/reference-kernels/problems/amd. Use when mxfp4_mm harness/test runs fail with mismatch counts, when comparing fp8-mm/task.yml or reference.py against the live MI355X contract, when writing or reviewing HIP-only correctness-first candidates, or when planning the next single semantic experiment to reduce live mismatches.
---

# AMD Live Reference Correctness

Use this skill to debug `mxfp4_mm` correctness against the live `amd-mxfp4-mm` judge before any performance tuning.

The same workflow generalizes to `moe_mxfp4` and `mixed_mla`: use the public/open-source reference path as the starting spec, but treat live KernelBot evidence as the final truth when the public path drifts.

## Workflow

1. Start from evidence, not assumptions.
   Run:
   `python3 /Users/v/reference-kernels/problems/amd/skills/amd-live-reference-correctness/scripts/summarize_mxfp4_mm_harness.py --repo /Users/v/reference-kernels/problems/amd`

2. Read the contract note before proposing a fix.
   Read `references/mxfp4-mm-live-contract.md`.
   Use `fp8-mm/task.yml` only for shape envelopes. Do not treat its old FP8 wording as the live semantic contract.

3. Keep real candidates HIP-only.
   Real `submission.py` candidates should execute HIP in `custom_kernel`.
   Use `torch.mm` or `aiter.gemm_a4w4` only in diagnostic probes under `.agent-loop/problems/mxfp4_mm/manual/`; do not treat those as final candidates.

4. Stay in `test` mode until one HIP candidate passes.
   Run:
   `python3 -m agent_loop harness-run --problem mxfp4_mm --source <submission.py> --family hip_explore --label <label> --stages test`

5. Change one semantic axis per run.
   Prefer a single hypothesis about `A` representation, `A` scale source, `B_q` meaning, `B` scale source, or shuffle/layout interpretation.
   Read `references/mxfp4-mm-correctness-playbook.md` when deciding the next axis.

6. Optimize only after zero mismatch.
   When a HIP candidate reaches `status=ok`, then move to `benchmark`, then `leaderboard`.

## What Got MM To Green

Reuse these lessons before re-deriving them:

- The harness was not the blocker once staged `test -> benchmark -> leaderboard` was in place. The blocker was live-reference correctness.
- Public AITER semantics were necessary but not sufficient. `mxfp4_mm` only went green after reconciling against live KernelBot output, not by trusting `fp8-mm/task.yml` or the public baseline alone.
- One semantic axis per run was critical. Useful axes were:
  - `A` shuffle false vs true
  - raw `A/B` vs provided `b_q`
  - raw B scale vs provided `b_scale_sh`
  - Triton quantizer vs HIP quantizer
  - packed-byte / nibble transform probes
- Keep diagnostic oracles outside the final candidate if needed, but keep the submitted `submission.py` HIP-only in `custom_kernel`.
- The winning correctness path came from using live `b_q` as calibration data:
  - public Triton raw quantization stayed the closest public spec
  - live `b_shuffle` matched `shuffle_weight(input b_q)`
  - live `b_scale_sh` matched Triton-style shuffled raw scale
  - live `b_q` still drifted from the public raw quant output
  - we learned small adjustment rules from live `B` and applied them to `A`
- The first green HIP reference was intentionally slow:
  - reconstruct logical `A_ref` and `B_ref` in Python
  - run a simple tiled HIP GEMM body
  - prove correctness first

## Reusable Correctness Pattern For MOE / MLA

When correctness is failing on another problem, apply this pattern:

1. Freeze optimization and use `test` only.
2. Identify the public/open-source contract source first.
3. Build a correctness-first HIP candidate whose hot path is real HIP, even if inputs are reconstructed in Python.
4. Compare live-provided tensors against public packers/quantizers tensor-by-tensor.
5. Run one semantic delta at a time and keep a harness artifact for each probe.
6. If public and local outputs match each other but fail live, assume live-reference drift and use the provided live tensors to learn the missing contract behavior.
7. Only switch to `benchmark` after the first `status=ok` test run.

## Non-Negotiables

- Preserve high-fidelity benchmarking rules; no stream/timing hacks.
- Stop broad rewrites. Prefer one contract hypothesis per run.
- If multiple branches tie the best mismatch, prefer the one closest to the public AITER reference construction documented in the references.
