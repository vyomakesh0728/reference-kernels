# mxfp4_mm Live Contract

## What to trust

- Live input tuple is `(a, b, b_q, b_shuffle, b_scale_sh)`.
- Real source of truth is the live `kernelbot` test result, not `fp8-mm/task.yml`.
- `fp8-mm/task.yml` is useful for shape envelopes only. Its old FP8/MI300 wording is stale for the current MI355X MXFP4 problem.

## Current reliable evidence

- Historical pass artifact exists at:
  `/Users/v/reference-kernels/problems/amd/.agent-loop/problems/mxfp4_mm/candidates/c56246533e6b/evaluation/result.txt`
- Current repo baseline now fails live `test` with the same `mismatch:13996` signature as the best HIP branches:
  `/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260312-093728-current-repo-baseline-test/stages/01_test/result.txt`

## Public AITER clues

Read these files when reconstructing the contract:

- `/tmp/aiter-inspect.vhxmO2/op_tests/test_gemm_a4w4.py`
- `/tmp/aiter-inspect.vhxmO2/aiter/ops/shuffle.py`
- `/tmp/aiter-inspect.vhxmO2/aiter/ops/quant.py`

Important clue from `test_gemm_a4w4.py`:

- Public AITER reference builds `run_torch(...)` from quantized packed tensors plus raw unshuffled scales.
- It then compares that reference against `aiter.gemm_a4w4(...)` using shuffled scales and `shuffle_weight(...)`.
- This is the closest public hint to the live judge, but it is not a guarantee that the competition input tensors match those exact intermediate tensors.

## Proven probe facts

From `/Users/v/reference-kernels/problems/amd/.agent-loop/problems/mxfp4_mm/manual/b_shuffle_probe/result.txt`:

- `b_shuffle == shuffle_weight(b_q_input, layout=(16, 16))`
- `b_scale_sh == e8m0_shuffle(raw_scale_from_quant(b, shuffle=False))`
- `b_q_input != quant(b, shuffle=False)[0]`
- `b_q_input != quant(b, shuffle=True)[0]`

From `/Users/v/reference-kernels/problems/amd/.agent-loop/problems/mxfp4_mm/manual/oracle_grid_probe/result.txt`:

- Internal probe best branch on the first case was `A_sf__B_in_raw`
- That means:
  - `A`: quantized/dequantized with `shuffle=False`
  - `B`: input `b_q` with raw unshuffled scale from `quant(b, shuffle=False)`

## Best current live branches

Best live mismatch plateau is still `13996` on the first case.

Branches tied at that level:

- `/Users/v/reference-kernels/problems/amd/.agent-loop/problems/mxfp4_mm/manual/hip_reference_tiled_retry_f32_oracle/submission.py`
- `/Users/v/reference-kernels/problems/amd/.agent-loop/problems/mxfp4_mm/manual/hip_reference_aqsh_arawscale_bq_rawscale/submission.py`

Clearly worse branches:

- `hip_reference_tiled_retry_f32_oracle_a_shuffle_true`: `16844`
- `hip_reference_tiled_a_false_b_provided`: `16500`
- `hip_reference_tiled_a_true_b_provided`: `16846`
- `hip_reference_raw_ab_f32`: `16280`
- `hip_reference_raw_a_bq_rawscale`: `15921`
- `hip_reference_raw_a_bq_providedscale`: `16492`
- `hip_reference_aqsh_arawscale_bq_providedscale`: `16500`

## Open problem

The unresolved part is the exact live meaning of `b_q` and the exact reference the judge currently uses.

Current evidence suggests:

- simple raw-vs-shuffled toggles are mostly exhausted
- the remaining gap is likely the exact packed-`B` reference path, not the tiled HIP kernel body itself
