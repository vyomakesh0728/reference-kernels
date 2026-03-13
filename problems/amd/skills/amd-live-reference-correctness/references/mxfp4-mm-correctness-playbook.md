# mxfp4_mm Correctness Playbook

## Goal

Get one HIP `submission.py` to pass live `test` on `amd-mxfp4-mm`.

Do not optimize performance until that happens.

## Default loop

1. Summarize current harness results.
   Run:
   `python3 /Users/v/reference-kernels/problems/amd/skills/amd-live-reference-correctness/scripts/summarize_mxfp4_mm_harness.py --repo /Users/v/reference-kernels/problems/amd`

2. Choose one semantic axis only.
   Preferred axes:
   - `A` packed representation
   - `A` scale source
   - `B_q` interpretation
   - `B` scale source
   - shuffle/layout reconstruction

3. Clone the best current HIP branch into a new manual experiment directory.

4. Run `test` only:
   `python3 -m agent_loop harness-run --problem mxfp4_mm --source <submission.py> --family hip_explore --label <label> --stages test`

5. Compare mismatch counts.
   Keep the branch only if it improves the first-case mismatch or clearly improves the overall mismatch pattern.

6. If the branch ties the best mismatch, keep it only if it is closer to the public AITER construction or gives a clearer next probe.

## Diagnostic probes

Use diagnostic probes when the next step is unclear.

Good probe pattern:

- print contract facts in `stdout`
- return a stable reference path so the run completes
- do not confuse the probe with a real final candidate

Current useful probe files:

- `/Users/v/reference-kernels/problems/amd/.agent-loop/problems/mxfp4_mm/manual/b_shuffle_probe/submission.py`
- `/Users/v/reference-kernels/problems/amd/.agent-loop/problems/mxfp4_mm/manual/oracle_grid_probe/submission.py`

## Guardrails

- Do not trust `fp8-mm/task.yml` semantics as the live reference.
- Do not submit pure `torch.mm` as the real candidate.
- Do not use `aiter.gemm_a4w4` in real HIP candidates.
- Do not change multiple contract axes in one run.
- Do not move to `benchmark` or `leaderboard` before `status=ok` in `test`.

## If stuck at a mismatch plateau

If multiple branches tie the same mismatch count:

1. stop broad semantic guessing
2. read the public AITER files again
3. compare the live competition tuple against the public AITER intermediate tensors
4. write a narrower probe around the unresolved tensor meaning

This is better than rewriting the HIP kernel body again without a stronger contract hypothesis.
