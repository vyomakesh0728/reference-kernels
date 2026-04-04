# m32_singlelaunch_rawb_nodup_apack_k512_r1 negative result

Summary
- Implemented a shape-scoped exact-public m32 one-launch approximation in `fp8-mm/submission.py` that removed the owned-workspace + standalone A-pack helper launch from the public m32 path.
- Implementation used the live raw `b_q` + shuffled `b_scale_sh` contract and folded A quantization into the exact m32 kernel body.
- Local `py_compile` passed.
- Remote `test` passed.
- Remote `benchmark` failed the keep gate catastrophically.

Remote runs
- Test run dir: `.agent-loop/harness_runs/mxfp4_mm/20260403-150303-m32-singlelaunch-rawb-nodup-apack-k512-r1-test`
- Test workflow: `https://github.com/gpu-mode/kernelbot/actions/runs/23950828766`
- Benchmark run dir: `.agent-loop/harness_runs/mxfp4_mm/20260403-150701-m32-singlelaunch-rawb-nodup-apack-k512-r1-benchmark`
- Benchmark workflow: `https://github.com/gpu-mode/kernelbot/actions/runs/23950915457`

Benchmark result
- Geomean: `33.1455 us`
- Per-shape:
  - `m4_n2880_k512`: `10.1 us`
  - `m16_n2112_k7168`: `20.1 us`
  - `m32_n2880_k512`: `147.0 us`
  - `m32_n4096_k512`: `147.0 us`
  - `m64_n7168_k2048`: `18.1 us`
  - `m256_n3072_k1536`: `16.7 us`

Interpretation
- This branch deleted the helper launch but reintroduced the dead ownership law in another form: raw A was re-quantized independently inside every output-column CTA.
- Keeping full `N32` CTA parallelism did not save the lane.
- The duplication law dominated badly enough that the branch regressed far above the 13.406 us anchor.
- This is distinct from the earlier dead single-CTA serial sweep lane, but reaches the same strategic conclusion: deleting a launch without preserving one-pack-per-call economics is not viable.

Action taken
- Reverted `fp8-mm/submission.py` back to `fp8-mm/submission_anchor_13p406.py` immediately after the benchmark miss.

Implication for next work
- Treat exact-public raw-A fused m32 lanes without a real cross-CTA reuse law as dead.
- If the m32 whole-call thesis is revisited, it should be through a true `b_shuffle` consumer with a non-duplicating A ownership law, or a different structural deletion entirely.
