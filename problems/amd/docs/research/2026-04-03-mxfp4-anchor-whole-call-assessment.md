# mxfp4_mm whole-call assessment from 13.406 us anchor

Anchor verification
- Live `fp8-mm/submission.py` matches `fp8-mm/submission_anchor_13p406.py` (`git diff --no-index` returned empty diff).
- Current best benchmark artifact: `.agent-loop/harness_runs/mxfp4_mm/20260402-120426-p0p5-runtime-collapse-owned-scaleflat-r1-benchmark/stages/01_benchmark/parsed_metrics.json`
- Geomean: `13.406361750228412 us`

Scored benchmark entries
- m4 k512 n2880: `9.9 us`
- m16 k7168 n2112: `19.6 us`
- m32 k512 n4096: `10.1 us`
- m32 k512 n2880: `9.98 us`
- m64 k2048 n7168: `18.1 us`
- m256 k1536 n3072: `16.4 us`

Wall vs GPU-self snapshot
Using `.agent-loop/harness_runs/mxfp4_mm/20260402-143032-baseline-recheck-profile-r1-harness/stages/01_profile_rocprof/profile/profile_summary.json`:

| shape | wall_us | gpu_self_us | a_pack_us | kernel_us | wall-self gap_us |
|---|---:|---:|---:|---:|---:|
| m4 | 9.900 | 2.918 | 2.079 | 0.839 | 6.982 |
| m16 | 19.600 | 2.598 | 2.119 | 0.479 | 17.002 |
| m32_n4096 | 10.100 | 2.078 | 1.599 | 0.479 | 8.022 |
| m32_n2880 | 9.980 | 2.438 | 1.959 | 0.479 | 7.542 |
| m64 | 18.100 | 0.958 | 0.479 | 0.479 | 17.142 |
| m256 | 16.400 | 0.958 | 0.479 | 0.479 | 15.442 |

Observation
- The 6-entry scored geomean is `13.406 us`, but the geomean of the profile GPU-self numbers is only about `1.81 us`.
- Therefore the dominant remaining loss is still whole-call/runtime law, not MFMA body math.

Live exact-path audit
In the current live file, every hot exact public path still does the same structural sequence:
1. Python allocates one owned workspace buffer.
2. C++ wrapper carves `c`, `a_packed`, and `a_scale` out of that workspace.
3. Wrapper launches `launch_mxfp4_pack_a_fixed_raw(...)`.
4. Wrapper launches the exact MFMA kernel.

Verified in:
- `mxfp4_mm_hip_mfma_scale_exact_m4_direct_entry_public_k512_owned_workspace`
- `mxfp4_mm_hip_mfma_scale_exact_m16_direct_entry_public_k7168_n2112_owned_workspace`
- `mxfp4_mm_hip_mfma_scale_exact_m32_direct_entry_public_k512_owned_workspace`
- `mxfp4_mm_hip_mfma_scale_exact_m64_direct_entry_public_k2048_n7168_owned_workspace`
- `mxfp4_mm_hip_mfma_scale_exact_m256_direct_entry_public_k1536_n3072_owned_workspace`

Implication
- Current public exact paths already deleted most old B-side temp laws.
- The common surviving tax is the standalone shared A-pack launch family plus per-call runtime/launch overhead around it.
- More generic wrapper cleanup by itself is unlikely to halve geomean again.
- Deleting the two-launch law is still the highest-confidence path to <=7 us.

Important dormant contract fact
- `b_shuffle` is still in the live tuple contract but, on the hot exact public path, is only asserted for row-count consistency and otherwise unused.
- This is a possible later structural lever, but it is not the first spend while the shared A-pack launch remains live.

Conclusion
- Yes: keep pushing whole-call.
- No: do not interpret that as “more small wrapper polish”.
- The next spend should be targeted at removing the remaining standalone A-pack helper launch from exact public paths, shape by shape, without reopening duplicate A-pack ownership or serial tile sweeps.

Ranked next branches
1. `m32_singlelaunch_bpreshuffle_nodup_apack_k512`
   - Best first portfolio spend because m32 is scored twice.
2. `m16_singlelaunch_bpreshuffle_nodup_apack_k7168_n2112`
   - Largest single visible latency bucket.
3. `m64_singlelaunch_bpreshuffle_nodup_apack_k2048_n7168`
4. `m256_singlelaunch_bpreshuffle_nodup_apack_k1536_n3072`
5. `m4_singlelaunch_bpreshuffle_nodup_apack_k512`
6. `wide_post_delete_body_retune_32x32x64`
   - Only after one structural deletion wins.

Guardrails
- Do not reopen AITER lanes.
- Do not reopen per-CTA fused-A or serial sweep ownership laws.
- Do not spend on MFMA-family churn before deleting the two-launch law.
- Treat Python/HIP graph replay as dead on this runner unless proven otherwise.

Useful envelope
- The existing frontier-analysis reference envelope (`m4 4.9`, `m16 9.6`, `m32_n2880 4.5`, `m32_n4096 4.6`, `m64 10.1`, `m256 9.4`) models to about `6.724 us` on the 6-entry scored portfolio.
- That means <=7 us is still consistent with a whole-call-first thesis, but only if the next branches actually delete launch/runtime structure rather than shaving helper internals.
