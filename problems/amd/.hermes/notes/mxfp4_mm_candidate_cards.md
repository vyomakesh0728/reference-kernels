# mxfp4_mm candidate cards (post-v135 baseline)

## Card: m4-fixed-cost-delete (non-A-pack)

shape:
- exact m4 public shape (m=4, n=2880, k=512; b_scale_sh cols=16) on direct-entry path

regime_tag:
- m4-fixed-cost-delete

deleted_cost_center:
- redundant wrapper validation/contiguity checks on the public k=512 path

expected_upside_source:
- remove constant overhead in the m4 direct-entry wrapper by trusting the already-contiguous, uint8-viewed inputs from custom_kernel; avoid redundant contiguity/type checks and duplicate view conversions for the public shape only

why_larger_than_noise:
- m4 baseline in v135 is 16.9 us; shaving 0.2–0.4 us (>=1%) from fixed per-call wrapper overhead is above noise on a tiny path

profile evidence (docs):
- mxfp4-profile-branch-queue.md lines 304–313: “m4-fixed-cost-delete” is the next legal lane
- mxfp4-exact-shape-frontier.md lines 30–34 and 114–118: public m4 latency and a_pack dominance context
- latest profile artifacts (not in repo):
  - /Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/profile_summary.json
  - /Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/candidate_cards.json

touched_symbols_or_regions (fp8-mm/submission.py):
- custom_kernel m4 dispatch: lines 3101–3117
- direct-entry wrapper (C++ inside HIP_SRC): lines 1415–1437
- optional: add a public-shape-only wrapper to bypass redundant contiguity/type checks

forbidden_edits:
- do not delete or alter A-pack (keep mxfp4_pack_a_fixed launch/contract intact)
- do not touch m8/m16/m32/m64/m256
- no new helper launches or scheduling/ownership changes
- keep raw b_q/b_scale_sh data contracts unchanged

success_gate:
- m4 public case <16.7 us (>=1% vs v135 baseline 16.9 us) with no regressions in other shapes and no geomean loss

allowed implementation direction:
- add a public-shape-only m4 direct-entry wrapper that bypasses redundant contiguity/shape checks
- ensure the wrapper only relies on inputs already made contiguous and uint8-viewed in custom_kernel

---

## Card: m64-address-last (shuffled-scale address cleanup)

shape:
- exact m64 public shape (m=64, n=7168, k=2048) on the current row-major scale materialization path (b_scale unshuffled from b_scale_sh)

regime_tag:
- m64-address-last

deleted_cost_center:
- per-element address math in m64 b_scale unshuffle (row_block/row_in_block/source_linear mapping) during b_prep

expected_upside_source:
- add an m64-specific unshuffle kernel with fixed rows/cols to reduce address math; keep compute kernel and raw b_q/b_scale_sh contracts intact

why_larger_than_noise:
- m64 baseline in v135 is 29.7 us; b_scale unshuffle runs every call across large scale grids. A specialized address path should shave microseconds without touching MFMA body.

profile evidence (docs):
- mxfp4-profile-branch-queue.md lines 314–321: “m64-address-last” is the next legal lane after m4
- mxfp4-exact-shape-frontier.md lines 118–120: m64 still splits time across a_pack/b_scale/kernel
- latest profile artifacts (not in repo):
  - /Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/profile_summary.json
  - /Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/candidate_cards.json

touched_symbols_or_regions (fp8-mm/submission.py):
- custom_kernel m64 path + b_prep: lines 2999–3028
- new m64-specific unshuffle kernel (to add) near existing mxfp4_unshuffle_b_scale_kernel lines 559–587
- m64 rawb kernel/wrapper: lines 2090–2187 (kept intact)

forbidden_edits:
- do not change m32/m256 paths or kernels
- do not alter A-pack or introduce new ownership/scheduling changes
- do not modify shared mxfp4_unshuffle_b_scale_kernel; add an m64-specific variant and call it only from the m64 path

success_gate:
- improve m64 timing by >=1% vs v135 m64 baseline (29.7 us) without geomean regression

allowed implementation direction:
- add m64-specific unshuffle kernel with constant rows/cols to reduce address math in b_prep
- keep compute kernel and raw b_q/b_scale_sh contracts intact
