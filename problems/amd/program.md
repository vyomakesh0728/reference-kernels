# mxfp4-mm Closed-Loop Optimization Program

Goal
- Drive the overall benchmark geomean to <= 7.0 us across the hot exact-shape family: m4, m16, m32, m64, m256.
- Optimize whole-call latency, not just kernel body time.

Scope
- This program is specific to mxfp4_mm on MI355X (gfx950).
- It replaces generic “training experiment” workflows with a kernel-optimization ladder and gating rules.

Non‑negotiable invariants
- Keep the live tuple contract: (a, b, b_q, b_shuffle, b_scale_sh).
- Preserve the current winning raw-contract path unless a whole cost center is deleted.
- One exact shape per branch (do not touch multiple exact shapes in one change).
- No A-pack reopen unless duplication-law proof is satisfied (paper-first).
- No fixed-CTA sweep across all N for thin shapes. No grid-cooperative launch. No full-grid barriers.
- Do not spend remote quota from subagents.

Success criteria
- Geomean improves by >= 0.2 us.
- No visible shape regresses > 0.1 us.
- Correctness passes on test + benchmark.

Stop condition
- Geomean <= 7.0 us and all shapes pass correctness validation.


Required artifacts (Candidate Card)
Every branch must start with the Candidate Card before any code is written:
- shape
- deleted_cost_center
- expected_upside_source
- why_larger_than_noise
- touched_symbols_or_regions
- forbidden_edits
- success_gate

Optional (but recommended if relevant): regime_tag, hypothesis, expected_gain, next_patch.

Reject the branch if any field is missing.


Global veto rules
Reject the branch immediately if:
- It sounds like cleanup/hoist/prep polish without a named deleted bucket.
- It changes more than one exact shape.
- It changes prep and scheduling together.
- It reuses a generic helper with only minor local polish.
- It reopens prep-only m16 or prep-only m32.


Allowed cost centers (whole-call deletions only)
- Python-side wrapper materialization and inflight retention.
- Generic mxfp4_pack_a_fixed on an exact-shape path.
- Generic B-pack or B-scale materialization reused across exact wide shapes.
- Per-iteration pointer arithmetic and scale-block recomputation in an exact kernel body.
- Dead exact-path epilogue, masking, or store work.

If a branch cannot map to one of these buckets, reject it.


Phase 0: Baseline anchor and profiling
- Identify the current benchmark winner (best measured trunk).
- profile_rocprof is allowed only for the current benchmark winner.
- Use profile_summary.json and candidate_cards.json to pick the next deletion bucket.
- Zip-derived Candidate Cards beat intuition.


Phase 1: Portfolio-first deletion ladder (order matters)
Ladder 1: compiled direct-entry everywhere
- Eliminate Python-owned exact-path shaping for hot shapes.
- Success: all hot exact shapes use compiled direct-entry when it exists.

Ladder 2: kill helper launches before body polish
- Delete standalone helpers only if the work does not reappear as duplicate hot-path work.
- Priority: remove m64 row-major b_scale repair if address math cost stays bounded.
- Hard rule: no new A-pack local feeder rewrites.

Ladder 3: remove temp laws, not just temp tensors
- Target write-then-read temp traffic created only to bridge helper kernels.
- Priority: m64 row-major b_scale temp, then other non-A-pack temps.
- A-pack only if duplication law changes (otherwise veto).

Ladder 4: constant-body public shapes
- Remove generic branches/tails/address shaping for public fixed shapes.
- Only if it deletes setup/addressing work with a portfolio story.

Ladder 5: runtime/orchestration collapse
- Reduce host/runtime overhead between dispatch, helper setup, temp allocation, and launch.

Closed lanes
- Local A-pack feeder rewrites: v121, v122, v125.
- Bounded C2/C4 exact-wide A-pack service.
- Macro-cluster A-pack designs without a legal handoff mechanism.
- m64 body-only unrolled specialization without a deleted bucket (v124).


Phase 2: A-pack speedups without duplication-law changes (legal)
Goal: make mxfp4_pack_a_fixed materially faster without changing ownership law.
Allowed lanes:
- CDNA4 FP4 scale-pack builtins (gfx950) with correctness fix.
- DME/raw_buffer_load_lds path to lower VGPR pressure and overlap loads.

Rules:
- Keep output semantics identical to the existing hand-rolled quantizer.
- Add a feature flag to switch between builtin path and hand-rolled path.
- Validate packed output against current path before remote spend.


Phase 3: Body-side MFMA shape switches
Goal: reduce instruction count and scale-load overhead in the body.
Allowed lane:
- Evaluate 32x32x64 MFMA scale bodies for m32/m64 vs the current 16x16x128.

Rules:
- Keep the live tuple contract and scale packing intact.
- Gate usage by exact shape; keep the old kernel available as fallback.


Phase 4: A-pack duplication-law research (paper-only)
This phase is research-only until a legal cross-CTA handoff mechanism exists.

A-pack reopen gate (all shapes)
Must state:
- reuse_factor
- duplication_factor
- saved_global_bytes_per_block
- why_total_quant_work_drops
Reject if:
- duplication_factor > reuse_factor
- per-CTA re-quant with no cross-CTA reuse law
- only local CTA reuse while duplication scales with output-column CTA count
- win claim is only “fewer launches” or “fewer temp bytes” without total-work reduction

Thin-family A-pack remote-spend gate (m4/m16)
Must also state:
- quant_dup_upper_bound
- parallelism_floor_ratio vs v101 thin baseline
- n_bundle_per_owner
- proof that quant work does not scale with full output-column CTA count
Reject if:
- parallelism_floor_ratio < 0.50 on m4 or m16
- fixed CTA sweep across full N
- grid-cooperative launch / full-grid barrier
- one producer serializes a full-call N traversal

Wide-family A-pack remote-spend gate (m32/m64/m256)
Must also state:
- reuse_factor_per_quant
- quant_dup_upper_bound
- saved_global_bytes_per_block
- new_internal_quant_scope
Reject if:
- per-CTA re-quant
- quant_dup_upper_bound > reuse_factor_per_quant
- changes both A feed and CTA ownership in one step
- no proof that total quant work drops

Macro-cluster arithmetic (paper-only)
- If one producer spans S consecutive N32 tiles: reuse=S, duplication=ceil(num_n_tiles/S).
- Minimum spans to clear first gate (reuse >= duplication):
  m32 n=4096: S>=12
  m32 n=2880: S>=12
  m64 n=7168: S>=16
  m256 n=3072: S>=12
- Stronger practical gate (reuse >= 2*dup):
  m32 n=4096: S>=16
  m32 n=2880: S>=16
  m64 n=7168: S>=24
  m256 n=3072: S>=16
- m16 public shape n=2112 (N tile=16): num_n_tiles=132
  reuse=S, duplication=ceil(132/S)
  first gate: S>=12, strong gate: S>=17
  parallelism floor ~ 1/S (fails thin gate if S>=12)

Operational conclusion:
- No A-pack reopen without a legal cluster-scoped handoff mechanism.
- DME is local only; it does not enable cross-CTA A reuse.


Phase 5: Run and promotion policy
Order is mandatory:
1) preflight (static-only ok)
2) test
3) benchmark
4) profile_rocprof (only for current benchmark winner)

Commands (example)
- preflight:
  python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop preflight --variant <variant> --source <path> --lane <lane> --hypothesis <text> --expected-gain <text> --next-patch <text> --runtime none
- test:
  python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop submit --variant <variant> --source <path> --lane <lane> --hypothesis <text> --expected-gain <text> --next-patch <text> --stage test
- benchmark:
  python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop submit --variant <variant> --source <path> --lane <lane> --hypothesis <text> --expected-gain <text> --next-patch <text> --stage benchmark
- profile_rocprof:
  python3 -m agent_loop --config agent_loop.toml mxfp4-closed-loop submit --variant <variant> --source <path> --lane <lane> --hypothesis <text> --expected-gain <text> --next-patch <text> --stage profile_rocprof

Promotion rules
- Rerun if within 1% of prior benchmark.
- Leaderboard only after two agreeing benchmark wins within 0.75%.
- If a test submit fails to trigger, immediately retry with --continue-after-fail.

Reporting template
- Preflight status and run dir
- Test status, run dir, workflow URL
- Benchmark geomean and per-shape times
- Any regressions > 0.1 us
- Next recommended deletion bucket


Phase 6: Documentation and negative results
- Record negative results (failed correctness, perf regressions, or instability) to avoid repeating closed lanes.
- Update any gating rule if evidence forces it.

End of program.
