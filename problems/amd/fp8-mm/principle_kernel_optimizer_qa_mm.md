# Principle Kernel Optimizer Q&A

## Baseline

- Locked frontier:
  - benchmark: [v101](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_v101/submission.py) at `25.2188 us`
  - leaderboard: [v101](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_v101/submission.py) at `26.2218 us`
- Active profile prior:
  - [profile_summary.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/profile_summary.json)
  - [candidate_cards.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-111357-compound-v101-profile/stages/01_profile_rocprof/profile/candidate_cards.json)
- Most important new negative:
  - [v102](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_m16_apack_fused_v102/submission.py) passed real MI355X `test` but benchmarked at `37.381 us`, with `m16` exploding to `401.0 us`
  - source of truth: [parsed_metrics.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260325-122923-native-scaled-exact-shape-m16-apack-fused-v102-benchmark/stages/01_benchmark/parsed_metrics.json)
- More recent negatives reinforce the same blocker shape:
  - [v103](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_m16_apack_persistent_v103/submission.py): `40.421 us`, `m16 647.0 us`
  - [v104](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_m16_apack_persistent_grid_v104/submission.py): `31.803 us`, `m16 156.0 us`
  - [v105](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_m16_apack_owner_v105/submission.py): `27.597 us`, `m16 63.1 us`
  - [v106](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_m16_apack_pc_v106/submission.py): `37.679 us`, `m16 426.0 us`
  - [v107](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_thin_aservice_v107/submission.py): `87.946 us`, `m4 352.0 us`, `m16 3400.0 us`

## Executive Answer

- We are still prep-dominated, not compute-dominated.
- The next real multiplier is not “switch instruction family now.”
- The next real multiplier path is:
  1. annihilate `A-pack` as a launch family
  2. do it without repeating the underlying quantization work inside every hot MFMA thread
  3. re-profile
  4. only then open the first true architecture-jump lane

## Q1. Are we done deleting prep/materialization?

- No.
- We cleared the first-wave obvious buckets:
  - `m64`: exact-wide `B-pack/repack` deleted
  - `m256`: exact-wide `B-pack/repack` deleted
  - `m32`: exact-wide `B-pack/repack` deleted
  - `m4`: tiny-path `b_scale_sh` / scale-decode deleted
  - `m16`: tiny-path `B-scale` materialization deleted
- The dominant remaining family is now `A-pack`.
- Current profile evidence:
  - `m16`: `a_pack_share ≈ 0.728`, `b_scale_decode_share = 0`, `kernel_share ≈ 0.272`
  - `m4`: `a_pack_share ≈ 0.714`, `b_scale_decode_share = 0`, `kernel_share ≈ 0.286`
  - `m32/m64/m256`: each is still about `1/3 A-pack`, `1/3 B-scale decode`, `1/3 kernel`

## Q2. What is the dominant blocker family right now?

- `A-pack` ownership and launch amortization.
- Not “one more cleanup.”
- Not “BMM now.”
- Not “wider scheduling now.”

Why this is high confidence:
- Thin shapes are explicitly `A-pack` dominated in the live `v101` profile.
- Wide shapes still carry the same compiled `A-pack` helper family.
- The exact-wide `B-pack` family has already been resolved positive where it was obvious.

## Q3. Why did `v102` fail so badly?

- `v102` deleted the separate `A-pack` launch in the wrong way.
- It moved full `A` quantization/packing directly into the exact `m16` MFMA kernel at:
  - inline helper: [v102 submission.py#L739](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_m16_apack_fused_v102/submission.py#L739)
  - hot kernel: [v102 submission.py#L1062](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_m16_apack_fused_v102/submission.py#L1062)
  - direct-entry launch: [v102 submission.py#L1242](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_exact_shape_m16_apack_fused_v102/submission.py#L1242)
- Structurally, that means the same `A` matrix got re-quantized once per `tile_col` CTA instead of once per call.
- For the benchmarked `m16, n=2112`, that means about `2112 / 16 = 132` column CTAs repeated the same quantization work.
- So `v102` did not annihilate the cost center. It deleted the launch and multiplied the underlying work.

Principle:
- “Delete the launch” is not enough.
- We must also preserve or improve amortization.
- A smaller specialized producer launch by itself is also not enough; `v105` shows that shrinking the standalone `A-pack` producer footprint without changing reuse/ownership still loses badly versus `v101`.
- A chunked exact `m16` producer/consumer kernel is also not enough in the tested shape; `v106` shows that simply moving one producer wave and four consumer waves into the same launch still fails catastrophically if the ownership model remains shape-local and synchronization-heavy.
- A family-wide on-chip thin service is also not enough if it collapses grid-level parallelism; `v107` shows that bounding quant duplication with only `4` persistent CTAs across the whole thin `N` sweep destroys performance even though it avoids repeated full-grid quantization.
- A grid-cooperative thin service is also not enough if it relies on whole-grid synchronization across the thin tile grid; `v108` cleared the host-side compile bugs, but its real MI355X rerun still destabilized the self-hosted runner badly enough that the workflow lost communication and artifacts, so that execution model is not safe to pursue in the current kernelbot setup.

## Q3b. What did `v107` prove that `v106` did not?

- `v107` passed correctness and removed the obvious compile/layout bugs from the first thin-family on-chip service, so its benchmark result is trustworthy as an ownership signal.
- The public thin benchmarks make the parallelism failure concrete:
  - `m4, n=2880`: the old exact `m4` path in `v101` launches about `2880 / 16 = 180` CTAs; `v107` capped the sweep at `4` CTAs, which is only about `11%` of the baseline CTA supply
  - `m16, n=2112`: the old exact `m16` path in `v101` launches about `2112 / 16 = 132` CTAs; `v107` again used `4` CTAs, which is only about `15%` of the baseline CTA supply
- So `v107` proves the next thin `A-pack` design must satisfy two constraints at once:
  - quant duplication must stay bounded
  - grid-level parallelism must stay close to the original tile-parallel regime

That is the real stricter gate now: no new `A-pack` branch unless it can quantify both numbers before remote spend.

## Q4. How far are we from the true architectural multiplier?

- Against `sub-7 us`, the gap is still large:
  - benchmark: `25.2188 -> <7 us` is about `3.60x`
  - leaderboard: `26.2218 -> <7 us` is about `3.75x`
- A useful inferred breakdown from the `v101` profile:
  - if we only delete the live thin `A-pack` launches on `m4/m16`, portfolio geomean is still only about `16.5 us`
  - if we delete `A-pack` across all exact shapes, portfolio geomean is still only about `12.6 us`
  - if all currently visible prep/materialization buckets disappeared and only current kernel bodies remained, the inferred floor is about `7.9 us`

Conclusion:
- Prep-family deletion alone is necessary but not sufficient.
- It is the runway, not the final multiplier.
- The final multiplier likely lives in body-side work after prep is mostly gone.

## Q5. What measurable gates separate “still prep-dominated” from “ready for a multiplier lane”?

- Still prep-dominated:
  - a single deletable prep bucket owns `>60%` of self CUDA
  - kernel share is `<35%`
  - `m4` and `m16` clearly meet this bar today
- Transition but not yet multiplier-ready:
  - buckets split roughly `1/3 / 1/3 / 1/3`
  - there is still a legal whole-bucket delete or the lane is explicitly closed
  - that is the current `m32/m64/m256` state
- Multiplier lane becomes legal only when:
  - no remaining deletable prep bucket is dominant
  - `kernel_share >= 0.50` on the target shape
  - no prep bucket remains above about `0.25`
  - a fresh real MI355X profile confirms that shift

For BMM / alternate small-shape instruction family:
- not legal now
- legal only after a true `A-pack` winner lands and the next profile says the body, not prep, is dominant

## Q6. What are the highest-confidence questions a principal kernel optimizer should ask now?

### Thin Shapes

- Can `m16` delete the separate exact `A-pack` launch while keeping the winning raw `b_scale_sh` path and without repeating `v102`’s per-thread quantization disaster?
- Can `m4` do the same while preserving the proven wave64 ownership model and avoiding the old helper-swap / broken-launch forms?
- Does the new ownership path amortize quantization once per call, row-block, or producer group, instead of once per output tile?
- Can `m8` get a true exact body before we ask it to participate in any multiplier lane?
- After true `A-pack` deletion, does `m4` or `m16` finally cross into kernel-dominant territory?

### Wide Shapes

- Are we targeting the active exact-wide path, or stale packed-B code that exact dispatch no longer uses?
- Should wide `A-pack` collapse be a true family move, or staged through `m32` as a proof branch if we want to preserve the one-shape gate?
- Is the next wide delete `A-pack` only or `B-scale` only? If the answer is “both,” are we violating the branch gate?
- Can we collapse wide `A-pack` while keeping the winning raw-`b_q` exact bodies intact?
- What fresh profile evidence would justify reopening `m64` or `m256` individually?

### Measurement / Promotion

- Why is ranked tax concentrated on `m32/m64/m256` while `m16` is flat-to-better?
- Is the next wide win a seed-robust data-movement fix rather than a benchmark-only micro-optimization?
- What is the cheapest falsifiable experiment that tells us whether wide `B-scale` decode must stay separate after `A-pack` collapse?

## Q7. What are the top high-confidence next hypotheses?

### Highest Value Overall

1. `m16` strict `A-pack` launch annihilation on top of `v101`, but not via full in-kernel per-thread quantization
2. `m4` strict `A-pack` launch annihilation with the same “delete the launch, preserve amortization” rule
3. wide-family `A-pack` collapse for `m32/m64/m256`
4. wide direct-from-shuffled `B-scale` consumption after wide `A-pack` collapse
5. constant-body clones once prep is no longer dominant

### Thin-Shape Interpretation

1. `m16` is the best next thin-shape bet
2. `m4` is second
3. `m8` should get a real exact body before any serious multiplier spend
4. only after `A-pack` is gone should we ask whether `m4` or `m16` deserve BMM / alternate instruction work

### Wide-Shape Interpretation

1. family-wide `A-pack` collapse is higher value than another isolated wide microbranch
2. wide `B-scale` direct-from-shuffled consumption is the next-best family materialization delete
3. wrapper/allocation collapse is lower-confidence by itself
4. architecture work like CTA-order/LDS remains closed until profile-backed evidence names it

## Q8. What ideas are explicitly premature right now?

- Another prep-only `m16` or prep-only `m32` branch
- Reopening `m64` or `m256` without a new profile-backed card
- Any `A-pack` annihilation that inlines full quantization into the hot MFMA threads
- Reusing the `m4` helper-swap form that already lost
- Broad thin rewrites that touch `m4/m8/m16` together
- Wide scheduling / LDS / CTA-order work before prep buckets are gone
- Leaderboard spends for branches whose benchmark edge is smaller than current ranked tax

## Q9. What is the likely sequence to get closest to `<7 us`?

1. Land `m16` true `A-pack` launch annihilation
2. Land `m4` true `A-pack` launch annihilation
3. Re-profile the new thin trunk
4. Take one wide-family `A-pack` collapse for `m32/m64/m256`
5. Re-profile immediately again
6. If wide prep is mostly gone, take the new profile-backed wide materialization/body delete
7. Only then open the real architecture-jump lane:
   - small-shape alternate instruction family / BMM-style lane
   - constant-shape body clones
   - register / launch-pressure redesign

## Bottom Line

- We still need huge breakthroughs.
- The next breakthroughs are not random.
- The disciplined path is:
  - kill prep launches first
  - keep amortization intact
  - keep parallelism intact at the same time
  - then re-open the compute architecture question
- `v102` gave the most important new lesson:
  - the multiplier path is **not** “do the same work inside the hot kernel”
  - it is “change ownership so the work is done less often, then let the kernel body become the limiter”
- `v107` adds the next hard lesson:
  - the multiplier path is also **not** “save the quant work by starving the grid”
  - the next legal `A-pack` candidate must preserve most of the original tile-parallel launch geometry while reducing repeated quant work
- `v108` adds the next systems lesson:
  - the multiplier path is also **not** “turn the whole thin tile grid into one cooperative group”
  - a candidate must clear work law, reuse law, parallelism floor, and execution-model safety before it deserves another remote slot
