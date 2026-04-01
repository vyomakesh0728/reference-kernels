# mxfp4-mm MI355X ISA Research Note

Scope: `mxfp4_mm` only. This is durable team context, not a run log.

## Source Of Truth

- The last archived standalone benchmark folder under `fp8-mm/` is still [v135 benchmark](/Users/v/reference-kernels/problems/amd/fp8-mm/20260329-125150-v135-benchmark/stages/01_benchmark/parsed_metrics.json) at `22.592 us`, but that is no longer the best kept trunk.
- The prior kept low-water mark in this repo was [native_scaled_exact_shape_m16_scaleaddr_t7 benchmark](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260330-130300-native-scaled-exact-shape-m16-scaleaddr-t7-benchmark/stages/01_benchmark/parsed_metrics.json) at `21.2289 us`.
- The current frontier is now [wavepack_directentry_t18 benchmark](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-103242-wavepack-directentry-t18-benchmark/stages/01_benchmark/parsed_metrics.json) at `14.5097 us`.
- Nearby follow-ups are informative:
  - [native_scaled_exact_shape_m64_scaleaddr_t11 benchmark](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260331-072846-native-scaled-exact-shape-m64-scaleaddr-t11-benchmark/stages/01_benchmark/parsed_metrics.json): `21.2875 us` geomean, basically a tie.
  - [native_scaled_exact_shape_m16_builtinpack_t12 benchmark](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260331-103149-native-scaled-exact-shape-m16-builtinpack-t12-benchmark/stages/01_benchmark/parsed_metrics.json): `21.8653 us`; `m16` improves to `25.9 us` but the portfolio regresses.
  - [native_scaled_exact_shape_m16_dmeapack_t14 benchmark](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260331-154757-native-scaled-exact-shape-m16-dmeapack-t14-benchmark/stages/01_benchmark/parsed_metrics.json): `25.8989 us`, strong negative.
  - [native_scaled_exact_shape_fusedapack_t16 benchmark](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-044716-native-scaled-exact-shape-fusedapack-t16-benchmark/stages/01_benchmark/parsed_metrics.json): `25.8180 us`, strong negative.
  - [m64_shscale_t19 benchmark](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-104830-m64-shscale-t19-benchmark/stages/01_benchmark/parsed_metrics.json): `14.6050 us`, a slight regression from `t18`; deleting exact `m64` `b_scale_decode` in the obvious way paid back too much of the win as in-kernel address law.
- `t18` also materially cuts every hot exact shape instead of trading winners and losers:
  - `m4_n2880_k512`: `16.9 -> 10.0 us`
  - `m32_n2880_k512`: `17.6 -> 10.1 us`
  - `m32_n4096_k512`: `17.6 -> 10.1 us`
  - `m16_n2112_k7168`: `25.9 -> 20.0 us`
  - `m64_n7168_k2048`: `30.3 -> 23.1 us`
  - `m256_n3072_k1536`: `26.6 -> 19.8 us`
- The gap from the current frontier to the stated `~7 us` goal is now about `2.07x`, so the target is still aggressive but no longer fantasy; the remaining wins still need whole-call law changes plus further shared helper or shape-local deletion.

## Repo Facts That Matter

- The kept March 30 wins were not MFMA-family changes. They were fixed-cost and scale-address deletions:
  - [agent-log.md](/Users/v/reference-kernels/problems/amd/agent-log.md)
  - `native_scaled_exact_shape_m4_fixedcost_t6`: `22.5921 -> 22.3918 us`
  - `native_scaled_exact_shape_m16_scaleaddr_t7`: `22.3918 -> 21.2289 us`
- The post-`t7` profile is the right current cost split:
  - [t7 profile summary](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260330-182516-native-scaled-exact-shape-m16-scaleaddr-t7-profile-rocprof/stages/01_profile_rocprof/profile/profile_summary.json)
  - `m16`: `a_pack_share = 0.500`, `kernel_share = 0.500`
  - `m4`: `a_pack_share = 0.704`, `kernel_share = 0.296`
  - `m32`: `a_pack_share = 0.715`, `kernel_share = 0.285`
  - `m64`: roughly `1/3 a_pack`, `1/3 b_scale_decode`, `1/3 kernel`
- The first real shared-helper rewrite is now validated:
  - `t18` keeps a wave-cooperative `mxfp4_pack_a_fixed`, preserves the exact-kernel MFMA bodies, and collapses exact `m64` back onto compiled direct entry.
  - The first `t17` cut compiled but failed correctness because the helper used divergent-lane byte gathers. Promoting the byte gathers to all lanes and predicating only the final stores made the branch test-green and benchmark-winning.
- The new active profile prior is now the `t18` zip-derived run:
  - [t18 profile summary](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-103934-wavepack-directentry-t18-profile-rocprof/stages/01_profile_rocprof/profile/profile_summary.json)
  - [t18 candidate cards](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-103934-wavepack-directentry-t18-profile-rocprof/stages/01_profile_rocprof/profile/candidate_cards.json)
  - `m4`: `a_pack_share = 0.722`
  - `m16`: `a_pack_share = 0.500`, `kernel_share = 0.500`
  - `m32`: `a_pack_share = 0.500`, `kernel_share = 0.500`
  - `m64`: `a_pack_share = 0.390`, `b_scale_decode_share = 0.455`, `kernel_share = 0.156`
  - `m256`: `a_pack_share = 0.790`, `kernel_share = 0.210`
- The active exact paths still all call the same `mxfp4_pack_a_fixed` helper before their exact kernels:
  - baseline scalar helper: [fp8-mm/submission.py#L414](/Users/v/reference-kernels/problems/amd/fp8-mm/submission.py#L414)
  - current m16 direct entry: [fp8-mm/submission.py#L1141](/Users/v/reference-kernels/problems/amd/fp8-mm/submission.py#L1141)
  - current m32 direct entry: [fp8-mm/submission.py#L1811](/Users/v/reference-kernels/problems/amd/fp8-mm/submission.py#L1811)
- That means helper-side A-pack speedups are still portfolio-wide leverage even though helper-side A-pack ownership rewrites are closed.

## What Is Already Closed

- Thin and exact-wide `A-pack` launch annihilation via duplicate local re-quantization is closed by repo evidence:
  - `v102`, `v103`, `v104`, `v105`, `v106`, `v107`, `v109`, `v121`, `v122`, `v125`
  - see [program.md](/Users/v/reference-kernels/problems/amd/program.md), [principle optimizer Q&A](/Users/v/reference-kernels/problems/amd/fp8-mm/principle_kernel_optimizer_qa_mm.md), and [mxfp4 exact-shape frontier](/Users/v/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/mxfp4-exact-shape-frontier.md)
- Simple DME-to-LDS staging without reuse is not a win for the helper:
  - `t14` is a clean negative.
- Fused-A compute-path rewrites that move quantization into the exact kernels without a new reuse law are also negative:
  - `t16` and the earlier `v102/v121` line.
- Wide raw shuffled-scale is real for `m32` and `m256`, but `m64` still overpays address math:
  - `v114`, `v115`, `v116`, `v118`, `t11`

## Hardware / ISA Facts Worth Keeping In Cache

Primary local references:

- [CDNA4 ISA guide](/Users/v/reference-kernels/problems/amd/important_papers/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [HipKittens: Fast and Furious AMD Kernels](/Users/v/reference-kernels/problems/amd/important_papers/HipKittens%20Fast%20and%20Furious%20AMD%20Kernels.pdf)
- [optimization.md](/Users/v/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/optimization.md)
- [amd-blog-insights.md](/Users/v/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/amd-blog-insights.md)

Key facts:

1. `V_MFMA_SCALE_F32_16X16X128_F8F6F4` and `V_MFMA_SCALE_F32_32X32X64_F8F6F4` are the correct scaled-FP4 matrix families here.
   - The ISA says the scale payload is fused into the MFMA-scale instruction and the block scale size is `32` in `K`.
   - For `16x16x128`, each row needs `4` scale bytes; across `16` rows that is `64` 8-bit scales, exactly one quarter of a VGPR across `64` lanes.
   - Takeaway: keep these MFMA families; optimize scale derivation/load frequency around them.

2. `DS_PERMUTE_B32` and `DS_BPERMUTE_B32` are true lane-routing instructions, not real LDS writes.
   - The ISA explicitly says they do not actually write LDS memory.
   - Takeaway: if a helper or exact kernel needs lane exchange, prefer these or related permlane paths over scalar byte traffic or real LDS staging.

3. CDNA4 supports direct global-memory-to-LDS vector loads and asynchronous LDS-to-register reads.
   - HipKittens calls out `buffer_load_dword[x1/x3/x4]` as direct HBM->LDS loads that skip the register file and accept constant offsets.
   - HipKittens also calls out `ds_read_b128` as a 16-byte LDS->register load gated by `lgkmcnt`.
   - Takeaway: HBM->LDS is only attractive when multiple lanes or multiple later instructions reuse the staged bytes. It is not automatically attractive for a no-reuse helper.

4. Wave specialization is usually the wrong first move on AMD MI355X.
   - HipKittens shows producer/consumer scheduling loses on AMD because static register partitioning burns registers on producer waves that do not contribute output math.
   - Takeaway: avoid resurrecting `A-pack` service designs that spend extra waves just moving data.

5. `sched_barrier`, `sched_group_barrier`, and `s_setprio` are real late-stage tools.
   - HipKittens uses them to order VMEM, MFMA, and DS clusters.
   - Takeaway: these are plausible only after a kernel body is already tiny and the data law is stable. They are not a substitute for deleting bytes or helper launches.

## Current Helper-Side Blind Spot

The repo has explored "where A-pack happens" far more than "how the shared helper itself maps work onto a wave".

The live helper still looks like this:

- [fp8-mm/submission.py#L414](/Users/v/reference-kernels/problems/amd/fp8-mm/submission.py#L414)
  - one thread owns one `1x32` A block
  - scalar loop over `32` BF16 values to compute `amax`
  - scalar loop over `16` pairs to quantize
  - `16` separate byte stores for the packed payload
- The DME experiment in [fp8-mm/hip_phase2_working.py#L682](/Users/v/reference-kernels/problems/amd/fp8-mm/hip_phase2_working.py#L682) improves ingress width but still:
  - loads into VGPR
  - copies into LDS
  - waits with `vmcnt(0)`
  - rereads from LDS through scalar loops
  - still emits `16` scalar byte stores

Interpretation:

- `t14` strongly suggests that HBM->VGPR->LDS->scalar-loop is the wrong shape when there is no cross-thread reuse.
- The profitable unexplored lane is not another ownership rewrite. It is a helper microarchitecture rewrite that keeps one quantization per `1x32` block but maps that work onto a wave more efficiently.

## Instruction-Level Patterns That Still Look Worth Spending On

### 1. Wave-synchronous `A-pack` helper for one `1x32` block

This is the highest-value legal experiment still visible.

Shape:

- one wave owns one or two `1x32` A blocks
- each active lane loads one BF16 input
- wave reduction computes `amax`
- even lanes pack two FP4 nibbles into one byte
- one lane stores the scale byte
- packed bytes are written as `4 x dword` or `1 x b128`, not `16 x byte`

Why it fits the current gate:

- ownership law does not change
- quant duplication does not increase
- no cross-CTA reuse claim is required
- helper stays a standalone helper, so this is an `A-pack speedup without duplication-law change`, which `program.md` explicitly allows

Likely instruction building blocks:

- wave reduce or lane broadcast for `amax`
- `DS_BPERMUTE_B32` / permlane / readlane style exchange for pairing odd-lane values
- `V_CVT_SCALEF32_PK_FP4_BF16` on even lanes when the builtin semantics are acceptable
- `i32x4` or `b128` stores for the final `16` packed bytes

### 2. Vector-store the helper output even before changing wave ownership

This is the smallest helper-side diff with real upside.

Current state:

- the baseline helper writes `packed_row[i]` one byte at a time
- the DME helper still writes `packed_row[i]` one byte at a time

Experiment:

- accumulate `16` packed bytes into `4` `uint32_t`s or one `i32x4_t`
- issue aligned vector stores to `a_packed`
- keep the scale-byte write separate

Reasoning:

- this cuts the helper store instruction count sharply without touching ownership or exact-kernel math
- it should compose with both scalar ingress and wave-synchronous ingress

### 3. Register-only vector ingress for the helper, not LDS staging

If the helper stays one-thread-per-block or two-blocks-per-wave, prefer:

- raw `b128` / `dwordx4` ingress into VGPR or packed `bf16x2` fragments
- immediate reduction and pack from registers
- no LDS unless there is actual cross-lane reuse

Why:

- `t14` is evidence against LDS round-trip without reuse
- HipKittens' HBM->LDS advice is for tiles that later feed DS reads, not for one-and-done helper data

### 4. `m64` exact public scale-address law delete

The helper is not the whole portfolio. `m64` still has a very clean non-helper target:

- keep the winning raw `b_q` exact path
- keep exact `m32/m256` intact
- make the exact `m64` shuffled-scale or repaired-row-major address law cheaper

This is still the cleanest non-`A-pack` exact-kernel target after the thin wins.

### 5. `m32` public `k=512` constant-body follow-through

`v118` showed the public `m32` body can move when the fixed shape really deletes address/setup work.

Keep:

- raw `b_q + b_scale_sh` contract
- fixed `k=512` law

Push further only if:

- the branch deletes actual generic loop/address work
- the next move is not just wrapper polish

### 6. Late-stage schedule hints only after the body shrinks

If a future exact body becomes tiny enough, then test:

- `sched_group_barrier(VMEM -> MFMA -> DS -> MFMA)`
- `sched_barrier(0)` cluster cuts
- `s_setprio` around MFMA-heavy spans

But only after the helper and address-law work above. Right now these are second-order.

## Recommended Experiment Order

1. Legal portfolio-wide helper rewrite:
   - wave-synchronous `mxfp4_pack_a_fixed`
   - no ownership-law change
   - vector stores required

2. Smaller helper-only fallback:
   - vector-store the current helper output
   - optionally pair with register-only `b128` ingress

3. Exact `m64` address-law delete:
   - keep the current best wide routing
   - attack only the remaining scale-address overhead

4. Exact `m32` public constant-body rerun/follow-through:
   - only if it still deletes a real bucket after the helper work

5. Schedule-hint / issue-order experiments:
   - only when the body is already the bottleneck

## Bottom Line

The repo is no longer in a "find a different MFMA" phase. The most under-explored high-value lane is:

- keep the current ownership law,
- keep the current MFMA family,
- stop paying scalar helper costs for `A-pack`,
- and only use LDS/DME when the bytes are actually reused.

That is the cleanest remaining path that can still move multiple exact shapes at once without reopening the closed duplication-law failures.

## 2026-04-01 Addendum: `t20` + fresh ISA pass

### New measured point

- New branch: `native_scaled_exact_shape_m4_fixedcost_fastdispatch_t20`
- Test run:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-122700-native-scaled-exact-shape-m4-fixedcost-fastdispatch-t20-test](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-122700-native-scaled-exact-shape-m4-fixedcost-fastdispatch-t20-test)
- Benchmark run:
  [/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-122947-native-scaled-exact-shape-m4-fixedcost-fastdispatch-t20-benchmark/stages/01_benchmark/parsed_metrics.json](/Users/v/reference-kernels/problems/amd/.agent-loop/harness_runs/mxfp4_mm/20260401-122947-native-scaled-exact-shape-m4-fixedcost-fastdispatch-t20-benchmark/stages/01_benchmark/parsed_metrics.json)
- Geomean: `14.5092 us` vs `14.5097 us` on `t18` (about `0.004%` better; effectively noise-level)
- Visible shapes:
  - `m4 10.0 us`
  - `m16 19.7 us`
  - `m32 10.2 / 10.1 us`
  - `m64 23.1 us`
  - `m256 19.9 us`

Interpretation:

- exact-`m4` fixed-cost dispatch cleanup is legal/test-green and does not hurt portfolio behavior,
- but this deletion class is now near the noise floor and is not a path to `<=7 us` by itself.

### Fresh ISA/doc mining results worth carrying forward

High-confidence next ideas:

1. exact `m64` `b_scale` decode delete only with cheaper in-kernel address law (not the naive `t19` form)
2. helper micro-architecture cuts that reduce wave helper instruction count (`__shfl` gather replacement via lane-permute idioms, vectorized ingress/store discipline)
3. shape-local vector-load cleanup for exact `m4/m16` bodies where scalar byte loops remain

Speculative but potentially large:

- native CDNA4 FP4 conversion builtins in helper quantization path, but only if they *replace* existing quant work and preserve current correctness contract

Anti-pattern still confirmed:

- deleting a helper launch while paying it back as kernel address/control overhead is not a true delete (`t19` proof)

### Quant target reminder (`<=7 us`)

- Required overall reduction from `14.5097` is still about `2.07x`
- Uniform cut needed is about `51.8%` per shape
- `m64 + m16 + m256` now account for about `75%` of required log-gap

Operational implication:

- prioritize branches that materially move `m64/m16/m256` buckets,
- treat small fixed-cost Python dispatch cleanups as tie-breakers only.
