# mxfp4_mm MI355X ~7us Frontier Plan (Aggressive, structural-first)

Status date: 2026-04-01
Baseline frontier: `14.5092 us` geomean (`t20`)
Primary objective: drive geomean to `~7 us` with high-impact architecture changes, not micro-polish.

## Hard framing

- Required reduction from 14.5092 -> 7.0 is ~`2.07x`.
- Uniform per-shape cut required is ~`51.8%`.
- `m64 + m16 + m256` account for ~`75%` of remaining log-gap.
- Therefore: no more quota focus on m4 micro-cleanups; m4 is guardrail unless it blocks compounding.

## Strategy policy (explicit)

1) Structural-first, correctness-second (within controlled bounds)
- We allow temporary correctness drift during large architectural jumps.
- Every aggressive branch must carry a correction phase using:
  - atol/rtol hit rates
  - MAE
  - max_abs
  - relative error histogram tail bins

2) One big bucket per branch
- One shape, one deleted cost center.
- No mixed prep+scheduling branches.

3) Quota efficiency
- No remote benchmark spend without a candidate card + kill criteria.
- Default per-branch budget: `1 test + 1 benchmark (+1 rerun only if near-noise)`.
- Freeze lane after 3 consecutive negatives until new profile evidence.

## Big-bet lanes (priority ordered)

### Lane A (P1): Shared A-pack engine rewrite (ownership law unchanged)
Deleted bucket:
- helper instruction cost in `mxfp4_pack_a_fixed_kernel_wave` (lane exchange + pack/store shape)

Structural changes:
- replace current wave helper with wave64 cooperative pack engine (vector ingress/stores, reduced shuffle traffic)
- preserve call contract so exact m16/m64/m256 direct-entry paths remain compatible

Expected impact (band):
- geomean: `-1.8 us` to `-3.8 us`
- strongest upside across m16/m256, moderate m64

Risk:
- nibble ordering and lane routing correctness

Correction plan:
- helper parity first (a_packed/a_scale contract)
- then full-shape output metrics (atol/rtol/MAE/max_abs/rel_hist)

Kill criteria:
- <0.8 us geomean gain OR (m16+m256 combined gain <10%)

---

### Lane B (P2): exact m64 address-law deletion (not naive t19)
Deleted bucket:
- m64 b_scale materialization + expensive in-kernel shuffled-scale address rebuild

Structural changes:
- specialized exact m64 shuffled-scale closed-form decode helper
- specialized m64 kernel variant for public shape (k2048,n7168)
- remove row-major b_scale temp path in m64 direct-entry wrapper

Expected impact (band):
- geomean: `-0.4 us` to `-1.1 us`
- m64 target: `23.1 -> 18.5..21.0 us`

Risk:
- address/remap math subtle bugs

Correction plan:
- deterministic scale-map equivalence tests
- end-to-end error metrics and histogram tail guardrails

Kill criteria:
- m64 not <=22.0 us or geomean regression >0.5%

---

### Lane C (P3): exact m16 constant-body architecture jump
Deleted bucket:
- m16 kernel-side fixed setup/address overhead in public shape path

Structural changes:
- shape-specialized m16 kernel body (`k=7168,n=2112`) with dual-output tile reuse and reduced hot-loop address law
- preserve A-pack contract first pass

Expected impact (band):
- geomean: `-0.3 us` to `-0.9 us`
- m16 target: `19.7 -> 15.5..17.8 us`

Risk:
- row mapping and writeback indexing

Correction plan:
- packed-input kernel A/B compare vs current m16 kernel
- full output metrics gates

Kill criteria:
- <1.5 us m16 gain on first benchmark

---

### Lane D (P4, speculative): native CDNA4 FP4 builtin path in helper
Deleted bucket:
- software quantize/adjust inner loop in A-pack helper

Structural changes:
- builtin conversion path that REPLACES (not duplicates) existing quant work
- fallback path retained

Expected impact (band):
- geomean: `-0.6 us` to `-1.8 us` if semantics align

Risk:
- known scale-convention mismatch class

Policy:
- local parity proof mandatory before any remote benchmark

## Milestone ladder

- M1: `<=12 us`
- M2: `<=10 us`
- M3: `<=8 us`
- M4: `<=7 us`

Working windows (high level):
- m16/m64/m256 must move first at every milestone
- m4/m32 stay as guardrails until M3, then become active movers for final convergence

Detailed checkpoint table and branch sequencing are in:
- `docs/plans/2026-04-01-mxfp4-7us-execution-roadmap.md`

## Multi-agent operating model

- Research agent:
  - produces candidate cards + vetoes + expected impact bands
- Implementation agent:
  - implements one approved card only (shape-local)
- Verification agent:
  - enforces preflight/test/benchmark/rerun/profile gates + milestone windows

Pipeline cadence:
- N+1 research while N implementation runs and N-1 is verified.

## Immediate next branch order

1) `m64-address-law-r1` (highest-confidence next swing)
2) `m16-orchestration/constbody-r1`
3) `m256-orchestration/constbody-r1`
4) `compound-r1` from winners only

No new m4 micro branch unless guardrail breach requires stabilization.
