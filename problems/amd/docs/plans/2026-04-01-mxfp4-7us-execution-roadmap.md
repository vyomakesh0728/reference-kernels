# MXFP4-MM Execution Roadmap to ~7 us (Multi-Agent)

## 0) Baseline anchor (starting point)

- Source benchmark: `t20`
  - `.agent-loop/harness_runs/mxfp4_mm/20260401-122947-native-scaled-exact-shape-m4-fixedcost-fastdispatch-t20-benchmark/stages/01_benchmark/parsed_metrics.json`
- Current geomean: `14.5092 us`
- Current visible per-shape means:
  - `m4`: `10.0 us`
  - `m16`: `19.7 us`
  - `m32`: `10.2 / 10.1 us`
  - `m64`: `23.1 us`
  - `m256`: `19.9 us`
- Active constraints (carry-over from program + gates):
  - One exact shape per branch.
  - One deleted cost center per branch.
  - No `A-pack` reopen without reuse-vs-duplication proof.
  - No prep+scheduling combo branches.

---

## 1) Milestone checkpoints and per-shape windows

Notes:
- Target windows are for `m64/m16/m256` (required movers).
- `m4/m32` are guardrails first, then become active movers by `8 us` and `7 us` milestones.
- Each milestone requires two agreeing benchmark wins (`<=0.75%` spread).

| Milestone | Geomean gate | m16 target window | m64 target window | m256 target window | m4 guardrail | m32 guardrail (both cases) |
|---|---:|---:|---:|---:|---:|---:|
| M1 | `<=12.0 us` | `13.5-14.5 us` | `14.0-15.5 us` | `13.5-15.0 us` | `<=10.3 us` (hard fail `>10.5`) | `<=10.5 us` (hard fail `>10.8`) |
| M2 | `<=10.0 us` | `9.8-10.8 us` | `10.0-11.2 us` | `9.8-11.0 us` | `<=9.6 us` (hard fail `>10.0`) | `<=9.8 us` (hard fail `>10.2`) |
| M3 | `<=8.0 us` | `7.0-7.8 us` | `7.2-8.2 us` | `7.0-8.0 us` | `<=8.4 us` (hard fail `>8.8`) | `<=8.5 us` (hard fail `>9.0`) |
| M4 | `<=7.0 us` | `5.9-6.8 us` | `6.0-7.0 us` | `5.9-6.9 us` | `<=7.3 us` (hard fail `>7.6`) | `<=7.4 us` (hard fail `>7.8`) |

Checkpoint checklist (applies at each milestone):
- [ ] Two benchmark wins at/under milestone geomean gate.
- [ ] `m16/m64/m256` all inside target windows.
- [ ] `m4/m32` satisfy guardrails.
- [ ] No correctness regressions in `test` stage.
- [ ] Last winner has a fresh `profile_rocprof` artifact.

---

## 2) Branch sequencing plan (milestone-by-milestone)

Conventions:
- Branch naming: `ms<12|10|8|7>/<shape>-<deleted_bucket>-r<round>`
- Promotion order: `preflight -> test -> benchmark -> (rerun if near noise) -> profile_rocprof`.
- Queue policy: 1 shape per branch, max 1 aggressive branch in-flight.

### M1: 14.5 -> 12 us (biggest absolute drops on m64/m16/m256)

Primary sequence:
- [ ] `ms12/m64-address-law-r1`
  - delete exact `m64` shuffled-scale address rebuild bucket (without reopening naive raw-scale regression path).
- [ ] `ms12/m16-orchestration-delete-r1`
  - delete exact `m16` wrapper/temp/runtime shaping bucket.
- [ ] `ms12/m256-orchestration-delete-r1`
  - delete exact `m256` wrapper/temp/runtime shaping bucket.
- [ ] `ms12/m64-constbody-k2048-r2`
  - public-shape constant-body delete of setup/address arithmetic.
- [ ] `ms12/m16-constbody-k7168-r2`
  - public-shape constant-body delete of setup/address arithmetic.
- [ ] `ms12/compound-r3`
  - compound only previously verified winners; no new hypothesis in compound branch.

Fallback if plateau after 3 negatives:
- [ ] open exactly one aggressive branch from template (Section 4), then close or land within fixed quota.

### M2: 12 -> 10 us (contract collapse + remaining fixed-cost deletes)

Primary sequence:
- [ ] `ms10/m256-k1536-address-delete-r1`
- [ ] `ms10/m64-rowmajor-bscale-law-delete-r1` (only if kernel-side address cost remains bounded)
- [ ] `ms10/m16-fastpath-dispatch-delete-r1`
- [ ] `ms10/m64-constbody-k2048-r2` (second-pass body law)
- [ ] `ms10/compound-r3`

Guardrail sequence (only if triggered):
- [ ] `ms10/m4-guardrail-stabilize-rg`
- [ ] `ms10/m32-guardrail-stabilize-rg`

### M3: 10 -> 8 us (all shapes begin converging)

Primary sequence:
- [ ] `ms8/m16-nonApack-fixedcost-delete-r1`
- [ ] `ms8/m64-nonApack-fixedcost-delete-r1`
- [ ] `ms8/m256-nonApack-fixedcost-delete-r1`
- [ ] `ms8/m4-fixedcost-delete-r1`
- [ ] `ms8/m32-fixedcost-delete-r1`
- [ ] `ms8/compound-r2`

Aggressive slot (optional, one only):
- [ ] `ms8/agg-shape-scoped-packengine-rx`
  - allowed only with explicit kill-switch + rollback plan.

### M4: 8 -> 7 us (final convergence + stability)

Primary sequence:
- [ ] `ms7/m16-final-bucket-r1`
- [ ] `ms7/m64-final-bucket-r1`
- [ ] `ms7/m256-final-bucket-r1`
- [ ] `ms7/m4-final-bucket-r1`
- [ ] `ms7/m32-final-bucket-r1`
- [ ] `ms7/compound-final-r2`

Stability/promotion sequence:
- [ ] rerun final twice (`<=0.75%` spread)
- [ ] profile final trunk
- [ ] leaderboard spend only after benchmark stability is proven

---

## 3) Multi-agent operating model (research / implementation / verification)

### Role contracts

Research agent:
- [ ] Reads latest frontier/profile queue/cost-center gate docs.
- [ ] Produces 1-3 candidate cards, with explicit vetoes.
- [ ] Stays one branch ahead of implementation.

Implementation agent:
- [ ] Implements exactly one approved card on one branch.
- [ ] Keeps diff minimal and shape-local.
- [ ] Adds kill-switch for aggressive branches.

Verification agent:
- [ ] Runs `preflight -> test -> benchmark` on every candidate.
- [ ] Enforces milestone windows + guardrails.
- [ ] Tags branch as `land`, `rerun`, or `discard`.

### Pipeline cadence (continuous 3-lane)

- [ ] T0: Research prepares cards for branch `N+1`.
- [ ] T0: Implementation codes branch `N`.
- [ ] T0: Verification validates branch `N-1`.
- [ ] T1 handoff artifacts required:
  - candidate card
  - patch/diff summary
  - benchmark delta + per-shape table
  - decision (`land/rerun/discard`) with reason

### WIP limits

- [ ] Max 3 active branches total.
- [ ] Max 1 aggressive branch active.
- [ ] No new branch if previous branch decision is unresolved.

---

## 4) Candidate Card template (aggressive branches)

Base fields (required by current gate):
- [ ] `shape`
- [ ] `deleted_cost_center`
- [ ] `expected_upside_source`
- [ ] `why_larger_than_noise`
- [ ] `touched_symbols_or_regions`
- [ ] `forbidden_edits`
- [ ] `success_gate`

Aggressive extensions (mandatory for aggressive branches):
- [ ] `aggression_class` (`contract_jump` | `kernel_body_jump` | `runtime_jump`)
- [ ] `milestone_target` (`12us` | `10us` | `8us` | `7us`)
- [ ] `predicted_delta_us` (geomean + touched shape)
- [ ] `blast_radius` (exact symbols + shape surfaces)
- [ ] `kill_switch` (env var or dispatch flag, default OFF)
- [ ] `rollback_plan` (exact commit/base to revert)
- [ ] `quota_budget` (`1 test + 1 benchmark + optional 1 rerun max`)
- [ ] `early_abort_conditions` (e.g., correctness fail, `>0.5 us` regression)
- [ ] `proof_not_relocated_work` (especially for `A-pack`-adjacent ideas)
- [ ] `required_observability` (what must appear in profile summary)

Aggressive branch acceptance checklist:
- [ ] Card passes all base + aggressive fields.
- [ ] Kill-switch verified locally before remote spend.
- [ ] Verification agent signs off quota budget.

---

## 5) Definition of Done (DoD) per milestone

### DoD: 12 us milestone
- [ ] Geomean `<=12.0 us` on two agreeing benchmark runs.
- [ ] `m16/m64/m256` inside M1 windows.
- [ ] `m4/m32` under M1 guardrails.
- [ ] At least 2 whole-bucket deletions landed on hot shapes.
- [ ] Updated branch queue and profile artifacts committed.

### DoD: 10 us milestone
- [ ] Geomean `<=10.0 us` on two agreeing benchmark runs.
- [ ] `m16/m64/m256` inside M2 windows.
- [ ] `m4/m32` under M2 guardrails.
- [ ] No unresolved correctness caveats in landed branches.
- [ ] At least one successful compound branch from independently verified wins.

### DoD: 8 us milestone
- [ ] Geomean `<=8.0 us` on two agreeing benchmark runs.
- [ ] `m16/m64/m256` inside M3 windows.
- [ ] `m4/m32` under M3 guardrails (now active movers).
- [ ] Any aggressive branch either landed with kill-switch default ON for rollout safety, or was discarded and logged.
- [ ] Fresh profile shows no reopened banned lane.

### DoD: 7 us milestone
- [ ] Geomean `<=7.0 us` on two agreeing benchmark runs.
- [ ] `m16/m64/m256` inside M4 windows.
- [ ] `m4/m32` under M4 guardrails.
- [ ] Final trunk profile + benchmark evidence archived.
- [ ] Promotion packet prepared (benchmark evidence, stability evidence, risk notes).

---

## 6) Anti-patterns to avoid (quota waste patterns)

Hard veto list:
- [ ] No branch without a named deleted cost center.
- [ ] No multi-shape branch (except controlled compound of already-verified commits).
- [ ] No prep+scheduling mixed edits.
- [ ] No re-open of known losing `A-pack` shapes (per-CTA re-quant, fixed-CTA persistent sweeps, grid-coop thin sweeps).
- [ ] No leaderboard spend before two benchmark wins.

Quota waste patterns (explicit):
- [ ] Chasing noise deltas (`<0.05 us`) with repeated benchmark spends.
- [ ] Spending remote quota without local kill-switch validation on aggressive branches.
- [ ] Re-running old negative lanes without new profile evidence.
- [ ] Opening new branch before prior `land/rerun/discard` decision.
- [ ] Wrapper-only cleanup branches that do not delete a whole bucket.
- [ ] Ignoring m4/m32 guardrail regressions while chasing m16/m64/m256.

Operational hygiene:
- [ ] Log every discard with exact failure mode to prevent repeat spend.
- [ ] Freeze lane after 3 consecutive negatives until new profile evidence arrives.
- [ ] Keep candidate-card-first discipline on every round.

---

## 7) Immediate next actions (from current t20 base)

- [ ] Lock `t20` as baseline in branch queue for this roadmap.
- [ ] Start M1 with `ms12/m64-address-law-r1`.
- [ ] In parallel, research agent prepares `m16-orchestration-delete` and `m256-orchestration-delete` cards.
- [ ] Verification agent enforces M1 windows/guardrails exactly as written.
