# MXFP4 Dual-Agent Autoloop (Research + Optimization)

## Mission
Drive MXFP4 portfolio geomean from ~22.6 µs toward ≤7.0 µs by running two autonomous roles:

1. **Research Scout Agent** — derives the next best bucket ideas across all shapes.
2. **Optimization Agent** — implements, tests, and benchmarks the highest-value idea.

The loop is portfolio-first (m4/m16/m32/m64/m256). No single-shape wins that regress other shapes.

## KernelGuard Compliance (Required)

All contract changes must remain KernelGuard-safe. The following are **disallowed** because
KernelGuard flags them as exploit patterns or timing manipulation:

- Cross-call result caching (data_ptr/id/_version based caches that short-circuit compute).
- CUDA/CUDAGraph replay or graph caching across calls.
- Unsynchronized multi-stream dispatch.
- Timer monkey-patching, stdout injection, or harness patching.

These are **allowed** (KernelGuard-safe when done without short-circuit returns):

- Compile-time shape specialization.
- Direct-entry dispatch and constant-body kernels.
- Single-stream, single-call execution with no cached outputs.
- JIT module caching for compiled kernels (no output reuse).

---

## Research Scout Agent Prompt (xhigh)

**Role:** Produce 1–3 Candidate Cards per round that are portfolio-positive and respect A-pack rules.

**Inputs to load each round**
- `skills/amd-mi355x-kernel-loop/SKILL.md`
- `skills/amd-mi355x-kernel-loop/references/mxfp4-exact-shape-frontier.md`
- `skills/amd-mi355x-kernel-loop/references/mxfp4-profile-branch-queue.md`
- `skills/amd-mi355x-kernel-loop/references/mxfp4-portfolio-ladder.md`
- `skills/amd-mi355x-kernel-loop/references/mxfp4-subagent-prompt.md`
- Any recent benchmark logs under `.agent-loop/harness_runs/mxfp4_mm/`

**Model policy**
- All scouts must use `gpt-5.2` with `reasoning_effort=xhigh`.

**Hard constraints**
- A-pack is **paper-first only**. No implementation unless:
  1) no per-N re-quant,
  2) no thin-shape parallelism collapse,
  3) total quant work drops.
- No harness/caller contract changes unless explicitly authorized.
- m64 is fragile: only touch if bucket is launch/temp/setup (no A-pack changes).
- Any contract change must avoid KernelGuard disallowed patterns listed above.

**Candidate Card template**
```
Candidate Card:
  title:
  bucket(s):
  shapes helped:
  expected geomean delta:
  mechanism:
  regression risk:
  proof-of-fit (why portfolio-positive):
  test/bench gate:
  files/symbols to touch:
```

---

## Optimization Agent Prompt (main)

**Role:** Implement exactly one Candidate Card per round, then test + benchmark.

**Loop protocol**
1. **Plan**: restate the candidate, expected geomean win, and gate.
2. **Implement**: smallest possible diff, no unrelated changes.
3. **Test**: run test stage if required by coordinator.
4. **Benchmark**: always run full portfolio benchmark.
5. **Evaluate**:
   - keep if geomean improves ≥0.2 µs and no shape regresses >0.1 µs.
   - discard if regression or gain <0.05 µs.
6. **Log**: append to `agent-log.md` with decision.

**A-pack guardrail**
- If a candidate touches A-pack without satisfying paper gate, reject it immediately.

---

## Mermaid Flow — Dual Agent Loop

```mermaid
flowchart TD
  START([Start: v135 base ~22.6 µs]) --> SCOUT

  SCOUT[Research Scout (xhigh)\nRead skills + latest benches\nPropose 1-3 Candidate Cards] --> PICK
  PICK{Coordinator picks top\nportfolio-positive card} -->|Yes| IMPLEMENT
  PICK -->|No viable card| SCOUT

  IMPLEMENT[Optimization Agent\nImplement minimal diff] --> TEST
  TEST{Test required?} -->|Yes| RUN_TEST
  TEST -->|No| RUN_BENCH
  RUN_TEST --> RUN_BENCH[Run benchmark\nCollect geomean + per-shape]

  RUN_BENCH --> EVAL{Geomean ≥0.2 µs gain\n& no regressions >0.1 µs?}
  EVAL -->|Yes| LAND[Land new base\nUpdate agent-log.md] --> SCOUT
  EVAL -->|No| DISCARD[Discard + log\nRevert to last base] --> SCOUT
```

---

## Notes
- The Research Scout should stay ahead of the Optimization Agent by one iteration.
- If A-pack is deemed infeasible on paper, pivot to non-A-pack buckets or explicitly request contract changes.
- Use xhigh reasoning for all scouts to avoid shallow conclusions.
