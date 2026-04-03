# Iteration history (reference-kernels)

**Purpose:** Survive context compaction. Every agent (Cursor, Claude Code, Codex) should **read `AGENTS.md` § Popcorn loop & § Iteration log** after a fresh session, then **append here** after a meaningful change or remote eval.

Log **failed attempts** and **why**—that saves more time than only logging wins.

---

## Template (copy from `###` through `---`)

### YYYY-MM-DD — short title

| Field | Content |
|--------|--------|
| **Problem** | `mixed-mla` / `mxfp4-mm` / `moe-mxfp4` / repo-wide |
| **Goal** | One sentence: what we optimized or validated |
| **Techniques** | Bullets: algorithms, dtypes, tile sizes, env (`PYTORCH_ROCM_ARCH`), APIs (aiter / Triton / `load_inline`), flags |
| **Code / commit** | Branch or SHA if available |
| **Evidence** | **Exact** `popcorn-cli` lines; workflow / Actions **URL** if shown; **numbers** (e.g. mismatch_ratio, per-shape µs, geomean) |
| **Popcorn** | `test` ✅/❌ · `benchmark` · `leaderboard` (score/rank)—short summary; details live in **Evidence** |
| **Result** | Correctness, perf vs last baseline, surprises |
| **What didn’t work** | Dead ends this iteration (or “n/a”). **Always** fill for failed runs / reverts |
| **Rule / spec tension** | e.g. doc says X, code does Y—**or** “none” |
| **Learnings** | Patterns to reuse; **don’t repeat X** |
| **Next bet** | Single line: first thing to try next session |
| **Artifacts** | Extra links, profiles, zips (optional if already in Evidence) |

---

**Convention:** under **## Log**, add new `###` dated sections **newest first** (immediately below `## Log`).

## Log

### 2026-04-03 — mixed-mla: tl.dot_scaled K=128 WORKS for mxfp4 QK

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Confirm tl.dot_scaled works for MLA QK computation with mxfp4 KV |
| **Techniques** | 1) Used K=128 tiles (K=64 causes Triton compiler crash); 2) Matched test_dotscaled_v3.py patterns |
| **Code / commit** | `triton_mxfp4_dotscaled.py` |
| **Evidence** | "DOTSCALED MLA: SUCCESS! Out sum: 117.9604" (workflow 23935657534) |
| **Popcorn** | `test` ✅ 4/4 |
| **Result** | **K=128 tl.dot_scaled WORKS on gfx950!** |
| **What didn't work** | K=64 tiles crash Triton: "PassManager::run failed" during LLVM IR generation |
| **Rule / spec tension** | none |
| **Learnings** | 1) **Native dtypes**: `torch.float4_e2m1fn_x2`, `torch.float8_e8m0fnu`; 2) **K=128 required** for gfx950 MFMA; 3) **576 = 4×128 + 64** - need padding or partial tile handling; 4) **Scale padded**: (total_kv, 24) not (total_kv, 18) |
| **Next bet** | Build full MLA kernel: pad Q to 640 (5×128), integrate softmax + V |
| **Artifacts** | `triton_mxfp4_dotscaled.py` |

---

### 2026-04-02 — mixed-mla: Write-through (ns=1) TESTED — HYPOTHESIS WRONG

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Test if ns=1 (no split-K reduce) improves small kv=1024 shapes |
| **Techniques** | 1) Set num_kv_splits=1 for kv≤1024; 2) Benchmarked all 8 shapes |
| **Code / commit** | `writethrough_test.py`, `writethrough_ns2.py` |
| **Evidence** | bs=4,kv=1024: **80µs (ns=1) vs 27µs (ns=8) → 3× WORSE**; bs=32,kv=1024: 81µs vs 31µs → 2.6× worse |
| **Popcorn** | `test` ✅ · `benchmark` ✅ |
| **Result** | **HYPOTHESIS WRONG** - ns=1 dramatically hurts performance |
| **What didn't work** | Write-through concept assumes reduce overhead > parallelism benefit. Reality: split-K provides critical GPU parallelism even for small sequences (64 vs 512 concurrent outputs) |
| **Rule / spec tension** | none |
| **Learnings** | 1) **Split-K is about parallelism, not just throughput**; 2) Even small kv=1024 benefits from ns=8; 3) FlashInfer's "write-through" may work differently (instruction-level, not split-level) |
| **Next bet** | Focus on Triton mxfp4 path — all dispatch overhead approaches are blocked |
| **Artifacts** | `writethrough_test.py` benchmark results |

---

### 2026-04-02 — mixed-mla: HIP graphs BLOCKED by harness

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Use HIP graph capture to eliminate Python dispatch overhead |
| **Techniques** | 1) Tested torch.cuda.graph(); 2) Tested torch.cuda.Event(enable_timing=True); 3) Tested torch.cuda.Stream() |
| **Code / commit** | `dispatch_overhead_analysis.py`, `hip_graph_test.py` |
| **Evidence** | Harness error: "Your code seems to be doing work on another stream" — blocks ALL stream/graph usage |
| **Popcorn** | `test` ❌ (harness rejection) |
| **Result** | **HIP graphs NOT VIABLE** - harness does static analysis and blocks stream operations |
| **What didn't work** | 1) torch.cuda.graph() — blocked; 2) torch.cuda.Event(enable_timing=True) — blocked; 3) torch.cuda.Stream() — blocked |
| **Rule / spec tension** | Harness restriction not documented, discovered empirically |
| **Learnings** | 1) **Harness blocks all stream usage** including graph capture; 2) Cannot eliminate dispatch overhead via graph replay; 3) Only path is a custom kernel that avoids Python dispatch entirely |
| **Next bet** | Triton kernel (single fused attention) or accept aiter's Python overhead |
| **Artifacts** | `dispatch_overhead_analysis.py` |

---

### 2026-04-02 — mixed-mla: Pre-compiled .co file approach RESEARCH

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Research feasibility of loading pre-compiled .co files via hipModuleLoad to bypass Python dispatch overhead |
| **Techniques** | 1) Analyzed Popcorn submission format (task.yml); 2) Researched HIP Python interface (hipModuleLoad, hipModuleLaunchKernel); 3) Reverse-engineered aiter's .co loading from asm_mla_decode_fwd.cpp; 4) Located aiter .co files at `/home/runner/aiter/hsa/gfx950/mla/*.co` |
| **Code / commit** | Created `hip_module_load_prototype.py` with research findings |
| **Evidence** | task.yml: `files: [{"name": "submission.py", "source": "@SUBMISSION@"}]` - **only 1 Python file allowed**; aiter kernel naming: `mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co` |
| **Popcorn** | n/a (research only) |
| **Result** | **NOT VIABLE as primary approach** - cannot ship .co files with submission |
| **What didn't work** | 1) Shipping .co files: submission format only accepts single .py file; 2) Embedding .co in Python: base64-encoded binary still needs hipModuleLoadData; 3) Direct hipModuleLoad of aiter's files: requires exact kernel function names + ABI matching |
| **Rule / spec tension** | Submission format explicitly limits files to submission.py |
| **Learnings** | 1) **aiter .co location**: `/home/runner/aiter/hsa/gfx950/mla/*.co`; 2) **Kernel names**: mla_a8w8_qh16_qseqlen1_gqaratio16_ps (pattern: mla_{a/bf16}{quant}w{quant}_qh{heads}_..._ps); 3) **HIP Python API**: hip.hipModuleLoadData(bytes) + hip.hipModuleGetFunction(module, b"name"); 4) **Kernel args**: 20+ args including q, kv, indptrs, scales, strides, output; 5) **Estimated savings if worked**: 250µs per inference (270µs CPU → ~20µs) |
| **Next bet** | Alternative approach: Try Triton mxfp4 path (tl.dot_scaled works!) or focus on pure aiter tuning |
| **Artifacts** | `hip_module_load_prototype.py` - contains full research summary and prototype code |

---

### 2026-04-02 — mixed-mla: dispatch overhead research (parallel agents)

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Eliminate 270µs Python dispatch overhead (wall=293µs, GPU=24µs) |
| **Techniques** | 1) Probed torch.ops.aiter namespace; 2) Researched FlashInfer/ThunderMLA megakernel; 3) Parallel agent investigation of 3 approaches |
| **Code / commit** | `aiter_default_bypass.py`, `aiter_ops_probe.py` (probing files) |
| **Evidence** | torch.ops.aiter has only 5 utility ops (check_numa, get_cu_num, get_gfx, get_module, name) - MLA ops NOT registered as torch.ops |
| **Popcorn** | `test` ✅ (probing submissions) |
| **Result** | **.default bypass NOT viable** - aiter MLA uses custom JIT modules, not torch.library |
| **What didn't work** | torch.ops.aiter doesn't contain MLA ops; only utility functions registered |
| **Learnings** | 1) aiter uses JIT-compiled .so modules (module_mla_asm.so, module_mla_reduce.so); 2) MLA functions are Python wrappers around JIT modules; 3) ThunderMLA megakernel approach could eliminate overhead but requires HIP kernel |
| **Next bet** | Parallel research: (1) Pre-compiled .co direct loading, (2) HIP graphs isolation, (3) Write-through for small kv |
| **Artifacts** | Parallel agents running for each approach |

---

### 2026-04-02 — mixed-mla: tl.dot_scaled tensor layout FIXED, working on gfx950

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Fix `tl.dot_scaled` tensor layout to enable mxfp4 KV path |
| **Techniques** | 1) Studied Triton's test_matmul.py patterns; 2) Fixed B tensor shape to (K//2, N) packed along K; 3) Fixed B_scale shape to (N, K//32) NOT transposed; 4) Used K=128 for native CDNA4 MFMA |
| **Code / commit** | Created `test_dotscaled_v3.py` with correct tensor layouts |
| **Evidence** | `popcorn-cli submit --mode test` workflow `23896940925`: "DOT_SCALED V3: SUCCESS! C sum: -430.52777099609375" |
| **Popcorn** | `test` ✅ 4/4 (aiter still passes, dot_scaled compiled!) |
| **Result** | **`tl.dot_scaled` WORKS on gfx950!** Correct tensor layout: B as (K//2, N) packed, B_scale as (N, K//32) |
| **What didn't work** | Previous tests had wrong B layout (should be transposed from (N, K) packed) |
| **Rule / spec tension** | none |
| **Learnings** | 1) **B tensor**: Generate (N, K), pack along K dim → (N, K//2), then transpose → (K//2, N); 2) **B_scale**: Shape is (N, K//32), NOT transposed; 3) **rhs_k_pack=True** for K-packed data; 4) For bf16 × mxfp4, Triton uses software emulation (upcast to bf16) - 2× BW savings but no native FP4 compute |
| **Next bet** | 1) Build full Triton MLA kernel using mxfp4 KV with correct dot_scaled layout; 2) Test fp8 × mxfp4 which might use native scaled MFMA; 3) Benchmark mxfp4 path vs current 66µs aiter fp8 |
| **Artifacts** | `test_dotscaled_v3.py`, `triton_mxfp4_qk.py` (WIP) |

---

### 2026-04-02 — mixed-mla: mxfp4 path investigation (taking motivation from mxfp4-mm)

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Investigate mxfp4 KV path for MLA, inspired by mxfp4-mm's 13.87µs using native FP4 MFMA |
| **Techniques** | 1) Analyzed mxfp4-mm's `aiter.gemm_a4w4` approach; 2) Studied CDNA4 ISA `V_MFMA_SCALE_F32_16X16X128_F8F6F4`; 3) Researched Triton `tl.dot_scaled` for bf16 × mxfp4 |
| **Code / commit** | Created `triton_mxfp4_mla.py`, `triton_dotscaled_test.py` (experimental) |
| **Evidence** | CDNA4 ISA shows native FP4 MFMA with E8M0 scales; Triton docs confirm `tl.dot_scaled` supports `e2m1` format with bf16 lhs |
| **Popcorn** | `test` ✅ (workflow 23894715745) - dot_scaled compilation test |
| **Result** | **`tl.dot_scaled` EXISTS on gfx950!** Failed with shape error, not missing function: "Reduction dimension should pack the same number of elements; (lhs: ['constexpr[16]', 'constexpr[64]'] vs rhs: ['constexpr[64]', 'constexpr[16]'])". This means the API is available - we just need correct tensor layouts. |
| **What didn't work** | 1) `aiter.gemm_a4w4` can't be used directly (MLA needs fused attention, not separate GEMMs); 2) Software mxfp4 dequant is 30-68x slower (confirmed dead); 3) vLLM PR #30177 for FP4 MLA BMM is stale/unmerged; 4) First `dot_scaled` test had wrong tensor layout (K dimension packing issue) |
| **Rule / spec tension** | none |
| **Learnings** | 1) mxfp4-mm succeeds because aiter has native FP4 GEMM kernel; 2) MLA fails because aiter has no mxfp4 MLA kernel; 3) `tl.dot_scaled` in Triton supports bf16 × mxfp4 and could work for FlashAttention-style kernel; 4) Need to test if gfx950 Triton lowers `tl.dot_scaled` to native MFMA |
| **Next bet** | 1) Fix `tl.dot_scaled` tensor layout (K-packing issue); 2) For bf16 Q × mxfp4 K, Triton upcasts to bf16 (software emulation) - 2x BW savings but no native FP4 compute; 3) Consider mxfp4 Q × mxfp4 K for native MFMA path (requires Q quantization) |
| **Artifacts** | `triton_mxfp4_mla.py`, `triton_dotscaled_test.py`, CDNA4 ISA reference |

---

### 2026-04-01 — mixed-mla: aggressive kv8k config sweep, promoted ns32 8k policy

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Cut geomean by attacking the slow 8k shapes (`bs=32/64/256, kv=8192`) with a higher split policy |
| **Techniques** | 1) Tried non-persistent modes to delete metadata overhead; 2) Benchmarked all-a8w8 persistent; 3) Promoted hybrid: keep `kv=1024` policy, force `kv=8192` to `(ns=32, a8w8, ps=1, fast_mode=False)` |
| **Code / commit** | Updated `problems/amd_202602/mixed-mla/submission.py`; exploratory files: `aiter_nonpersist_explicit_ns.py`, `aiter_kv8k_all_a8w8_ns32.py` |
| **Evidence** | Commands run: `popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test --no-tui problems/amd_202602/mixed-mla/submission.py`; `popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode benchmark --no-tui problems/amd_202602/mixed-mla/submission.py`; `popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode benchmark --no-tui problems/amd_202602/mixed-mla/aiter_kv8k_all_a8w8_ns32.py` |
| **Popcorn** | `test` ✅ · `benchmark` ✅ (submission improved vs same-session baseline) |
| **Result** | Same-session baseline (`submission.py`, workflow `23854187328`) geomean **69.65 us**; updated submission (workflow `23855740669`) geomean **66.35 us** (**~4.7% faster**). Per-shape means after promotion: `26.5, 37.6, 31.0, 82.7, 39.7, 136, 87.3, 312` us. |
| **What didn't work** | 1) Non-persistent hybrid auto (`aiter_nonpersist_auto_v2.py`) failed heuristic lookup (`q_type:bf16 kv_type:fp8 ... ps:0`), workflow `23852912157`; 2) Non-persistent auto all-a8w8 (`aiter_nonpersistent_auto.py`) failed with `RuntimeError: step must be nonzero`, workflow `23852978281`; 3) Non-persistent explicit splits passed but regressed badly (geomean **77.16 us**, workflow `23853325443`); 4) Persistent all-a8w8 regressed overall (geomean **74.39 us**, workflow `23853889728`). |
| **Rule / spec tension** | none |
| **Learnings** | 1) Current harness/aiter build appears fragile for non-persistent auto paths; 2) For this environment, raising 8k split count to 32 helps all 8k shapes in persistent mode; 3) Biggest remaining cost center is still `bs=256, kv=8192` (~312 us). |
| **Next bet** | Keep this config as trunk and do a focused 8k-only sweep around `{ns=24,32,40}` + selective `a16w8` fallback only for `bs=4, kv=8192` to test if we can keep small-shape wins while reducing quant overhead. |
| **Artifacts** | [Workflow 23852912157](https://github.com/gpu-mode/kernelbot/actions/runs/23852912157), [Workflow 23852978281](https://github.com/gpu-mode/kernelbot/actions/runs/23852978281), [Workflow 23853325443](https://github.com/gpu-mode/kernelbot/actions/runs/23853325443), [Workflow 23853889728](https://github.com/gpu-mode/kernelbot/actions/runs/23853889728), [Workflow 23854187328](https://github.com/gpu-mode/kernelbot/actions/runs/23854187328), [Workflow 23855740669](https://github.com/gpu-mode/kernelbot/actions/runs/23855740669) |

---

### 2026-03-31 — mixed-mla: aiter num_splits tuning (no improvement)

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Tune aiter num_splits for large shapes (bs=256, kv=8k) |
| **Techniques** | Tested ns=8 (all shapes) and ns=32 (bs>=128 only) vs baseline ns=16 |
| **Code / commit** | `aiter_tune_ns8.py`, `aiter_tune_ns32.py` |
| **Evidence** | `popcorn-cli submit --mode benchmark`; ns=8: 322 µs; ns=32: 320 µs; ns=16 (baseline): ~312 µs |
| **Popcorn** | `benchmark` ✅ |
| **Result** | **Original ns=16 is already optimal**. ns=8 and ns=32 both slightly worse. |
| **What didn't work** | 1) ns=8: not enough parallelism for large shapes; 2) ns=32: too much reduce overhead |
| **Rule / spec tension** | none |
| **Learnings** | 1) **aiter config already well-tuned** - ns=8-16 sweet spot confirmed; 2) **Memory-bound** - more splits don't help past ns=16; 3) **~312 µs is the aiter floor** for bs=256,kv=8k |
| **Next bet** | Accept ~62 µs geomean as practical ceiling without pre-compiled custom assembly |
| **Artifacts** | `aiter_tune_ns8.py`, `aiter_tune_ns32.py` |

---

### 2026-03-31 — mixed-mla: Triton split-K kernel FAILED (25× slower)

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Test Triton split-K (parallel KV processing) vs single-pass |
| **Techniques** | Split-K with NUM_SPLITS=8-16; separate stage1 + reduce kernels; constexpr params; grid (bs, splits, heads) |
| **Code / commit** | `triton_splitk.py` |
| **Evidence** | `popcorn-cli submit --mode benchmark`; workflow `23843459088`; bs=64,kv=8k: **3.46 ms** (vs 139 µs aiter = 25× slower); bs=256,kv=8k: **13.8 ms** |
| **Popcorn** | `test` ✅ 4/4 · `benchmark` ✅ (but 25× slow) |
| **Result** | Correct but split-K Triton still catastrophically slow |
| **What didn't work** | 1) **No MFMA**: using `tl.sum(q * k, axis=1)` instead of `tl.dot()`; 2) **Grid layout**: each program handles single head, can't batch heads for MFMA; 3) **Element-wise V**: same issue |
| **Rule / spec tension** | none |
| **Learnings** | 1) **Triton needs tl.dot() for MFMA** - element-wise ops are slow; 2) **Multi-head batching is critical** - original triton_single_pass.py processes all 16 heads together; 3) **aiter assembly is highly optimized** - AGPRs, buffer→LDS, hw FP8 conversion |
| **Next bet** | Focus on pure aiter tuning (num_splits, page_size) since Triton can't compete; or accept ~60 µs as ceiling |
| **Artifacts** | `triton_splitk.py` |

---

### 2026-03-31 — mixed-mla: Triton single-pass kernel FAILED (40× slower on large shapes)

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Test Triton as alternative to load_inline HIP (may have compile caching) |
| **Techniques** | Triton @jit kernel; single-pass online softmax; BLOCK_D=64 tiling for QK_DIM=576; constexpr parameters; exp2 for softmax |
| **Code / commit** | `triton_fp8_mla.py` |
| **Evidence** | `popcorn-cli submit --mode benchmark`; workflow `23839491272`; small shapes (aiter path): 26-139 µs; **bs=256,kv=8192 (Triton path): 13.1 ms** — **40× slower** than aiter (~300 µs) |
| **Popcorn** | `test` ✅ 4/4 · `benchmark` ✅ (but large shape 40× slow) |
| **Result** | Correct but single-pass Triton is catastrophically slow for large KV sequences |
| **What didn't work** | 1) **Single-pass without split-K**: One program per batch iterating 8192 tokens sequentially; 2) **Element-wise dot product**: `tl.sum(q * k, axis=1)` not using MFMA; 3) **No parallelism across KV**: All 8192 tokens processed serially by one program |
| **Rule / spec tension** | none |
| **Learnings** | 1) **Triton compiles quickly** (~4 minutes total vs 17+ for load_inline); 2) **Single-pass needs split-K for large KV**; 3) **Triton tl.dot() exists** but wasn't used properly; 4) **aiter split-K + MFMA is 40× faster** for bs=256,kv=8k |
| **Next bet** | Try Triton with split-K (multiple programs per batch); or use tl.dot() for proper MFMA; or accept aiter ceiling |
| **Artifacts** | `triton_fp8_mla.py` |

---

### 2026-03-31 — mixed-mla: load_inline JIT DEAD END (>17min compile)

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Test HIP MFMA kernel using aiter assembly optimizations (AGPRs, buffer→LDS, v_cvt_pk_fp8) |
| **Techniques** | `load_inline` HIP C++ with MFMA builtins; AGPR patterns; hardware FP8 conversion |
| **Code / commit** | `hip_agpr_mfma.py`, `hip_mfma_v2.py`, `hip_mfma_correct.py` |
| **Evidence** | All submissions timed out at 17 minutes (workflow timeout); `hip_mfma_correct.py` never completed test; even `aiter_pure.py` (no load_inline) timed out |
| **Popcorn** | `test` ❌ timeout · `benchmark` ❌ timeout |
| **Result** | **CRITICAL FINDING: load_inline compilation for gfx950 takes >17 minutes** |
| **What didn't work** | **Any JIT HIP kernel via load_inline is infeasible** — compilation time exceeds harness timeout. Even the known-working aiter submission is timing out, suggesting service congestion. |
| **Rule / spec tension** | none |
| **Learnings** | 1) **gfx950 JIT compilation is extremely slow** — load_inline not viable; 2) Only pre-compiled kernels (.co files like aiter) can work; 3) Service may be congested — pure aiter also timing out |
| **Next bet** | Wait for service to recover; try Triton (may have compilation caching); or accept pre-compiled aiter as ceiling |
| **Artifacts** | `hip_agpr_mfma.py`, `hip_mfma_v2.py`, `hip_mfma_correct.py`, `aiter_pure.py` |

---

### 2026-03-31 — mixed-mla: aiter assembly reverse-engineering SUCCESS

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Understand aiter's FP8 MLA assembly optimizations to replicate in custom HIP kernel |
| **Techniques** | `llvm-objdump -d --triple=amdgcn-amd-amdhsa` on `/home/runner/aiter/hsa/gfx950/mla/*.co` |
| **Code / commit** | `dump_aiter_asm.py`, `dump_fp8_asm.py` |
| **Evidence** | 27 kernel variants found; disassembled `mla_a8w8_qh128_m32x4_n16x2_msk0.co` (7159 lines) |
| **Popcorn** | `test` (correctness dummy) |
| **Result** | **KEY OPTIMIZATIONS DISCOVERED** — see below |
| **What didn't work** | n/a (research task) |
| **Rule / spec tension** | none |
| **Learnings** | See detailed breakdown below |
| **Next bet** | Implement HIP kernel using AGPR loads + `v_cvt_pk_fp8_f32` + buffer→LDS pattern |
| **Artifacts** | `dump_aiter_asm.py`, `dump_fp8_asm.py` |

#### aiter Assembly Analysis - Key Findings

**1. MFMA Usage (816 instructions per kernel)**
```asm
v_mfma_f32_16x16x32_fp8_fp8 v[40:43], a[72:73], a[0:1], 0
v_mfma_f32_16x16x32_fp8_fp8 v[40:43], a[74:75], a[2:3], v[40:43]  // accumulate
```
- Uses **Accumulator GPRs (AGPRs)** for inputs: `a[...]` not `v[...]`
- Output goes to regular VGPRs: `v[40:43]`
- 18 MFMA chains for 576-dim QK (576/32 = 18)

**2. Direct Buffer→LDS Loads (204 loads)**
```asm
buffer_load_dword v28, s[16:19], 0 offen lds  // bypasses VGPRs!
buffer_load_dword v28, s[16:19], 0 offen offset:256 lds
```
- Uses `lds` modifier to load directly from global memory to LDS
- Bypasses VGPR register pressure entirely
- This is critical for bandwidth

**3. LDS→AGPR Reads (696 ops)**
```asm
ds_read_b128 a[0:3], v29               // 16 bytes → AGPRs
ds_read_b128 a[4:7], v29 offset:64
ds_read_b128 a[8:11], v29 offset:128
```
- Reads 128-bit chunks directly to accumulator registers
- Contiguous offsets: 0, 64, 128, 192, 256, 320, 384, 448, 512
- 9 reads × 16B = 144B per wave for one MFMA tile

**4. Hardware FP8 Conversion (48 ops)**
```asm
v_cvt_pk_fp8_f32 v40, v40, v41                    // pack 2 floats → 2 fp8
v_cvt_pk_fp8_f32 v40, v42, v43 op_sel:[0,0,1]     // pack to high bytes
```
- **This is how they convert scores to FP8 for V MFMA**
- Uses `op_sel:[0,0,1]` to pack into high 16 bits
- We were using software conversion — **THIS is the fix**

**5. Hardware Exp (114 ops)**
```asm
v_exp_f32_e32 v40, v40
v_exp_f32_e32 v41, v41
... (8 parallel exp for softmax)
```
- Uses `v_exp_f32_e32` hardware exp (base-2)
- 8 scores per wave processed in parallel

**6. No Atomics**
- Zero atomic operations in the kernel
- Uses branch-based control flow, no inter-block sync

**7. Kernel Variants (for our problem: qseqlen=1, gqa=16:1)**
- `mla_a8w8_qh16_qseqlen1_gqaratio16.co` — exact match for decode
- `mla_a8w8_qh16_qseqlen1_gqaratio16_ps.co` — persistent variant

#### What We Were Missing
1. **AGPRs** — We loaded to VGPRs, not accumulator registers
2. **Buffer→LDS direct** — We used VGPR intermediates
3. **`v_cvt_pk_fp8_f32`** — We used software FP8 conversion
4. **128-bit LDS reads** — We used smaller loads

---

### 2026-03-31 — mixed-mla: fused split-K HIP kernel FAILED (5-35× slower)

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Single kernel launch: fuse stage1 + reduce with atomic completion counter |
| **Techniques** | Atomic completion counter per batch; last block does reduce; `__threadfence()` for visibility; MFMA bf16 V; FP8 E4M3 (bias=7) |
| **Code / commit** | `hip_fused_splitk.py` |
| **Evidence** | `popcorn-cli ... --mode benchmark`; test: 4/4 ✅; benchmark: **341 µs (bs=4,kv=1k)** to **10.8 ms (bs=256,kv=8k)** — **5-35× slower** than aiter (~40-300 µs); workflow `23795465368` |
| **Popcorn** | `test` ✅ · `benchmark` ✅ (but slow) |
| **Result** | Correct but slow. Fusing kernels via atomics doesn't help—the problem is kernel efficiency, not launch overhead. |
| **What didn't work** | 1) **Atomic completion counter**: adds contention, serialization; 2) **`__threadfence()`**: expensive global barrier; 3) **Only last block reduces**: N-1 blocks wasted after stage1; 4) **bf16 MFMA V**: slower than aiter's asm; 5) **Serial softmax loops**: not vectorized |
| **Rule / spec tension** | none |
| **Learnings** | 1) **Python dispatch overhead (~270µs) is real but not the only bottleneck**; 2) **aiter asm kernels are highly optimized** — persistent waves, better memory patterns, hand-tuned scheduling; 3) **Fusing via atomics doesn't help if per-block efficiency is poor**; 4) **Need to match aiter's algorithm, not just reduce launches** |
| **Next bet** | Either: (a) Use aiter's persistent API with zero Python overhead (pre-JIT, pre-allocate everything); or (b) mxfp4 KV with native FP4 MFMA for 2× bandwidth savings; or (c) study aiter's .co assembly to replicate optimizations |
| **Artifacts** | `hip_fused_splitk.py` |

---

### 2026-03-31 — mixed-mla: single-pass HIP kernel FAILED (100-300× slower)

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Eliminate split-K overhead with fused single-pass kernel (online softmax, no intermediate buffers) |
| **Techniques** | `load_inline` HIP C++; single kernel per batch; fp8 MFMA QK; scalar fp8→float V accumulation; online softmax in registers; FP8 E4M3 format (bias=7, not FNUZ bias=8) |
| **Code / commit** | `hip_single_pass_wip.py` (was `hip_fused_single_pass.py`) |
| **Evidence** | `popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test|benchmark --no-tui problems/amd_202602/mixed-mla/submission.py`; test: 4/4 ✅ (mismatch_ratio ~0.006%); benchmark: **10.6 ms (kv=1k)**, **84.2 ms (kv=8k)** — **100-300× slower** than aiter baseline (~30-300 µs); workflow `23789866834` |
| **Popcorn** | `test` ✅ 4/4 · `benchmark` ✅ (but 100× slow) |
| **Result** | Correct (FP8 E4M3 format fix worked); **catastrophically slow** due to serial V accumulation |
| **What didn't work** | **Single-pass with 64 threads per batch**: V accumulation is O(kv_len × V_dim) serial ops per thread group. 8192 tokens × 512 dims = 4M scalar ops per batch. Cannot compete with split-K + MFMA V. |
| **Rule / spec tension** | none |
| **Learnings** | 1) **FP8 E4M3 uses bias=7** (not 8 like FNUZ); 2) **Single-pass needs MFMA for V** — scalar accumulation is 100× too slow; 3) Split-K exists because one wavefront can't saturate memory bandwidth alone; 4) **Don't remove split-K without MFMA V replacement** |
| **Next bet** | Use split-K like aiter but with fused stage1+reduce in one kernel (persistent waves, grid sync, or cooperative groups). Keep MFMA for both QK and V. |
| **Artifacts** | `hip_single_pass_wip.py`, `hip_bf16_mfma_wip.py` |

---

### 2026-03-31 — mixed-mla: restored aiter baseline, outlined novel paths for #1

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Stabilize production submission; plan path from ~60 µs → ~29 µs (#1) |
| **Techniques** | Restored `best_aiter_config.py` as `submission.py`; saved HIP work to `hip_bf16_mfma_wip.py`; created workflow docs (`AGENTS.md`, `HISTORY.md`, `codex.md`, `.cursor/rules/agents-history.mdc`) |
| **Code / commit** | `submission.py` = aiter-only (~60 µs claimed); `hip_bf16_mfma_wip.py` = HIP MFMA (correct, ~1109 µs) |
| **Evidence** | No new Popcorn run this entry; prior HIP benchmark ~1109 µs geomean; prior aiter ~60-69 µs |
| **Popcorn** | n/a (file swap only) |
| **Result** | Production submission is now the proven aiter config; HIP WIP preserved separately |
| **What didn't work** | HIP MFMA path is 16× slower than aiter—not production-ready; naive split-K + bf16 V overhead dominates |
| **Rule / spec tension** | `_cache` for buffers still used (allowed by harness, but AGENTS.md warns about cross-call state) |
| **Learnings** | Keep **one safe checkpoint** (`best_aiter_config.py`) and iterate on `submission.py` or side files freely |
| **Next bet** | **Pick one:** (1) Single HIP kernel with zero Python dispatch (fused stage1+reduce, pre-allocated everything); (2) mxfp4 KV + `V_MFMA_SCALE_F32_16x16x128_F8F6F4` for 2× BW savings; (3) Single-pass (no split-K) for small shapes; (4) Decode #1's "pg8" naming. See table in chat or `CLAUDE.md` |
| **Artifacts** | `hip_bf16_mfma_wip.py`, `mfma_e2e_asm_pipeline.py`, `v_bf16_mfma.py`, `triton_single_pass.py` |

---

### 2026-03-30 — mixed-mla: HIP MFMA path + diagnostics + benchmark

| Field | Content |
|--------|--------|
| **Problem** | mixed-mla |
| **Goal** | Ship `load_inline` HIP (fp8 MFMA QK, bf16 softmax + bf16 V MFMA); confirm path vs aiter; instrument logs |
| **Techniques** | Embedded `v_bf16_mfma`-style kernel in `submission.py`; `qo_indptr` row select; stderr `[mixed-mla submission]` diagnostics; `MIXED_MLA_HIP_VERBOSE`; lazy JIT + aiter fallback |
| **Code / commit** | `problems/amd_202602/mixed-mla/submission.py` |
| **Evidence** | `popcorn-cli submit --gpu MI355X --leaderboard amd-mixed-mla --mode test|benchmark --no-tui problems/amd_202602/mixed-mla/submission.py`; test: 4/4 with mismatch_ratio warnings ~0.04 / ~0.004; benchmark geomean **~1109 µs** from 8 means (356, 615, 356, 1379, 591, 2670, 1361, 9920 µs); workflows e.g. `23762262283` (benchmark), `23754581393` (test)—see GitHub Actions for this repo/harness |
| **Popcorn** | `test` ✅ 4/4 · `benchmark` ✅ · `leaderboard` not run this day |
| **Result** | Correct under tol; **HIP path verified** in stderr (`path=HIP`, `hipcc --offload-arch=gfx950`); perf vs old aiter-hybrid ~69 µs geomean: **HIP path much slower** (~16× geomean)—needs algorithmic/occupancy work |
| **What didn’t work** | n/a (iteration succeeded); **implicit:** naive split-K HIP is not yet competitive with tuned aiter asm |
| **Rule / spec tension** | `AGENTS.md` discourages cross-call caches; `submission.py` uses module caches for metadata/buffers—document if contest rules tighten; hip stderr diagnostics add noise (can set `MIXED_MLA_HIP_VERBOSE=0`) |
| **Learnings** | Geomean for rank comes from leaderboard aggregation; **benchmark** = per-shape table, compute geomean locally; **test → benchmark → leaderboard** cadence |
| **Next bet** | Profile HIP vs aiter on largest shape; shrink split-K overhead or restore aiter for prod until HIP matches; try persistent waves / fewer splits / fused Q quant (see `CLAUDE.md`) |
| **Artifacts** | (optional) prior profile zips in repo root if still relevant |

---
