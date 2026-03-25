# MoE Branch Queue

Use this as the first-week queue for `moe_mxfp4`.

## Branch 1

- Variant: `dispatch_pack_sparse256`
- Lane: `dispatch_pack`
- Regime: `re256_de256_bs512_topk8`
- Deleted cost center: repeated routed-token sort/regroup/padding before expert compute
- Retrieval pack:
  - `q29-fused-moe-padding-free-packing`
  - `q32-fused-moe-github-motivation-links`
  - `padding-free routed expert packing touched experts`
  - `sorted_token_ids sorted_expert_ids num_valid_ids`
- Expected upside source: ScatterMoE-style touched-expert packing plus SonicMoE tile-aware sparse routing
- Forbidden edits:
  - call `fused_moe(` in the hot path
  - change more than the `dispatch_pack` lane
  - rebuild all experts in Python
- Success gate: clear `re256_de256_bs512_topk8` win and global `<170 us`

## Branch 2

- Variant: `stage1_grouped_bf16_sparse256_v1`
- Lane: `stage1_core`
- Regime: `re256_de256_bs512_topk8`
- Deleted cost center: repeated stage1 launch plus unfused routed gate/up materialization after dispatch
- Retrieval pack:
  - `q30-fused-moe-persistent-pipeline`
  - `stage1 grouped bf16 gate up swiglu fused expert tile pipeline`
  - `ck_moe_stage1 block_m sorted_weights shuffled scale-aware`
- Expected upside source: no-anchor grouped stage1 compute over packed routed and shared-expert rows with a stage1-owned output layout
- Forbidden edits:
  - change dispatch or top-k semantics
  - use an anchor fallback for non-target shapes
  - ignore appended shared experts in `topk_ids`
  - add stage2 ownership in the same branch
- Success gate: pass `test`, then benchmark for a stable global win large enough to justify the next stage1 branch

## Branch 2B

- Variant: `stage1_grouped_bf16`
- Lane: `stage1_core`
- Regime: `re32_de512_bs512_topk8`
- Deleted cost center: anchor-backed gate/up stage launches and repeated expert metadata walking after dispatch
- Retrieval pack:
  - `q30-fused-moe-persistent-pipeline`
  - `stage1 grouped bf16 gate up swiglu fused expert tile pipeline`
  - `ck_moe_stage1 block_m sorted_weights shuffled scale-aware`
- Expected upside source: touched-expert-only grouped stage1 compute with stable bf16 math before deeper HIP tuning
- Forbidden edits:
  - change dispatch semantics
  - add stage2 ownership in the same branch
  - eagerly dequantize every expert in Python
- Success gate: native stage1 path plus fused dispatch beats the control on `re32_de512_bs512_topk8` and global `<150 us`

## Branch 3

- Variant: `stage2_grouped_weighted`
- Lane: `stage2_reduce`
- Regime: `re32_de2048_bs512_topk8`
- Deleted cost center: separate stage2 reduction plus weighted combine/index-add overhead
- Retrieval pack:
  - `q30-fused-moe-persistent-pipeline`
  - `stage2 weighted reduction down projection index_add expert outputs`
  - `ck_moe_stage2 sorted_token_ids sorted_expert_ids weighted epilogue`
- Expected upside source: native grouped stage2 with weighted epilogue in the heavy large-expert regime
- Forbidden edits:
  - change dispatch or stage1 in the same branch
  - rebuild every expert in Python
  - keep weighted reduction outside the owned stage2 path
- Success gate: clear win on `re32_de2048_bs512_topk8` and end-to-end `<135 us`

## Milestone

- First milestone: stable `<=140 us` geomean with no anchor fallback in the native hot path.
- Only after a stable `stage1_core` win should MFMA/LDS tuning begin.
