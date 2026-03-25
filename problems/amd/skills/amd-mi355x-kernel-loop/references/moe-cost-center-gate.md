# MoE Cost-Center Gate

Use this before opening or approving any `moe_mxfp4` branch.

## Candidate Card

Every MoE branch must start with exactly one Candidate Card containing:

- `lane`
- `regime_tag`
- `deleted_cost_center`
- `expected_upside_source`
- `why_larger_than_noise`
- `forbidden_edits`
- `success_gate`

`lane` must be one of:

- `dispatch_pack`
- `stage1_core`
- `stage2_reduce`
- `shared_expert`
- `full_pipeline`

`regime_tag` must be one of:

- `re256_de256_bs16_topk8`
- `re256_de256_bs128_topk8`
- `re256_de256_bs512_topk8`
- `re32_de512_bs16_topk8`
- `re32_de512_bs128_topk8`
- `re32_de512_bs512_topk8`
- `re32_de2048_bs512_topk8`

## Code-Enforced Vetoes

The coordinator should block remote spend when any of these are true:

- the Candidate Card is incomplete
- the evidence pack is incomplete: missing `motivation_refs` or `retrieval_queries`
- a non-anchor lane still routes the hot path through `fused_moe(`
- `topk_ids` or `topk_weights` are no longer visible in `custom_kernel`
- the candidate does not visibly own its declared lane
- a non-anchor candidate rebuilds all experts in Python

## Prompt And Review Vetoes

Reject the branch even before code review when any of these are true:

- it changes more than one lane
- it touches more than one regime family
- it changes routing semantics or top-k semantics
- it uses an anchor fallback instead of owning correctness for the active contract
- it mixes structural cost-center deletion with low-level MFMA or occupancy tuning
- it claims progress while the hot path is still anchor-backed
- it cannot name one deleted MoE bucket that is larger than normal rerun noise

## Lane Order

Open branches in this order only:

1. `dispatch_pack`
2. `stage1_core`
3. `stage2_reduce`
4. `shared_expert`
5. `full_pipeline`

## Success Ladder

- `dispatch_pack`: clear `re256_de256_bs512_topk8` win and global `<170 us`
- `stage1_core`: global `<150 us` with no >7% regressions outside the target lane
- `stage2_reduce`: clear `re32_de2048_bs512_topk8` win and global `<135 us`
- `full_pipeline`: `<120 us` before serious scaled-MFMA specialization
- milestone 1: stable `<=140 us` geomean
- scaled-MFMA/LDS tuning starts only after a stable `stage1_core` win

## Anti-Goals

- do not spend the main budget on AITER wrapper tuning
- do not start with scaled-MFMA as the first MoE bet
- do not ship a HIP kernel that preserves the same padded, relaunch-heavy generic MoE structure
- do not hide anchor fallback behind helper indirection and call the branch native
