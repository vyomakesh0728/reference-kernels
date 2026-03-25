# Fused MoE Research Pack

This directory is the local paper and repo canon for `moe_mxfp4`.

It exists for two reasons:

1. give human operators one place to read the structural MoE kernel ideas
2. give Codex/LLM sub-agents a stable local corpus that can be indexed by `amd_kernel_rag`

The working rule for this repo is:

- keep `topk_ids` and `topk_weights` fixed
- attack packing, padding, IO, kernel boundaries, and expert-stage fusion
- do not import model-level routing changes into the leaderboard path

## Files

- `links.md`: raw GitHub and paper links for agent prompts
- `architectural_multipliers.md`: short list of the highest-value transferable bets
- `scattermoe.md`: padding-free packing and fused reorder/linear notes
- `sonicmoe.md`: tile-aware scheduling and IO overlap notes
- `flashmoe.md`: persistent fused pipeline notes
- `tutel.md`: adaptive execution and regime split notes
- `deepseek_v3.md`: shared-expert and fine-grained-routed-expert workload notes

## Current Repo Fit

The live MoE contract already provides routed experts as input and the current repo submission still routes the hot path through `aiter.fused_moe(...)`.

That means the useful ideas here are:

- padding-free routed-expert packing
- fused reorder plus expert compute
- persistent expert-tile scheduling
- shared-expert split
- regime-specialized kernels

The ideas here are not a license to change routing semantics.
