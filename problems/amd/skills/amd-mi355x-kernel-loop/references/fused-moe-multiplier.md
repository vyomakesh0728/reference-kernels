# Fused MoE Multiplier Canon

Read this before planning or prompting sub-agents for `moe_mxfp4`.

Pair it with:

- `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-cost-center-gate.md`
- `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-branch-queue.md`
- `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-subagent-prompt.md`

## Local Canon

- `/root/reference-kernels/problems/amd/important_papers/fused_moe/README.md`
- `/root/reference-kernels/problems/amd/important_papers/fused_moe/architectural_multipliers.md`
- `/root/reference-kernels/problems/amd/important_papers/fused_moe/links.md`
- `/root/reference-kernels/problems/amd/important_papers/fused_moe/scattermoe.md`
- `/root/reference-kernels/problems/amd/important_papers/fused_moe/sonicmoe.md`
- `/root/reference-kernels/problems/amd/important_papers/fused_moe/flashmoe.md`
- `/root/reference-kernels/problems/amd/important_papers/fused_moe/tutel.md`
- `/root/reference-kernels/problems/amd/important_papers/fused_moe/deepseek_v3.md`
- `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-cost-center-gate.md`
- `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-branch-queue.md`
- `/root/reference-kernels/problems/amd/skills/amd-mi355x-kernel-loop/references/moe-subagent-prompt.md`

## Mandatory Rules

- keep `topk_ids` and `topk_weights` fixed
- do not claim progress if the hot path still goes through `fused_moe`
- do not spend search budget on router changes
- optimize structural cost centers first: packing, persistent scheduling, shared-expert split, stage fusion
- require one Candidate Card, one lane, and one regime per branch

## Raw Motivation Links

- `https://github.com/microsoft/tutel`
- `https://github.com/deepseek-ai/DeepSeek-V3`
- `https://github.com/shawntan/scattermoe`
- `https://github.com/osayamenja/FlashMoE`
- `https://github.com/Dao-AILab/sonic-moe`
