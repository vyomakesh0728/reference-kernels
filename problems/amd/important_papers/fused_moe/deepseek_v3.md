# DeepSeekMoE And DeepSeek-V3 Digest

## Transferable Idea

DeepSeekMoE and DeepSeek-V3 explain the workload shape better than they explain the kernel.
The repo-fit lesson is that shared experts and many fine-grained routed experts should not be forced through one identical schedule.

## What To Borrow

- explicit shared-expert split
- fine-grained routed-expert regime awareness
- expect sparse-per-expert loads even at larger token counts

## What Not To Borrow Blindly

- auxiliary-loss-free balancing
- speculative decoding
- model-level routing changes

## Implementation Hook

Use these reports to justify `shared_expert_split` and separate sparse-versus-dense routed-expert kernels.
