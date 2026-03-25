# ScatterMoE Digest

## Transferable Idea

ScatterMoE is the best direct reference for padding-free expert execution.
The key repo-fit lesson is to stop treating `sort -> gather -> GEMM -> scatter` as separate costs and instead build expert-local packets that feed the expert linear stages directly.

## What To Borrow

- padding-free expert packing
- minimize excessive input copies
- expert-local worklists instead of padded dense expert batches
- fused reorder plus expert-linear flow where possible

## What Not To Borrow Blindly

- any training-specific assumptions that do not survive the live inference contract
- any logic that changes expert selection or token multiplicity

## Implementation Hook

Use ScatterMoE as the design reference for the `dispatch_pack` and `stage1_core` lanes first.
