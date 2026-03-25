# SonicMoE Digest

## Transferable Idea

SonicMoE is the best reference for tile-aware scheduling in fine-grained sparse-expert regimes.
Its strongest repo-fit lesson is that IO dominates small-expert sparse execution, so work scheduling and tile ownership matter as much as the math kernel.

## What To Borrow

- tile-aware scheduling keyed by tokens-per-expert
- IO and compute overlap
- stage-local buffering
- structural focus on sparse, fine-grained expert workloads

## What Not To Borrow Blindly

- token rounding if it changes routing semantics
- Hopper and Blackwell implementation details
- training-backward memory tricks

## Implementation Hook

Use SonicMoE to shape the `dispatch_pack_sparse256`, `hip_persistent_sparse256`, and per-regime scheduling logic.
