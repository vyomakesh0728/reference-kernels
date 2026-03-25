# Tutel Digest

## Transferable Idea

Tutel is not the first kernel microarchitecture reference for this repo, but it is useful for execution-policy design.

## What To Borrow

- adaptive execution by expert-load regime
- dynamic workload-aware path selection
- avoiding one static execution schedule for all expert distributions

## What Not To Borrow Blindly

- distributed-first assumptions
- expert-parallel communication strategies that do not matter on the current single-GPU race

## Implementation Hook

Use Tutel to justify lane splits and regime tags, not as the primary code template.
