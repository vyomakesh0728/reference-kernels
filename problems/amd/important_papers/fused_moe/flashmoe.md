# FlashMoE Digest

## Transferable Idea

FlashMoE is valuable because it argues for a persistent fused MoE pipeline, not because its CUDA implementation is portable.

## What To Borrow

- GPU-resident scheduling
- persistent kernel mindset
- collapsing dispatch, expert compute, and combine boundaries
- keeping metadata and work queues close to the kernel instead of the CPU

## What Not To Borrow Blindly

- NVSHMEM and distributed communication details
- CUDA-only libraries and kernel templates
- treat it as a direct shared-expert implementation blueprint

## Implementation Hook

Use FlashMoE to guide the `full_pipeline` lane after partial lanes are green.
