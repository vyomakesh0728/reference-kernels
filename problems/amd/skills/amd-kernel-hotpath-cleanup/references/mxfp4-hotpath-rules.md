# `mxfp4-mm` Hot-Path Rules

Use this file when cleaning or migrating the current `mxfp4-mm` winner.

## Current Invariants

- Thin served shapes:
  - `m=4/8/16`
  - scaled-MFMA family: `16x16x128`
- Wide served shapes:
  - `m>=32`
  - scaled-MFMA family: `32x32x64`
- The promoted line does **not** use:
  - producer/consumer waves
  - `__builtin_amdgcn_s_barrier()`
  - LDS swizzle
  - XCD-aware CTA mapping
  - ping-pong or multi-wave scheduling

## Contract Rules

- Do not change scale-byte ownership in a cleanup pass.
- Do not change the lane-to-fragment contract in a cleanup pass.
- Keep `opus/opus.hpp` only for the wide typed `32x32x64` wrapper until a narrower replacement is justified.
- Keep one correctness fallback path for unsupported shapes.
- Treat any `aiter` use on benchmark-served shapes as debt to remove.
- Treat `aiter` in the correctness island as temporary but legal until replaced.

## Served-Shape Targets

The first helper-to-HIP migration targets are:
- thin A pack/scale prep
- thin B-scale unshuffle
- thin direct-entry glue
- wide A pack/scale prep
- wide direct-B fragment prep
- wide B-scale unshuffle

Do not begin wide scheduling work until these are stable in HIP/C++.
