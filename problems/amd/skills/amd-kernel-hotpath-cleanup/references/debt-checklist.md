# Debt Checklist

Use this checklist before and after each cleanup pass.

## Imports

- Remove imports that have no live references in the winner.
- Flag imports used only by dead helpers.
- Keep imports used by the correctness island until that island is replaced.

## Helper Definitions

- Collapse duplicated helper definitions to one copy.
- Delete definition-only helpers.
- Flag helpers referenced only by fallback code.
- Flag helpers referenced only by served-shape code.

## Hot Path vs Correctness Island

- Keep served-shape branches grouped together.
- Keep fallback/reference code grouped together.
- Do not let benchmark-hot branches depend on fallback-only helpers unless unavoidable.

## Scratch Ownership

- Prefer C++/HIP ownership of served-shape scratch tensors.
- Remove `_MFMA_SCALE_INFLIGHT` only after the relevant served path owns its scratch inside HIP/C++.
- Do not remove fallback safety just to shrink Python code.

## `mxfp4-mm` Specific

- Thin path:
  - preserve `16x16x128`
  - preserve direct-B contract
- Wide path:
  - preserve `32x32x64`
  - preserve direct-B fragment ABI
- Do not introduce producer/consumer, barriers, XCD swizzle, or LDS swizzle in cleanup-only passes.
