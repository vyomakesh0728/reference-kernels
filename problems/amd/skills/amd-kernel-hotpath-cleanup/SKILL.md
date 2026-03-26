---
name: amd-kernel-hotpath-cleanup
description: Clean dead code, isolate fallback debt, and trim Python-side helper scaffolding around promoted AMD MI355X kernel winners in /Users/v/reference-kernels/problems/amd. Use when a `submission.py` is already correctness-green and benchmark-relevant, but still contains stale imports, duplicate helpers, fallback-only code mixed into the hot path, or served-shape helper debt that should be migrated into HIP/C++ without changing MFMA semantics.
---

# AMD Kernel Hotpath Cleanup

## Overview

Use this skill when an AMD competition kernel is already working and the job is to make the winner leaner, clearer, and cheaper without breaking correctness or purity. The default target is the benchmark-hot path first, not full repo janitorial work.

## Workflow

1. Start from the promoted or best measured candidate, not from an old anchor.
   For `mxfp4-mm`, prefer the newest promoted manual candidate under `.agent-loop/manual/`.

2. Audit before deleting.
   Run:
   `python3 /Users/v/reference-kernels/problems/amd/skills/amd-kernel-hotpath-cleanup/scripts/audit_hotpath_debt.py --submission <submission.py>`

3. Preserve one correctness fallback.
   A winner should keep exactly one correctness island for unsupported shapes until an equivalent slow path exists. Do not turn “one-way ticket” into “undefined behavior outside benchmark shapes.”

4. Separate hot path from correctness island.
   In the winner file, keep the benchmark-served routes visually and logically separate from fallback/reference helpers.
   For `mxfp4-mm`, served routes are the `m<=16` scaled-MFMA path and the `m>=32` scaled-MFMA path.

5. Delete only what is provably dead or debt-only in the current pass.
   Safe first-pass removals are:
   - unused imports
   - duplicate helper definitions
   - helper functions with no references
   - stale preshuffle helpers that are no longer used by served shapes
   - Python scratch-management glue that a new C++/HIP wrapper now owns

6. Keep MFMA semantics fixed during cleanup.
   Do not change:
   - MFMA instruction family
   - scale-byte ownership
   - lane-to-fragment contract
   - launch shape
   unless the task is explicitly a performance candidate rather than a cleanup pass.

7. For helper-to-HIP migration, move only served-shape prep first.
   For `mxfp4-mm`, move these into HIP/C++ before deeper scheduling work:
   - A pack/scale prep
   - B-scale unshuffle
   - direct-contract repack/entry glue for served shapes

8. Validate every cleanup or migration candidate in the normal loop.
   - static sanity
   - remote `test`
   - remote `benchmark`
   - `leaderboard` only for measured winners

## `mxfp4-mm` Rules

Read [references/mxfp4-hotpath-rules.md](references/mxfp4-hotpath-rules.md) before editing.

Current defaults:
- thin family uses `16x16x128`
- wide family uses `32x32x64`
- no producer/consumer, no `__builtin_amdgcn_s_barrier()`, no XCD swizzle in the promoted line
- `opus.hpp` is only for the wide typed MFMA wrapper
- `aiter` on served shapes is debt unless it is still required by the correctness island

## Resources

### scripts/

- `audit_hotpath_debt.py`
  Audit a promoted AMD kernel winner for:
  - unused imports
  - duplicate helper definitions
  - definition-only helper functions
  - fallback-only helper references
  - served-shape helper debt and remaining `aiter` usage

### references/

- [mxfp4-hotpath-rules.md](references/mxfp4-hotpath-rules.md)
  Current `mxfp4-mm` invariants and what must not change in a cleanup pass.
- [debt-checklist.md](references/debt-checklist.md)
  Checklist for dead imports, duplicate helpers, fallback islands, scratch ownership, and helper-to-HIP migration.
