# Search Playbook

## Default Sequence

1. Inspect status and `knowledge.json`.
2. If one problem shows repeated identical failures, focus there first.
3. Generate one concrete semantic repair.
4. Submit through `agent_loop`.
5. Keep only real ranked improvements.
6. Once a problem plateaus, move to the next problem or reopen parallel search.

## What Good Progress Looks Like

- repeated correctness failures collapse into a passing Triton path
- passing Triton path begins to close the latency gap to the anchor
- policy/profile changes correspond to actual new kernel structure, not just renamed templates
- HIP experiments keep one language/compiler path, one measurable lever, and one keep/revert decision

## What Does Not Count

- unused Triton scaffold while the hot path stays on an anchor op
- repeating the same failing family with only cosmetic edits
- broad rewrites that change multiple semantics at once while correctness is still broken

## Plateau Rules

- If the same failure summary repeats several times, stop broad search and repair that semantic issue directly.
- If Triton passes but remains far slower than the anchor, switch to a latency-focused rewrite instead of tile churn.
- If one problem is noisy across too many axes, pause the others and learn that problem first.

## Current Team Policy

- `mxfp4_mm` is the first-focus problem until the hot path is genuinely understood.
- For `mxfp4_mm`, the current default is HIP-first on `gfx950` via `load_inline`, not a Triton detour.
- Restarting campaigns is fine because cached baseline bootstrap avoids resubmitting the same anchor source.
- Failed mutations are pruned after learning, so use `knowledge.json` and pruned summaries rather than old candidate directories as the durable memory.
