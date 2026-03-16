# HipKittens Bootstrap For `mxfp4_mm`

This repo can use HipKittens as a local kernel-engineering backend, but with one important caveat:

- HipKittens mainline currently exposes scaled-MFMA wrappers and strong scheduling primitives.
- Our target problem (`mxfp4_mm`) still requires a custom MXFP4 operand/feed contract (packing + scale semantics) that must be preserved.

So the fastest path is:

1. Use HipKittens to stabilize instruction shape + schedule structure.
2. Keep our proven `mxfp4_mm` packing/contract logic and adapt it into that structure.
3. Validate only on remote MI355X (no local MI355X benchmark path here).

## Local Setup

From repo root:

```bash
bash agent_loop/scripts/setup_hipkittens_local.sh
```

Optional update:

```bash
bash agent_loop/scripts/setup_hipkittens_local.sh --update
```

This sets up:

- `third_party/HipKittens`
- `HIPKITTENS_ROOT`
- `HIPKITTENS_INCLUDE`

## Integration Strategy For `mxfp4_mm`

Start from your direct scaled probe (not the stable trunk):

- `.agent-loop/manual/native_scaled_m64_direct_chunk1_probe_v1/submission.py`

Then:

1. Add HipKittens include path to `load_inline(... extra_cuda_cflags=[...])`.
2. Mirror HK/AMD intrinsic wrapper style from:
   - `third_party/HipKittens/include/ops/warp/register/tile/mma.cuh`
3. Keep our existing A/B packing and scale inputs; do not assume HK default scale behavior matches MXFP4 contract.
4. Probe in strict order:
   - `m=32` -> `m=64` (2x32 chunks) -> `m=256`
5. Only when direct path is fault-free and correct, move to broader performance tuning.

## Guardrails

- Do not modify `fp8-mm/hip_phase2_working.py` until direct `32x32x64` path is healthy.
- Avoid early occupancy/ping-pong campaigns before operand contract is stable.
- Treat HipKittens as a bring-up accelerator for contract/scheduling, not a drop-in replacement for MXFP4 semantics.
