# HIP Thin-Path Entry Breakthrough - 2026-03-19

This snapshot records the next `mxfp4_mm` win after the wide-path `v45` breakthrough.

## Promoted Candidate

- Promoted submission: `.agent-loop/manual/native_scaled_m4m8_direct_entry_v52/submission.py`
- Problem: `mxfp4_mm`
- Strategy:
  - keep the working `v49`/`v45` scaled-MFMA contracts intact
  - keep the wide `32x32x64` path stable
  - reduce thin-family fixed overhead by giving `m=4/8` a direct compiled entry path into the existing `16x16x128` family

## What Passed

- `test` mode: success on MI355X
- `benchmark` mode: success on MI355X
- `leaderboard` mode: success on MI355X

## Workflow Links

- `v49 test`: `23310415478`
- `v49 benchmark`: `23310517625`
- `v49 leaderboard`: `23310673433`
- `v51 test`: `23310741183`
- `v51 benchmark`: `23310841505`
- `v52 test`: `23311020763`
- `v52 benchmark`: `23311144749`
- `v52 leaderboard`: `23311337718`
- `v50 wide test`: `23311020487`
- `v50 wide benchmark`: `23311338365`

## Results

- previous promoted benchmark (`v45`): `39.355 us`
- previous promoted ranked score (`v45`): `40.093 us`
- intermediate thin direct-B win (`v49`) benchmark: `32.715 us`
- intermediate thin direct-B win (`v49`) ranked score: `33.597 us`
- micro-refinement (`v51`) benchmark: `32.707 us`
- new promoted benchmark (`v52`): `32.291 us`
- new promoted ranked score (`v52`): `33.138 us`

Improvement from `v45` to `v52`:
- benchmark: about `18.0%`
- ranked: about `17.3%`

Per-shape means from the winning `v52` cluster benchmark:
- `m=4, n=2880, k=512`: `18.9 us`
- `m=16, n=2112, k=7168`: `47.1 us`
- `m=32, n=4096, k=512`: `26.0 us`
- `m=32, n=2880, k=512`: `24.6 us`
- `m=64, n=7168, k=2048`: `54.7 us`
- `m=256, n=3072, k=1536`: `36.4 us`

Per-shape means from the `v52` ranked run:
- `m=4, n=2880, k=512`: `19.8 us`
- `m=16, n=2112, k=7168`: `47.2 us`
- `m=32, n=4096, k=512`: `26.8 us`
- `m=32, n=2880, k=512`: `25.4 us`
- `m=64, n=7168, k=2048`: `55.8 us`
- `m=256, n=3072, k=1536`: `37.3 us`

## What Changed

### `v49`: thin direct-B fragment path

- Candidate: `.agent-loop/manual/native_scaled_m16_bfrag_direct_v49/submission.py`
- Change:
  - for the `16x16x128` family, stop paying the generic thin B repack/rebuild path
  - use row-major `b_q` bytes directly in the thin kernel while keeping row-major unshuffled `b_scale`
  - leave the wide `32x32x64` path unchanged
- Result:
  - benchmark dropped from `39.355 us` to `32.715 us`
  - biggest win was `m16`: `108 us -> 46.8 us`
  - `m4` also improved: `27.2 us -> 20.7 us`

### `v51`: thin `m16` scale/address pointer hoist

- Candidate: `.agent-loop/manual/native_scaled_m16_bfrag_direct_scaleptr_v51/submission.py`
- Change:
  - clone the exact `m16` kernel/wrapper only
  - preserve the `v49` byte-ingress and scale-byte contract exactly
  - hoist and linearize A/B scale pointer math in the hot loop
  - keep `m4/8` on the original `v49` thin launcher
- Result:
  - benchmark moved slightly to `32.707 us`
  - this was a real but tiny refinement, not a new rung

### `v52`: thin `m4/m8` direct compiled entry

- Candidate: `.agent-loop/manual/native_scaled_m4m8_direct_entry_v52/submission.py`
- Change:
  - keep `v49` thin MFMA math and direct-B contract intact
  - give `m4/m8` a dedicated compiled entry path that owns A-pack and B-scale unshuffle locally
  - reduce `m4/m8` entry overhead without touching `m16` or the wide family
- Result:
  - benchmark improved to `32.291 us`
  - primary gain was `m4`: `20.7 us -> 18.9 us`
  - `m16` and wide shapes stayed roughly flat

## What Regressed

- `v50` wide A-fragment direct repack:
  - candidate: `.agent-loop/manual/native_scaled_m32_afrag_direct_v50/submission.py`
  - status: correctness-green, benchmark-green, but slower at `38.987 us`
  - main damage:
    - `m32 4096`: `26.0 us -> 34.6 us`
    - `m32 2880`: `24.5 us -> 33.4 us`
    - `m64`: `54.7 us -> 66.0 us`
    - `m256`: `36.3 us -> 46.9 us`
- Conclusion:
  - wide-path A direct fragment repack, as implemented here, is not a keep
  - the next wide step should not reuse this layout directly

## Current Regime Map

- `m=4,8`: native scaled-MFMA on the `16x16x128` family through the new `v52` direct entry
- `m=16`: native scaled-MFMA on the `16x16x128` family through the `v49` thin direct-B path, with `v51` showing a small bookkeeping refinement
- `m>=32`: native scaled-MFMA on the direct `32x32x64` family from the `v45` wide direct-B breakthrough
- no pointer-keyed cache, replay, or cross-call output/intermediate reuse is used in the promoted path

## What We Learned

- The next big win after `v45` came from the thin family, not the wide family.
- For these decode-style small-`M` shapes, reducing entry/contract overhead still matters a lot after the MFMA contract is correct.
- `v49` proved that thin B reconstruction was still costing too much.
- `v52` then showed `m4/m8` still had a separate entry-overhead lever even after `v49`.
- The attempted wide A direct-fragment repack regressed badly, so wide ownership/layout work still needs a narrower or different hypothesis.

## Next Direction

- Keep `v52` as the new promoted base.
- Continue thin and wide lanes separately.

Thin lane next:
- look for one more isolated `m16` or `m4/m8` overhead cut that preserves the `v49/v52` direct-B contract
- prioritize fixes that reduce launch/setup/address work without reopening the scale/feed semantics

Wide lane next:
- keep the `32x32x64` compute core stable
- avoid the failed `v50` A-fragment repack layout
- revisit wide data movement with a narrower ownership/coalescing hypothesis before any CDNA4 scheduling ladder work
