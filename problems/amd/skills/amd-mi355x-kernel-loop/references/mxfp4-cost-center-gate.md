# `mxfp4-mm` Cost-Center Branch Gate

## Candidate Card

Every `mxfp4_mm` branch must begin with this exact card before any code is written:

- `shape`
- `deleted_cost_center`
- `expected_upside_source`
- `why_larger_than_noise`
- `touched_symbols_or_regions`
- `forbidden_edits`
- `success_gate`

If any field is missing, the branch is rejected.

## Veto Rules

Reject the branch immediately if:

- the description sounds like cleanup, hoist, fast path, prep improvement, or cleaner version of the same path without naming a deleted bucket
- more than one exact shape is touched
- prep and scheduling are changed together
- the branch reuses a generic helper with only minor local polish
- the branch reopens prep-only `m16` or prep-only `m32`

## Allowed Cost Centers

Only these buckets qualify as “whole cost center” deletions:

- Python-side wrapper materialization and inflight retention
- generic `mxfp4_pack_a_fixed` on an exact-shape path
- generic B-pack or B-scale materialization reused across exact wide shapes
- per-iteration pointer arithmetic and scale-block recomputation inside an exact kernel body
- dead exact-path epilogue, masking, or store work

If a branch cannot map to one of these buckets, reject it.

## Profile Evidence Rule

For any branch opened after a `profile_rocprof` run:

- the Candidate Card must cite the latest `profile_summary.json` and `candidate_cards.json`
- if kernelbot returned a built-in rocPROF trace zip, that zip is the active evidence source
- do not open a branch that contradicts the latest zip-derived Candidate Card without writing down the reason first

Operational rule:

- zip-derived branch order beats intuition
- benchmark-only hunches do not reopen a shape once a stronger profile card exists

## Current Shape Ladder

- `m64`
  - `m64-C1` was the first exact body-level deletion attempt and did not survive rerun stability
  - next allowed: delete generic exact `m64` A-pack
  - after that: only if a prior exact `m64` bucket wins, try `m64`-only B-stream reuse
- `m256`
  - first: delete Python-side exact `m256` materialization and inflight retention (`v91`)
  - second: delete generic exact-wide `B` repack on the exact `m256` path by feeding raw `b_q` directly (`v92`)
  - third: delete generic exact `m256` A-pack
  - only after two direct-body plateaus: reopen CTA-order, then 4-wave, then 8-wave scheduling
- `m4`
  - first: delete generic exact `m4` A-pack
  - second: delete generic exact `m4` B-scale decode
- `m16`
  - first: delete generic exact `m16` B-scale materialization
  - second: delete generic exact `m16` A-pack
  - third: delete dead exact `m16` body work
- `m32`
  - first: delete dead exact `m32` epilogue and generic loop arithmetic
  - second: delete generic exact `m32` A-pack
- `m8`
  - first: delete shared thin compute/wrapper behavior by landing a true exact `m8` body
  - second: delete generic exact `m8` A-pack

## Remote Promotion Rule

- `test`
- `benchmark`
- rerun if within `1%`
- `leaderboard` only after two agreeing benchmark wins within `0.75%`

Do not spend a remote slot on a branch that fails the Candidate Card gate.

## Recent Bucket Outcomes

- `m64-C1`: delete per-iteration pointer arithmetic and scale-block recomputation in the exact `m64` body
  - first benchmark: promising
  - rerun: failed stability gate
  - policy update: do not keep polishing the same exact `m64` body bucket; move to a different deleted bucket or a different shape
