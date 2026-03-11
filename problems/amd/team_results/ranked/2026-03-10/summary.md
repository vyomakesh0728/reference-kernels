# Ranked Snapshot - 2026-03-10

This snapshot records the first successful MI355X ranked submissions for all three target problems.

## Current promoted submissions

| Problem | Promoted file | Candidate | Kernel style | Ranked status |
| --- | --- | --- | --- | --- |
| `mxfp4_mm` | `fp8-mm/submission.py` | `e33fce052407` | AITER `gemm_a4w4` anchor | success |
| `moe_mxfp4` | `moe/submission.py` | `de3f49e97e29` | AITER `fused_moe` anchor | success |
| `mixed_mla` | `mla-decode/submission.py` | `aaf6e3b5f4d4` | AITER fp8 MLA decode anchor | success |

## Raw ranked artifacts

- `mxfp4_mm.txt`
- `moe_mxfp4.txt`
- `mixed_mla.txt`

## Workflow links

- `mxfp4_mm`: `22904416587`
- `moe_mxfp4`: `22904418749`
- `mixed_mla`: `22904422749`

## Quick metric snapshot

- `mxfp4_mm` ranked means: `12.6`, `26.5`, `13.4`, `13.7`, `14.4`, `13.0` microseconds
- `moe_mxfp4` ranked means: `138`, `223`, `254`, `95.7`, `131`, `216`, `352` microseconds
- `mixed_mla` ranked means: `155`, `156`, `152`, `194`, `158`, `237`, `199`, `389` microseconds

## Timing note

- The observed end-to-end ranked submission wall clock was much larger than kernel execution time
- We are treating `338s` as the current ranked reference budget and `600s` as the client-side timeout budget in `agent_loop.toml`

## Kernel note

- The agent loop contains real kernel exploration code in `agent_loop/kernel_mutator.py`
- The currently promoted ranked-success submissions are not pure HIP kernels yet
- We switched to contract-faithful anchors first so the branch has valid leaderboard baselines before more aggressive low-level kernel work
