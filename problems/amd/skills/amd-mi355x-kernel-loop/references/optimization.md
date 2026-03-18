# MI355X Optimization Landscape

Use this file after the problem snapshot and before broad optimization. It is the compact playbook for deciding which lever matters on AMD MI355X / CDNA4.

## First Principles

For these competition kernels, speed usually comes from a sequence:

1. get the contract correct
2. remove host/Python/materialization overhead
3. fix kernel-side data movement
4. only then tune deeper scheduling

Do not start with occupancy folklore or square-GEMM cargo cults. The right lever depends on shape, layout, and what the hot path is actually doing.

## Minimal Shape Diagnosis

Read `task.yml` and bucket the shapes before proposing changes.

### Thin / decode-style regimes

Typical signs:
- small `M`
- moderate or large `N`
- moderate or large `K`
- many calls where launch/setup cost can dominate

What usually matters most:
- launch count
- host/orchestration overhead
- contract materialization
- addressing math
- unnecessary temp allocation
- extra repack/unshuffle kernels

What usually matters less at first:
- deep double buffering
- aggressive multi-wave scheduling
- heroic occupancy tuning

### Wide / throughput regimes

Typical signs:
- `M >= 32` or larger wave-friendly tiles
- enough work per launch that MFMA throughput matters

What usually matters most:
- feeding the matrix core correctly and cheaply
- ownership/layout mapping
- coalesced and aligned loads
- vectorized ingress
- LDS traffic shape and bank behavior
- avoiding redundant contract repair in the hot path

What usually comes later:
- shared staging
- double buffering
- ping-pong
- multi-wave overlap

## Core Metrics To Reason About

You do not need perfect counters to ask the right questions.

### FLOPs

For GEMM-like work:

```text
FLOPs ~= 2 * M * N * K
```

This is the ceiling-side numerator for arithmetic intensity and rough TFLOP/s.

### Bytes moved

Start with the bytes that must exist for one call:

- A bytes
- B bytes
- scale bytes
- output bytes

Then add the real overhead bytes:

- repacked copies
- unshuffled scale buffers
- preshuffle intermediates
- temporary reconstructed references
- extra global or LDS staging passes

If performance is poor and the “extra bytes” dwarf the true input/output bytes, fix materialization first.

### Arithmetic intensity

```text
arithmetic intensity = FLOPs / bytes moved
```

Use it as a direction signal:

- low intensity: bandwidth or overhead dominated
- high intensity but poor speed: hot-loop feed, issue behavior, or scheduling is likely the problem

For small-`M` decode-style shapes, the kernel can still behave like a fixed-overhead problem even if the theoretical arithmetic intensity looks respectable.

### Effective throughput

```text
effective TFLOP/s = FLOPs / time
effective bandwidth = bytes moved / time
```

These do not need to be exact to be useful. They help answer:

- are we obviously launch/overhead limited?
- are we using any meaningful fraction of hardware math?
- are we paying too many bytes for the work?

## Launch Geometry And Resource Shape

### Launch dims

Always ask:

- what does one workgroup own?
- what does one wave own?
- what does one lane own?
- does that ownership match the physical layout of the hot loads?

Bad launch geometry often shows up as:

- strided loads on the wrong axis
- many scalar address calculations
- low useful work per launch
- extra boundary masking everywhere

### Threads / waves

Use wave ownership as a data-layout question, not just a “more waves” question.

Ask:

- are adjacent lanes reading adjacent bytes?
- are lanes reconstructing bytes/nibbles/scales that could be prepacked once?
- are different waves duplicating the same B-side or scale-side work?

### Occupancy

Do not treat occupancy as a magic target.

Occupancy is a tradeoff among:

- VGPR usage
- SGPR usage
- LDS usage
- workgroup size
- outstanding latency to hide

Low occupancy is only bad if it starves the kernel. High occupancy is not automatically good if it forces scalarized loads, extra address math, or register spilling.

Without real hardware counters, treat occupancy as a hypothesis, not a proven bottleneck.

## Memory Hierarchy

### Coalesced memory access

This is usually the first kernel-side speed lever after contract correctness.

Ask:

- which tensor axis is physically contiguous?
- are adjacent threads reading adjacent addresses on that axis?
- are we forcing per-lane gather/scatter work that could be transformed into contiguous vector loads?

For many successful `mxfp4` changes, the real win came from changing what each lane loads, not from changing the MFMA instruction itself.

### Coalesced axis

Pick ownership so the naturally contiguous axis is also the axis lanes walk together.

Common failure pattern:

- correct logical tile
- wrong physical ownership
- every lane reconstructs its fragment from scattered scalar bytes

Better pattern:

- prepack once into the ABI the kernel wants
- let each lane issue contiguous, aligned loads for its fragment

### Vectorized loads and stores

Vectorization helps when:

- alignment is real
- the ABI/layout is stable
- the hot loop is doing many scalar byte or halfword loads

Vectorization hurts when:

- alignment is only assumed, not guaranteed
- it forces extra shuffle work
- it changes a working ABI to a speculative one

Safe rule:

- prefer vectorized loads that preserve an already-proven contract
- avoid vectorized loads that require inventing a new contract at the same time

### Cache utilization

Think about cache as “avoiding redundant global traffic,” not as an abstract score.

Ask:

- are we rereading B or scales that could be reused within a workgroup?
- is the data already small enough that explicit staging is overkill?
- are extra staging kernels costing more than the cache miss they avoid?

For the small shapes in this repo, naive shared staging often regresses because the synchronization and extra movement cost more than the saved reads.

### Memory bandwidth

Bandwidth matters in two ways:

1. true global bandwidth pressure
2. self-inflicted bandwidth from extra copies/materialization

The second one is often the real problem in early candidates.

Before asking “are we bandwidth bound?” ask:

- how many avoidable bytes are we creating?
- how many times do we transform the same logical tensor before launch?

### LDS / shared-memory bank behavior

Ask these questions only once global layout is sane:

- do multiple lanes hit the same bank pattern repeatedly?
- is the staging layout chosen for bank behavior or just convenience?
- is the barrier/LDS overhead actually paying for less VMEM traffic?

If a simple no-LDS or direct-fragment path already wins, do not add LDS just because it feels more GPU-like.

## MFMA-Specific Guidance

### Operand legality vs performance

Separate:

1. “does the MFMA contract work?”
2. “is the MFMA path fast?”

For `mxfp4`, the big progress came from first stabilizing the operand/feed contract, then removing the extra data movement around it.

### Direct contract consumption

Direct-contract consumption is powerful when it removes:

- row-major rebuilds
- scale unshuffle in Python
- nibble extraction/repacking inside the hot loop

It is dangerous when it assumes:

- raw `B_shuffle` is already the exact ABI
- raw `b_scale_sh` is already the exact scale layout the kernel wants

The safe pattern is:

- preserve a proven ABI
- move the transform earlier or make it cheaper
- do not change ABI and schedule and scale packing all at once

## Small-Shape Priorities

For the small or thin shapes that dominate decode-style work:

1. minimize launches
2. minimize contract materialization
3. minimize address/setup overhead
4. fuse prep only when it stays contract-safe
5. keep scheduling simple until real evidence says otherwise

This is why a seemingly “more advanced” kernel can lose badly to a simpler one if it adds:

- extra staging
- extra synchronizations
- extra wrapper logic
- extra scale packing

## Wide-Shape Priorities

For the wider `M` regimes:

1. keep the winning MFMA core stable
2. improve direct fragment/feed layout
3. coalesce loads
4. vectorize proven-aligned fragment loads
5. reduce per-lane scalar work
6. only then revisit ownership remap, shared staging, and scheduling

If a candidate regresses while changing A/B scale ABI or staging shape, first go back to the last stable contract and try a smaller hot-loop data-movement change.

## Practical Decision Tree

### If `test` is failing

- fix correctness only
- one semantic axis per run
- no scheduling experiments

### If `test` is green but performance is flat across very different shapes

- suspect fixed overhead
- suspect host/materialization work
- inspect temp allocation, repacks, unshuffles, and wrapper layers

### If wide shapes are much worse than thin shapes

- inspect fragment ownership
- inspect coalescing
- inspect bytewise hot-loop work
- inspect scale handling overhead

### If thin shapes are still bad after correctness is stable

- suspect launch count and materialization more than MFMA choice
- specialize thin family separately
- do not copy wide scheduling ideas blindly

### If a candidate changes ABI and performance at the same time

- do not trust speed until correctness is solid
- prefer reverting to the last known ABI and reapplying only the cheaper load/store idea

## What Helped In This Repo

From the `mxfp4` path through `v45`, the durable lessons were:

- compiled A-pack mattered
- compiled B-scale unshuffle mattered
- direct wide B-fragment prepack mattered
- direct scale ABI rewrite was riskier and regressed in `v47`
- vectorized hot-loop loads are worth testing only after the ABI is already stable

## What To Record For Each Candidate

Always write down:

- exact hypothesis
- which family/regime it targets
- whether it changes contract, materialization, or scheduling
- test result
- benchmark result
- per-shape deltas
- keep/discard decision

That is how the optimization landscape becomes reusable instead of turning into folklore.
