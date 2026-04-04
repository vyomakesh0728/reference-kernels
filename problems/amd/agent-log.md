# MXFP4 Optimization Log

## 2026-03-29

- **v134** — Bucket 5 (A-pack fused into m16 constant-body)
  - Shapes: m16 (primary), portfolio-wide
  - Geomean: **22.8445 → 34.8514 us**
  - Outcome: **Discarded** (m16 exploded to 460 us; per-N re-quant duplication)
  - Note: Fusing A-pack without reuse across N tiles is catastrophic.

- **v135** — Bucket 4 (m256 public constant-body)
  - Shapes: m256 (primary), portfolio-wide
  - Geomean: **22.8445 → 22.5921 us**
  - Outcome: **Kept** (new base)
  - Note: m256 improved to 26.0 us with no regressions >0.1 us.

- **v136** — Buckets 1+2 (m64 raw-scale-shuffled, delete unshuffle + b_scale temp)
  - Shapes: m64 (primary), portfolio-wide
  - Geomean: **22.5921 → 22.5433 us**
  - Outcome: **Discarded** (gain <0.1 us; m64 regressed to 29.9 us)
  - Note: Does not meet ≥0.2 us gate; skip raw-scale-shuffled path.

- **v137** — Bucket 4 (m32 k512 inner-loop address trim)
  - Shapes: m32 (primary), portfolio-wide
  - Geomean: **22.5921 → 22.5754 us**
  - Outcome: **Discarded** (gain <0.05 us; m16 regressed to 36.4 us)
  - Note: Address math trim not portfolio-positive; revert to v135.

- **A-pack paper gate** — Duplication-law analysis (no code)
  - Shapes: m16/m32/m64/m256
  - Outcome: **Blocked** (no legal CTA-cluster handoff in HIP; only global-temp or grid-coop options)
  - Note: Macro-cluster reuse needs S≥16 (N32 tiles) for m32/m256 and S≥22 for m64; m16 (N16 tiles) needs S≥17 for 2× gate. Without CTA cluster + shared staging, reuse cannot be implemented legally.

## 2026-03-30

- **native_scaled_exact_shape_m4_fixedcost_t6** — Bucket 4 (m4 k512 constant-body scale remap hoist)
  - Shapes: m4 (primary), portfolio-wide
  - Geomean: **22.5921 → 22.3918 us**
  - Outcome: **Kept** (m4 improved to 16.1 us, no regressions >0.1 us)
  - Note: Precomputed shuffled-scale indices once per thread; removed per-step remap in the 4-step m4 loop.

- **native_scaled_exact_shape_m16_scaleaddr_t7** — Bucket 4 (m16 k7168 scale remap fast-path)
  - Shapes: m16 (primary), portfolio-wide
  - Geomean: **22.3918 → 21.2289 us**
  - Outcome: **Kept** (m16 improved to 26.7 us; m4/m32/m64 steady)
  - Note: Hoisted base_const and used constant-224 div/mod per step to cut 56x scalar remap overhead.
