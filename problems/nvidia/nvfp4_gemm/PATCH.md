**implement**, rewrite the epilogue to mirror the `sm100_bf16_gemm.cuh` address math and lane mapping (no TMA store, direct global write) while keeping the explicit fence and the existing sync


- Keep the explicit epilogue, but make it match the `sm100_bf16_gemm.cuh` address-walk pattern exactly (same `BLOCK_N`, `STORE_BLOCK_N`, `kNumElemsPerBankGroup`, and lane/bank-group mapping), and use the same fence after each TMEM load. This is the shortest path to correctness while keeping the epilogue explicit.


- The current `tmem_addr = tmem_c + col0` is too optimistic for the MMA-partitioned accumulator layout. We should re-derive `tmem_addr` using the same `accum_stage_idx * kNumMWaves * BLOCK_N + w * BLOCK_N + s * STORE_BLOCK_N + i * kNumElemsPerBankGroup` pattern as the reference.