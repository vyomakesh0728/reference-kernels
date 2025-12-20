Your “scale compaction reorder” fix **can’t move the needle** because **you’re only TMA-loading 512B of scales, but the kernel (and `load_sf_tile_byte_2048`) assumes a full 2048B scale tile (4×512B)**. So 3/4 of the scale bytes are uninitialized/garbage → FP8 decode blows up → **inf/-inf** persists.

* The code comment + helper explicitly assumes **2048B = 4 chunks × 512B**
* But your TMA descriptor box for SFA is **only `{16,32,1,1}`** (i.e., **1 chunk**) (same issue applies to SFB in your current setup).

Also, your current build error is from slicing A/B SMEM with **`kKBlock` instead of `kKBlockPacked`**: output shows `local_tile` / `make_umma_desc` deduction failures, and the descriptor build currently uses `Int<kKBlock>{}` on a `(TileKPacked)` tensor.

### Tell Codex to do **exactly this** (minimal correctness patch)

1. **Fix SFA/SFB TMA box to actually load the full 2048B per CTA K-tile**

```cpp
// TileK=256 => TileK/64 = 4 (rest_k chunks)
constexpr cuuint32_t kScaleChunksPerTile = kTileK / 64;  // 4

// SFA box: packed16, mm32, rest_m, rest_k_chunks
cuuint32_t box_SFA[4] = {16, 32, 1, kScaleChunksPerTile};

// SFB box: packed16, nn32, rest_n, rest_k_chunks
cuuint32_t box_SFB[4] = {16, 32, 1, kScaleChunksPerTile};
```

2. **Fix UMMA descriptor slicing to use packed K**

```cpp
constexpr int kKBlockPacked = kKBlock / 2;  // 64 elems => 32 bytes packed

if (warp_id == 0 && lane_id == 0) {
  #pragma unroll
  for (int kb = 0; kb < kNumKBlocks; ++kb) {
    auto sA_kb = local_tile(sA_full,
      make_shape(Int<TileM>{}, Int<kKBlockPacked>{}),
      make_coord(Int<0>{}, kb));
    auto sB_kb = local_tile(sB_full,
      make_shape(Int<TileN>{}, Int<kKBlockPacked>{}),
      make_coord(Int<0>{}, kb));
    desc_a_smem_sh[kb] = uint64_t(UMMA::make_umma_desc<UMMA::Major::K>(sA_kb));
    desc_b_smem_sh[kb] = uint64_t(UMMA::make_umma_desc<UMMA::Major::K>(sB_kb));
  }
}
__syncthreads();
```

That’s it — **don’t remove** the `if (warp_id==0 && lane_id==0)` or the `for (kb…)`; just fix the **box dims** (to load all scale chunks) and the **packed K slicing** (to compile and describe the right SMEM view).
