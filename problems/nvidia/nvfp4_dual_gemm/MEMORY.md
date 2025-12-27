## 2025-09-13
- Applied 3-stage AB pipeline prefetch/try-wait flow and TMA descriptor prefetch in `submission.py`; no tests run.
## 2025-09-13
- Tried per-shape configs (n256 tile, k4096, epilogue sub-tiling) and reverted to stable n128 config after failures; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 58.960 us).
## 2025-09-13
- Switched configs and epilogue variants while attempting larger tiles; restored correctness and ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 59.597 us).
## 2025-09-13
- Tried higher pipeline stages, larger tiles, and CTA-group paths; restored to 128x128 config with `num_ab_stage=3`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 58.978 us).
## 2025-09-13
- Set `num_ab_stage=2` for the 128x128 config in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 185.701 us, regression).
## 2025-09-13
- Restored `num_ab_stage=3` in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 58.964 us, back to prior baseline).
## 2025-09-13
- Added per-shape config `cta1_128x128_tmem128` (num_tmem_alloc_cols=128) for N=3072 shapes in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 58.648 us, slight improvement).
## 2025-09-13
- Mapped N=4096 shapes to `cta1_128x128_tmem128` in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 58.837 us, slight regression).
## 2025-09-13
- Added per-shape config `cta1_128x128_tmem512` for N=4096 shapes in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 58.730 us, slight improvement vs tmem128 but not vs baseline).
## 2025-09-13
- Tried `cta1_128x256` for N=4096 shapes; `python3 test_correctness.py` passed but `python3 test_benchmark.py` hit CUDA_ERROR_INVALID_VALUE on the N=4096 cases. Remapped N=4096 back to `cta1_128x128_tmem512`; reran tests (correctness pass, geom mean 58.631 us).
## 2025-09-13
- Added `cta1_128x128_tmem64` and mapped N=3072 shapes to it in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 58.710 us, no improvement).
## 2025-09-13
- Added `cta1_128x128_t256` (threads_per_cta=256) for N=4096 shapes in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 74.323 us, regression).
## 2025-09-13
- Remapped N=4096 shapes to `cta1_128x128_tmem512` and N=3072 shapes to `cta1_128x128_tmem128` in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 58.944 us).
## 2025-09-13
- Added `cta1_128x128_tmem512_ab4` (num_ab_stage=4) for N=4096 shapes in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 65.504 us, regression).
## 2025-09-13
- Remapped N=4096 shapes back to `cta1_128x128_tmem512` after ab4 regression; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 58.769 us).
## 2025-09-13
- Added `cta1_128x128_acc2` (num_acc_stage=2) for N=4096 shapes in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 58.999 us, slight regression).
## 2025-09-13
- Tried `cta1_128x128_tmem1024` for N=4096 shapes; `python3 test_correctness.py` passed but `python3 test_benchmark.py` failed due to num_columns limit (max 512). Remapped N=4096 back to `cta1_128x128_tmem512`; reran tests (correctness pass, geom mean 58.709 us).
## 2025-09-13
- Added `cta1_128x128_tmem256` for N=4096 shapes in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 58.766 us, no improvement).
## 2025-09-13
- Increased `prefetch_k_tile_cnt` to `num_ab_stage - 1` in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 56.445 us, improvement).
## 2025-09-13
- Tried `prefetch_k_tile_cnt = num_ab_stage` in `submission.py`; `python3 test_correctness.py` timed out (likely hang). Reverted to `num_ab_stage - 1`; reran tests (correctness pass, geom mean 52.992 us).
## 2025-09-13
- Mapped N=4096 shapes back to `cta1_128x128_tmem256` after ab4 regression; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 56.118 us).
## 2025-09-13
- Mapped N=3072 shapes to `cta1_128x128_tmem256` in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 56.328 us, slight regression).
## 2025-09-13
- Remapped N=3072 shapes back to `cta1_128x128_tmem128` in `submission.py`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 55.938 us).
## 2025-09-13
- Tried `cta1_128x128_tmem128_ab4` for N=3072 shapes; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 63.492 us, regression). Remapped N=3072 back to `cta1_128x128_tmem128`; reran tests (correctness pass, geom mean 55.871 us).
## 2025-09-13
- Updated mainloop to run across full CTA (removed warp0-only gating), limited TMA copies to tidx==0, and cleaned unused imports/vars in `submission.py`; tests not run.
## 2025-09-13
- Restored warp0-only mainloop and descriptor prefetch gating after full-CTA mainloop caused launch failures; tests not run yet.
## 2025-09-13
- Ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 56.294 us).
## 2025-09-13
- Replaced submission.py with persistent warp-specialized dual GEMM port from nvfp4_gemm.py; corrected loop variable naming to satisfy DSL; tests pending.
## 2025-09-13
- Added TMA-store epilogue path (smem staging + TMA store pipeline) to dual GEMM in `submission.py`, including `sC` storage and `tma_atom_c`; ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 33.468 us, improved).
## 2025-09-13
- Tried N=3072 config with 128x192 tile + swizzle=2; `python3 test_correctness.py` passed but `python3 test_benchmark.py` regressed to geom mean 39.719 us; reverted to default mapping and reran tests (correctness pass, geom mean 33.387 us).
## 2025-09-13
- Added N=3072 configs with 128x64 tiles + swizzle=2 and set num_acc_stage=2 only for N<128 tiles; updated acc TMEM offsets accordingly. Ran `python3 test_correctness.py` (pass) and `python3 test_benchmark.py` (geom mean 31.334 us; N=3072/K=4096 improved to 17.725 us).
