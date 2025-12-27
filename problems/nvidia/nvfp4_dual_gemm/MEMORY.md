
- Changed N=4096 config to 128x128 tile with 1x1 cluster and raster_along_m=True in `submission.py`; ran `python3 test_benchmark.py` and it failed due to CUDA device error (cudaGetDeviceCount error 304), so no new performance data.
- Ran `python3 test_benchmark.py` after switching N=4096 to 128x128 tile with 1x1 cluster; geom mean 27.750 us (per-case: 35.260, 31.023, 17.800, 30.458 us).
