(cuda-gdb) run
The program being debugged has been started already.
Start it from the beginning? (y or n) y
Starting program: /root/reference-kernels/problems/nvidia/nvfp4_gemv/.venv/bin/python3 test_kernel_only.py
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[New Thread 0x7ffeaa9ff640 (LWP 27387)]
[New Thread 0x7ffeaa1fe640 (LWP 27388)]
[New Thread 0x7ffea99fd640 (LWP 27389)]
[New Thread 0x7ffea91fc640 (LWP 27390)]
[New Thread 0x7ffea89fb640 (LWP 27391)]
[New Thread 0x7ffea81fa640 (LWP 27392)]
[New Thread 0x7ffea79f9640 (LWP 27393)]
[New Thread 0x7ffea71f8640 (LWP 27394)]
[New Thread 0x7ffea69f7640 (LWP 27395)]
[New Thread 0x7ffea61f6640 (LWP 27396)]
[New Thread 0x7ffea59f5640 (LWP 27397)]
[New Thread 0x7ffea51f4640 (LWP 27398)]
[New Thread 0x7ffea49f3640 (LWP 27399)]
[New Thread 0x7ffea41f2640 (LWP 27400)]
[New Thread 0x7ffea39f1640 (LWP 27401)]
[New Thread 0x7ffea31f0640 (LWP 27402)]
[New Thread 0x7ffea29ef640 (LWP 27403)]
[New Thread 0x7ffea21ee640 (LWP 27404)]
[New Thread 0x7ffea19ed640 (LWP 27405)]
[New Thread 0x7ffea11ec640 (LWP 27406)]
[New Thread 0x7ffea09eb640 (LWP 27407)]
[New Thread 0x7ffea01ea640 (LWP 27408)]
[New Thread 0x7ffe9f9e9640 (LWP 27409)]
[New Thread 0x7ffe9f1e8640 (LWP 27410)]
[New Thread 0x7ffe9e9e7640 (LWP 27411)]
[New Thread 0x7ffe9e1e6640 (LWP 27412)]
[New Thread 0x7ffe9d9e5640 (LWP 27413)]
[New Thread 0x7ffe9d1e4640 (LWP 27414)]
[New Thread 0x7ffe9c9e3640 (LWP 27415)]
[New Thread 0x7ffe995dc640 (LWP 27417)]
[New Thread 0x7ffe93fff640 (LWP 27418)]
[Thread 0x7ffea19ed640 (LWP 27405) exited]
[Thread 0x7ffe9d1e4640 (LWP 27414) exited]
[Thread 0x7ffe9d9e5640 (LWP 27413) exited]
[Thread 0x7ffe9e1e6640 (LWP 27412) exited]
[Thread 0x7ffe9e9e7640 (LWP 27411) exited]
[Thread 0x7ffe9f9e9640 (LWP 27409) exited]
[Thread 0x7ffea01ea640 (LWP 27408) exited]
[Thread 0x7ffea09eb640 (LWP 27407) exited]
[Thread 0x7ffea11ec640 (LWP 27406) exited]
[Thread 0x7ffea21ee640 (LWP 27404) exited]
[Thread 0x7ffea29ef640 (LWP 27403) exited]
[Thread 0x7ffea31f0640 (LWP 27402) exited]
[Thread 0x7ffea39f1640 (LWP 27401) exited]
[Thread 0x7ffea41f2640 (LWP 27400) exited]
[Thread 0x7ffea49f3640 (LWP 27399) exited]
[Thread 0x7ffea51f4640 (LWP 27398) exited]
[Thread 0x7ffea59f5640 (LWP 27397) exited]
[Thread 0x7ffea61f6640 (LWP 27396) exited]
[Thread 0x7ffea69f7640 (LWP 27395) exited]
[Thread 0x7ffea71f8640 (LWP 27394) exited]
[Thread 0x7ffea79f9640 (LWP 27393) exited]
[Thread 0x7ffea81fa640 (LWP 27392) exited]
[Thread 0x7ffea89fb640 (LWP 27391) exited]
[Thread 0x7ffea91fc640 (LWP 27390) exited]
[Thread 0x7ffea99fd640 (LWP 27389) exited]
[Thread 0x7ffeaa1fe640 (LWP 27388) exited]
[Thread 0x7ffeaa9ff640 (LWP 27387) exited]
[Thread 0x7ffe9c9e3640 (LWP 27415) exited]
[Thread 0x7ffe9f1e8640 (LWP 27410) exited]
[Detaching after fork from child process 27419]
Testing kernel execution for all configurations...
============================================================

Warmup run...
[New Thread 0x7ffe9c9e3640 (LWP 27427)]

[SCALE DEBUG] sfa_ref_cpu shape=torch.Size([7168, 1024, 1]), device=cuda:0
[SCALE DEBUG] sfb_ref_cpu shape=torch.Size([128, 1024, 1]), device=cuda:0
[DEBUG] No padding needed, K_scales=1024
[DEBUG] a_bytes: shape=(1, 7168, 8192), stride=(58720256, 8192, 1), elem_size=1, numel=58720256, bytes=58720256, data_ptr=0x7ffc2e000000
[DEBUG] b_bytes: shape=(1, 128, 8192), stride=(1048576, 8192, 1), elem_size=1, numel=1048576, bytes=1048576, data_ptr=0x7ffc65400000
[DEBUG] sfa_bytes: shape=(1, 7168, 1024), stride=(7340032, 1024, 1), elem_size=1, numel=7340032, bytes=7340032, data_ptr=0x7ffc5a000000
[DEBUG] sfb_bytes: shape=(1, 128, 1024), stride=(131072, 1024, 1), elem_size=1, numel=131072, bytes=131072, data_ptr=0x7ffc5fb27000
[DEBUG] sfa_bytes[0,0,:8] (batch 0, row 0, first 8 scales): [64, 0, 64, 64, 0, 64, 64, 56]
[DEBUG] sfa_bytes[0,1,:8] (batch 0, row 1, first 8 scales): [0, 56, 56, 0, 64, 56, 56, 64]
[DEBUG] sfa_bytes[0,2,:8] (batch 0, row 2, first 8 scales): [0, 64, 64, 56, 0, 64, 56, 0]
[DEBUG] sfa_bytes[0,3,:8] (batch 0, row 3, first 8 scales): [64, 64, 0, 56, 56, 56, 56, 56]
[DEBUG] sfa_bytes[0,4,:8] (batch 0, row 4, first 8 scales): [56, 56, 64, 64, 0, 0, 0, 64]
[DEBUG] sfb_bytes[0,0,:8] (batch 0, row 0, first 8 scales): [64, 56, 56, 0, 64, 64, 64, 56]
[DEBUG] a_bytes range: [0x7ffc2e000000, 0x7ffc31800000) (58720256 bytes)
[DEBUG] b_bytes range: [0x7ffc65400000, 0x7ffc65500000) (1048576 bytes)
[DEBUG] sfa_bytes range: [0x7ffc5a000000, 0x7ffc5a700000) (7340032 bytes)
[DEBUG] sfb_bytes range: [0x7ffc5fb27000, 0x7ffc5fb47000) (131072 bytes)
[DEBUG] a shape=(1, 7168, 8192), stride=(58720256, 8192, 1), data_ptr=0x7ffc2e000000
[DEBUG] b shape=(1, 128, 8192), stride=(1048576, 8192, 1), data_ptr=0x7ffc65400000
✅(Python) a_bytes 128-byte alignment check passed: 0x7ffc2e000000
✅(Python) b_bytes 128-byte alignment check passed: 0x7ffc65400000
LAUNCH DEBUG: M=7168 K=16384 L=1
LAUNCH DEBUG: using fp4_gemv_rank2_cta
✓ Warmup completed

Test: M=7168, K=16384, L=1
Config: rank-2: CTA + SWIZZLE_NONE + box_k=16
Speed of Light Target: 8.622 μs

[SCALE DEBUG] sfa_ref_cpu shape=torch.Size([7168, 1024, 1]), device=cuda:0
[SCALE DEBUG] sfb_ref_cpu shape=torch.Size([128, 1024, 1]), device=cuda:0
[DEBUG] No padding needed, K_scales=1024
[DEBUG] a_bytes: shape=(1, 7168, 8192), stride=(58720256, 8192, 1), elem_size=1, numel=58720256, bytes=58720256, data_ptr=0x7ffc2a000000
[DEBUG] b_bytes: shape=(1, 128, 8192), stride=(1048576, 8192, 1), elem_size=1, numel=1048576, bytes=1048576, data_ptr=0x7ffc65500000
[DEBUG] sfa_bytes: shape=(1, 7168, 1024), stride=(7340032, 1024, 1), elem_size=1, numel=7340032, bytes=7340032, data_ptr=0x7ffc28000000
[DEBUG] sfb_bytes: shape=(1, 128, 1024), stride=(131072, 1024, 1), elem_size=1, numel=131072, bytes=131072, data_ptr=0x7ffc5fb83800
[DEBUG] sfa_bytes[0,0,:8] (batch 0, row 0, first 8 scales): [64, 0, 64, 64, 0, 64, 64, 56]
[DEBUG] sfa_bytes[0,1,:8] (batch 0, row 1, first 8 scales): [0, 56, 56, 0, 64, 56, 56, 64]
[DEBUG] sfa_bytes[0,2,:8] (batch 0, row 2, first 8 scales): [0, 64, 64, 56, 0, 64, 56, 0]
[DEBUG] sfa_bytes[0,3,:8] (batch 0, row 3, first 8 scales): [64, 64, 0, 56, 56, 56, 56, 56]
[DEBUG] sfa_bytes[0,4,:8] (batch 0, row 4, first 8 scales): [56, 56, 64, 64, 0, 0, 0, 64]
[DEBUG] sfb_bytes[0,0,:8] (batch 0, row 0, first 8 scales): [64, 56, 56, 0, 64, 64, 64, 56]
[DEBUG] a_bytes range: [0x7ffc2a000000, 0x7ffc2d800000) (58720256 bytes)
[DEBUG] b_bytes range: [0x7ffc65500000, 0x7ffc65600000) (1048576 bytes)
[DEBUG] sfa_bytes range: [0x7ffc28000000, 0x7ffc28700000) (7340032 bytes)
[DEBUG] sfb_bytes range: [0x7ffc5fb83800, 0x7ffc5fba3800) (131072 bytes)
[DEBUG] a shape=(1, 7168, 8192), stride=(58720256, 8192, 1), data_ptr=0x7ffc2a000000
[DEBUG] b shape=(1, 128, 8192), stride=(1048576, 8192, 1), data_ptr=0x7ffc65500000
✅(Python) a_bytes 128-byte alignment check passed: 0x7ffc2a000000
✅(Python) b_bytes 128-byte alignment check passed: 0x7ffc65500000
LAUNCH DEBUG: M=7168 K=16384 L=1
LAUNCH DEBUG: using fp4_gemv_rank2_cta

[SCALE DEBUG] sfa_ref_cpu shape=torch.Size([7168, 1024, 1]), device=cuda:0
[SCALE DEBUG] sfb_ref_cpu shape=torch.Size([128, 1024, 1]), device=cuda:0
[DEBUG] No padding needed, K_scales=1024
[DEBUG] a_bytes: shape=(1, 7168, 8192), stride=(58720256, 8192, 1), elem_size=1, numel=58720256, bytes=58720256, data_ptr=0x7ffc2a000000
[DEBUG] b_bytes: shape=(1, 128, 8192), stride=(1048576, 8192, 1), elem_size=1, numel=1048576, bytes=1048576, data_ptr=0x7ffc65500000
[DEBUG] sfa_bytes: shape=(1, 7168, 1024), stride=(7340032, 1024, 1), elem_size=1, numel=7340032, bytes=7340032, data_ptr=0x7ffc28000000
[DEBUG] sfb_bytes: shape=(1, 128, 1024), stride=(131072, 1024, 1), elem_size=1, numel=131072, bytes=131072, data_ptr=0x7ffc5fb83800
[DEBUG] sfa_bytes[0,0,:8] (batch 0, row 0, first 8 scales): [64, 0, 64, 64, 0, 64, 64, 56]
[DEBUG] sfa_bytes[0,1,:8] (batch 0, row 1, first 8 scales): [0, 56, 56, 0, 64, 56, 56, 64]
[DEBUG] sfa_bytes[0,2,:8] (batch 0, row 2, first 8 scales): [0, 64, 64, 56, 0, 64, 56, 0]
[DEBUG] sfa_bytes[0,3,:8] (batch 0, row 3, first 8 scales): [64, 64, 0, 56, 56, 56, 56, 56]
[DEBUG] sfa_bytes[0,4,:8] (batch 0, row 4, first 8 scales): [56, 56, 64, 64, 0, 0, 0, 64]
[DEBUG] sfb_bytes[0,0,:8] (batch 0, row 0, first 8 scales): [64, 56, 56, 0, 64, 64, 64, 56]
[DEBUG] a_bytes range: [0x7ffc2a000000, 0x7ffc2d800000) (58720256 bytes)
[DEBUG] b_bytes range: [0x7ffc65500000, 0x7ffc65600000) (1048576 bytes)
[DEBUG] sfa_bytes range: [0x7ffc28000000, 0x7ffc28700000) (7340032 bytes)
[DEBUG] sfb_bytes range: [0x7ffc5fb83800, 0x7ffc5fba3800) (131072 bytes)
[DEBUG] a shape=(1, 7168, 8192), stride=(58720256, 8192, 1), data_ptr=0x7ffc2a000000
[DEBUG] b shape=(1, 128, 8192), stride=(1048576, 8192, 1), data_ptr=0x7ffc65500000
✅(Python) a_bytes 128-byte alignment check passed: 0x7ffc2a000000
✅(Python) b_bytes 128-byte alignment check passed: 0x7ffc65500000
LAUNCH DEBUG: M=7168 K=16384 L=1
LAUNCH DEBUG: using fp4_gemv_rank2_cta

[SCALE DEBUG] sfa_ref_cpu shape=torch.Size([7168, 1024, 1]), device=cuda:0
[SCALE DEBUG] sfb_ref_cpu shape=torch.Size([128, 1024, 1]), device=cuda:0
[DEBUG] No padding needed, K_scales=1024
[DEBUG] a_bytes: shape=(1, 7168, 8192), stride=(58720256, 8192, 1), elem_size=1, numel=58720256, bytes=58720256, data_ptr=0x7ffc2a000000
[DEBUG] b_bytes: shape=(1, 128, 8192), stride=(1048576, 8192, 1), elem_size=1, numel=1048576, bytes=1048576, data_ptr=0x7ffc65500000
[DEBUG] sfa_bytes: shape=(1, 7168, 1024), stride=(7340032, 1024, 1), elem_size=1, numel=7340032, bytes=7340032, data_ptr=0x7ffc28000000
[DEBUG] sfb_bytes: shape=(1, 128, 1024), stride=(131072, 1024, 1), elem_size=1, numel=131072, bytes=131072, data_ptr=0x7ffc5fb83800
[DEBUG] sfa_bytes[0,0,:8] (batch 0, row 0, first 8 scales): [64, 0, 64, 64, 0, 64, 64, 56]
[DEBUG] sfa_bytes[0,1,:8] (batch 0, row 1, first 8 scales): [0, 56, 56, 0, 64, 56, 56, 64]
[DEBUG] sfa_bytes[0,2,:8] (batch 0, row 2, first 8 scales): [0, 64, 64, 56, 0, 64, 56, 0]
[DEBUG] sfa_bytes[0,3,:8] (batch 0, row 3, first 8 scales): [64, 64, 0, 56, 56, 56, 56, 56]
[DEBUG] sfa_bytes[0,4,:8] (batch 0, row 4, first 8 scales): [56, 56, 64, 64, 0, 0, 0, 64]
[DEBUG] sfb_bytes[0,0,:8] (batch 0, row 0, first 8 scales): [64, 56, 56, 0, 64, 64, 64, 56]
[DEBUG] a_bytes range: [0x7ffc2a000000, 0x7ffc2d800000) (58720256 bytes)
[DEBUG] b_bytes range: [0x7ffc65500000, 0x7ffc65600000) (1048576 bytes)
[DEBUG] sfa_bytes range: [0x7ffc28000000, 0x7ffc28700000) (7340032 bytes)
[DEBUG] sfb_bytes range: [0x7ffc5fb83800, 0x7ffc5fba3800) (131072 bytes)
[DEBUG] a shape=(1, 7168, 8192), stride=(58720256, 8192, 1), data_ptr=0x7ffc2a000000
[DEBUG] b shape=(1, 128, 8192), stride=(1048576, 8192, 1), data_ptr=0x7ffc65500000
✅(Python) a_bytes 128-byte alignment check passed: 0x7ffc2a000000
✅(Python) b_bytes 128-byte alignment check passed: 0x7ffc65500000
LAUNCH DEBUG: M=7168 K=16384 L=1
LAUNCH DEBUG: using fp4_gemv_rank2_cta
✓ Kernel executed successfully! Output shape: torch.Size([7168, 1, 1])
  Runs: 3
  Mean: 23734176.00 ns (23734.18 μs, 23.734 ms)
  Std:  4661.68 ns (4.66 μs)
  Err:  2691.42 ns (2.69 μs)
  Best: 23729120.25 ns (23729.12 μs, 23.729 ms)
  Worst: 23738304.14 ns (23738.30 μs, 23.738 ms)
  Relative error: 0.011%
  📊 vs Speed of Light: 23734.18 μs / 8.622 μs = 2752.75x slower
     Best: 23729.12 μs / 8.622 μs = 2752.16x slower

Test: M=4096, K=7168, L=8
Config: rank-3: Cluster + SWIZZLE_128B + box_k=K_scales_padded
Speed of Light Target: 17.275 μs

[SCALE DEBUG] sfa_ref_cpu shape=torch.Size([4096, 448, 8]), device=cuda:0
[SCALE DEBUG] sfb_ref_cpu shape=torch.Size([128, 448, 8]), device=cuda:0
[DEBUG] No padding needed, K_scales=448
[DEBUG] a_bytes: shape=(8, 4096, 3584), stride=(14680064, 3584, 1), elem_size=1, numel=117440512, bytes=117440512, data_ptr=0x7ffb27000000
[DEBUG] b_bytes: shape=(8, 128, 3584), stride=(458752, 3584, 1), elem_size=1, numel=3670016, bytes=3670016, data_ptr=0x7ffc7ee00000
[DEBUG] sfa_bytes: shape=(8, 4096, 448), stride=(1835008, 448, 1), elem_size=1, numel=14680064, bytes=14680064, data_ptr=0x7ffc64000000
[DEBUG] sfb_bytes: shape=(8, 128, 448), stride=(57344, 448, 1), elem_size=1, numel=458752, bytes=458752, data_ptr=0x7ffc5fb83800
[DEBUG] sfa_bytes[0,0,:8] (batch 0, row 0, first 8 scales): [64, 0, 0, 56, 0, 0, 56, 0]
[DEBUG] sfa_bytes[0,1,:8] (batch 0, row 1, first 8 scales): [56, 0, 56, 0, 0, 56, 0, 0]
[DEBUG] sfa_bytes[0,2,:8] (batch 0, row 2, first 8 scales): [56, 56, 56, 56, 56, 56, 0, 0]
[DEBUG] sfa_bytes[0,3,:8] (batch 0, row 3, first 8 scales): [64, 64, 64, 56, 56, 0, 56, 0]
[DEBUG] sfa_bytes[0,4,:8] (batch 0, row 4, first 8 scales): [64, 64, 64, 56, 64, 0, 56, 0]
[DEBUG] sfb_bytes[0,0,:8] (batch 0, row 0, first 8 scales): [56, 0, 64, 56, 0, 56, 64, 0]
[DEBUG] a_bytes range: [0x7ffb27000000, 0x7ffb2e000000) (117440512 bytes)
[DEBUG] b_bytes range: [0x7ffc7ee00000, 0x7ffc7f180000) (3670016 bytes)
[DEBUG] sfa_bytes range: [0x7ffc64000000, 0x7ffc64e00000) (14680064 bytes)
[DEBUG] sfb_bytes range: [0x7ffc5fb83800, 0x7ffc5fbf3800) (458752 bytes)
[DEBUG] a shape=(8, 4096, 3584), stride=(14680064, 3584, 1), data_ptr=0x7ffb27000000
[DEBUG] b shape=(8, 128, 3584), stride=(458752, 3584, 1), data_ptr=0x7ffc7ee00000
✅(Python) a_bytes 128-byte alignment check passed: 0x7ffb27000000
✅(Python) b_bytes 128-byte alignment check passed: 0x7ffc7ee00000
LAUNCH DEBUG: M=4096 K=7168 L=8
LAUNCH DEBUG: using fp4_gemv_rank3_cluster
Device limits:
  sharedMemPerBlock: 49152 (48.0 KB)
  sharedMemPerBlockOptin: 232448 (227.0 KB)
  maxBlocksPerMultiProcessor: 32
  clusterDimSupported: 1
  Requested shared_bytes: 189440 (185.0 KB)
Launch config verification:
  kernel_ptr=0x7ffea81a8cd0
  grid=(32,8,1)
  block=(320,1,1)
  shared_bytes=189440
✓ Rank-3 L4, L8 kernel launched successfully!
^C
Thread 1 "python3" received signal SIGINT, Interrupt.
[Switching focus to CUDA kernel 1, grid 161, cluster (2,0,0), block (4,0,0), thread (0,0,0), device 0, sm 146, warp 2, lane 0]
0x00007ffea15f9a00 in void fp4_gemv_rank3_cluster<128, 256, 320>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, CUtensorMap_st const*, CUtensorMap_st const*, CUtensorMap_st const*, CUtensorMap_st const*, __half*, int, int, int, int)<<<(32,8,1),(320,1,1)>>> ()
(cuda-gdb) info cuda threads       
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count                 PC Filename  Line 
Kernel 1
*  (4,0,0)   (0,0,0)     (4,0,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
   (5,0,0)   (0,0,0)     (5,0,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (10,0,0)   (0,0,0)    (10,0,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (11,0,0)   (0,0,0)    (11,0,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (12,0,0)   (0,0,0)    (12,0,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (13,0,0)   (0,0,0)    (13,0,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (14,0,0)   (0,0,0)    (14,0,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (15,0,0)   (0,0,0)    (15,0,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (16,0,0)   (0,0,0)    (16,0,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (17,0,0)   (0,0,0)    (17,0,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (22,0,0)   (0,0,0)    (22,0,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (23,0,0)   (0,0,0)    (23,0,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (24,0,0)   (0,0,0)    (24,0,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (25,0,0)   (0,0,0)    (25,0,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (26,0,0)   (0,0,0)    (26,0,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (27,0,0)   (0,0,0)    (27,0,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (12,1,0)   (0,0,0)    (12,1,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (13,1,0)   (0,0,0)    (13,1,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (14,1,0)   (0,0,0)    (14,1,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (15,1,0)   (0,0,0)    (15,1,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (10,2,0)   (0,0,0)    (10,2,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (11,2,0)   (0,0,0)    (11,2,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (14,2,0)   (0,0,0)    (14,2,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (15,2,0)   (0,0,0)    (15,2,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (16,2,0)   (0,0,0)    (16,2,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (17,2,0)   (0,0,0)    (17,2,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (26,2,0)   (0,0,0)    (26,2,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (27,2,0)   (0,0,0)    (27,2,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
   (6,3,0)   (0,0,0)     (6,3,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
   (7,3,0)   (0,0,0)     (7,3,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (12,3,0)   (0,0,0)    (12,3,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (13,3,0)   (0,0,0)    (13,3,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (22,3,0)   (0,0,0)    (22,3,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (23,3,0)   (0,0,0)    (23,3,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (24,3,0)   (0,0,0)    (24,3,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (25,3,0)   (0,0,0)    (25,3,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (26,3,0)   (0,0,0)    (26,3,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (27,3,0)   (0,0,0)    (27,3,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (24,4,0)   (0,0,0)    (24,4,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (25,4,0)   (0,0,0)    (25,4,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (26,4,0)   (0,0,0)    (26,4,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (27,4,0)   (0,0,0)    (27,4,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (28,4,0)   (0,0,0)    (28,4,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (29,4,0)   (0,0,0)    (29,4,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
   (8,5,0)   (0,0,0)     (8,5,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
   (9,5,0)   (0,0,0)     (9,5,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (10,5,0)   (0,0,0)    (10,5,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (11,5,0)   (0,0,0)    (11,5,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (12,5,0)   (0,0,0)    (12,5,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (13,5,0)   (0,0,0)    (13,5,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (14,5,0)   (0,0,0)    (14,5,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (15,5,0)   (0,0,0)    (15,5,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (16,5,0)   (0,0,0)    (16,5,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (17,5,0)   (0,0,0)    (17,5,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (20,5,0)   (0,0,0)    (20,5,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (21,5,0)   (0,0,0)    (21,5,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (30,5,0)   (0,0,0)    (30,5,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
--Type <RET> for more, q to quit, c to continue without paging--c
  (31,5,0)   (0,0,0)    (31,5,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
   (4,6,0)   (0,0,0)     (4,6,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
   (5,6,0)   (0,0,0)     (5,6,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
   (6,6,0)   (0,0,0)     (6,6,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
   (7,6,0)   (0,0,0)     (7,6,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (12,6,0)   (0,0,0)    (12,6,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (13,6,0)   (0,0,0)    (13,6,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (18,6,0)   (0,0,0)    (18,6,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (19,6,0)   (0,0,0)    (19,6,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (22,6,0)   (0,0,0)    (22,6,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (23,6,0)   (0,0,0)    (23,6,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (28,6,0)   (0,0,0)    (28,6,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (29,6,0)   (0,0,0)    (29,6,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
  (30,6,0)   (0,0,0)    (30,6,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
  (31,6,0)   (0,0,0)    (31,6,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
   (6,7,0)   (0,0,0)     (6,7,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
   (7,7,0)   (0,0,0)     (7,7,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
   (8,7,0)   (0,0,0)     (8,7,0)    (319,0,0)   320 0x00007ffea15f9a00              0 
   (9,7,0)   (0,0,0)     (9,7,0)    (319,0,0)   320 0x00007ffea15f61d0              0 
(cuda-gdb) backtrace          
#0  0x00007ffea15f9a00 in void fp4_gemv_rank3_cluster<128, 256, 320>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, CUtensorMap_st const*, CUtensorMap_st const*, CUtensorMap_st const*, CUtensorMap_st const*, __half*, int, int, int, int)<<<(32,8,1),(320,1,1)>>> ()
(cuda-gdb) cuda kernel block thread
kernel 1, block (4,0,0), thread (0,0,0)
(cuda-gdb) thread apply all bt     

Thread 33 (Thread 0x7ffe9c9e3640 (LWP 27427) "cuda-EvtHandlr"):
#0  0x00007ffff7d6dc3f in poll () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007ffee96d3e27 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007ffee97c35f7 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007ffee96bfbb3 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007ffff7ce9ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#5  0x00007ffff7d7b8c0 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 32 (Thread 0x7ffe93fff640 (LWP 27418) "python3"):
#0  0x00007ffff7d7ae9e in epoll_wait () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007ffe9883a796 in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#2  0x00007ffe98837cf9 in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#3  0x00007ffe98839d5a in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#4  0x00007ffe9884488a in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#5  0x00007ffe98844d9b in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#6  0x00007ffe987fd90d in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#7  0x00007ffe98a52dfd in ?? () from /lib/x86_64-linux-gnu/libcudadebugger.so.1
#8  0x00007ffff7ce9ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#9  0x00007ffff7d7b8c0 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 31 (Thread 0x7ffe995dc640 (LWP 27417) "cuda00001400006"):
#0  0x00007ffff7d6dc3f in poll () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007ffee96d3e27 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007ffee97c35f7 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007ffee96bfbb3 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007ffff7ce9ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#5  0x00007ffff7d7b8c0 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 1 (Thread 0x7ffff7c54000 (LWP 27384) "python3"):
#0  0x00007ffee9818b41 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#1  0x00007ffee95d6183 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007ffee9614e3b in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007ffeea354137 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007ffeea354585 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#5  0x00007ffee95e29f4 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#6  0x00007ffee9608ce2 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#7  0x00007ffee96145d5 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#8  0x00007ffee95614f9 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#9  0x00007ffee96f0f77 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#10 0x00007ffee96e1520 in cuCtxSynchronize () from /lib/x86_64-linux-gnu/libcuda.so.1
#11 0x00007ffff6e0fb8b in ?? () from /root/reference-kernels/problems/nvidia/nvfp4_gemv/.venv/lib/python3.10/site-packages/torch/lib/../../nvidia/cuda_runtime/lib/libcudart.so.12
#12 0x00007ffff6e4a43a in cudaDeviceSynchronize () from /root/reference-kernels/problems/nvidia/nvfp4_gemv/.venv/lib/python3.10/site-packages/torch/lib/../../nvidia/cuda_runtime/lib/libcudart.so.12
#13 0x00007ffea81a846d in launch_fp4_gemv_optimized(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long) () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#14 0x00007ffea8198155 in void std::__invoke_impl<void, void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>(std::__invoke_other, void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor&&, at::Tensor&&, at::Tensor&&, at::Tensor&&, at::Tensor&&, long&&, long&&, long&&, long&&) () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#15 0x00007ffea8191d32 in std::__invoke_result<void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>::type std::__invoke<void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>(void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor&&, at::Tensor&&, at::Tensor&&, at::Tensor&&, at::Tensor&&, long&&, long&&, long&&, long&&) () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#16 0x00007ffea818a631 in std::invoke_result<void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>::type std::invoke<void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>(void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor&&, at::Tensor&&, at::Tensor&&, at::Tensor&&, at::Tensor&&, long&&, long&&, long&&, long&&) () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#17 0x00007ffea8182bc1 in torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}::operator()(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long) const () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
--Type <RET> for more, q to quit, c to continue without paging--c
#18 0x00007ffea81a2b83 in void pybind11::detail::argument_loader<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>::call_impl<void, torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}&, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, pybind11::detail::void_type>(torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}&, std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, pybind11::detail::void_type&&) && () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#19 0x00007ffea819e10c in std::enable_if<std::is_void<void>::value, pybind11::detail::void_type>::type pybind11::detail::argument_loader<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>::call<void, pybind11::detail::void_type, torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}&>(torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}&) && () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#20 0x00007ffea8198362 in pybind11::cpp_function::initialize<torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}, void, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long, pybind11::name, pybind11::scope, pybind11::sibling, char [26]>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), void (*)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), pybind11::name const&, pybind11::scope const&, pybind11::sibling const&, char const (&) [26])::{lambda(pybind11::detail::function_call&)#3}::operator()(pybind11::detail::function_call&) const () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#21 0x00007ffea819887c in pybind11::cpp_function::initialize<torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}, void, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long, pybind11::name, pybind11::scope, pybind11::sibling, char [26]>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), void (*)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), pybind11::name const&, pybind11::scope const&, pybind11::sibling const&, char const (&) [26])::{lambda(pybind11::detail::function_call&)#3}::_FUN(pybind11::detail::function_call&) () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#22 0x00007ffea81809a2 in pybind11::cpp_function::dispatcher(_object*, _object*, _object*) () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#23 0x00005555556de852 in ?? ()
#24 0x00005555556d512b in _PyObject_MakeTpCall ()
#25 0x00005555556ced96 in _PyEval_EvalFrameDefault ()
#26 0x00005555556df0ac in _PyFunction_Vectorcall ()
#27 0x00005555556c9460 in _PyEval_EvalFrameDefault ()
#28 0x00005555557adbe6 in ?? ()
#29 0x00005555557adab6 in PyEval_EvalCode ()
#30 0x00005555557d4528 in ?? ()
#31 0x00005555557ceb7f in ?? ()
#32 0x00005555557d42c5 in ?? ()
#33 0x00005555557d3808 in _PyRun_SimpleFileObject ()
#34 0x00005555557d34e7 in _PyRun_AnyFileObject ()
#35 0x00005555557c7a8e in Py_RunMain ()
#36 0x00005555557a1a8d in Py_BytesMain ()
#37 0x00007ffff7c7ed90 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#38 0x00007ffff7c7ee40 in __libc_start_main () from /lib/x86_64-linux-gnu/libc.so.6
#39 0x00005555557a1985 in _start ()
(cuda-gdb) 