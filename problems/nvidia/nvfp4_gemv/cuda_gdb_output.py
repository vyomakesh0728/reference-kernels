cuda-gdb --args python3 test_kernel_only.py
NVIDIA (R) cuda-gdb 12.8
Portions Copyright (C) 2007-2024 NVIDIA Corporation
Based on GNU gdb 13.2
Copyright (C) 2023 Free Software Foundation, Inc.
License GPLv3+: GNU GPL version 3 or later <http://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
Type "show copying" and "show warranty" for details.
This CUDA-GDB was configured as "x86_64-pc-linux-gnu".
Type "show configuration" for configuration details.
For bug reporting instructions, please see:
<https://forums.developer.nvidia.com/c/developer-tools/cuda-developer-tools/cuda-gdb>.
Find the CUDA-GDB manual and other documentation resources online at:
    <https://docs.nvidia.com/cuda/cuda-gdb/index.html>.

For help, type "help".
Type "apropos word" to search for commands related to "word"...
Reading symbols from python3...
(No debugging symbols found in python3)
(cuda-gdb) run
Starting program: /root/reference-kernels/problems/nvidia/nvfp4_gemv/.venv/bin/python3 test_kernel_only.py
[Thread debugging using libthread_db enabled]
Using host libthread_db library "/lib/x86_64-linux-gnu/libthread_db.so.1".
[New Thread 0x7ffeaa9ff640 (LWP 23881)]
[New Thread 0x7ffeaa1fe640 (LWP 23882)]
[New Thread 0x7ffea99fd640 (LWP 23883)]
[New Thread 0x7ffea91fc640 (LWP 23884)]
[New Thread 0x7ffea89fb640 (LWP 23885)]
[New Thread 0x7ffea81fa640 (LWP 23886)]
[New Thread 0x7ffea79f9640 (LWP 23887)]
[New Thread 0x7ffea71f8640 (LWP 23888)]
[New Thread 0x7ffea69f7640 (LWP 23889)]
[New Thread 0x7ffea61f6640 (LWP 23890)]
[New Thread 0x7ffea59f5640 (LWP 23891)]
[New Thread 0x7ffea51f4640 (LWP 23892)]
[New Thread 0x7ffea49f3640 (LWP 23893)]
[New Thread 0x7ffea41f2640 (LWP 23894)]
[New Thread 0x7ffea39f1640 (LWP 23895)]
[New Thread 0x7ffea31f0640 (LWP 23896)]
[New Thread 0x7ffea29ef640 (LWP 23897)]
[New Thread 0x7ffea21ee640 (LWP 23898)]
[New Thread 0x7ffea19ed640 (LWP 23899)]
[New Thread 0x7ffea11ec640 (LWP 23900)]
[New Thread 0x7ffea09eb640 (LWP 23901)]
[New Thread 0x7ffea01ea640 (LWP 23902)]
[New Thread 0x7ffe9f9e9640 (LWP 23903)]
[New Thread 0x7ffe9f1e8640 (LWP 23904)]
[New Thread 0x7ffe9e9e7640 (LWP 23905)]
[New Thread 0x7ffe9e1e6640 (LWP 23906)]
[New Thread 0x7ffe9d9e5640 (LWP 23907)]
[New Thread 0x7ffe9d1e4640 (LWP 23908)]
[New Thread 0x7ffe9c9e3640 (LWP 23909)]
[New Thread 0x7ffe995dc640 (LWP 23911)]
[New Thread 0x7ffe93fff640 (LWP 23912)]
[Thread 0x7ffeaa9ff640 (LWP 23881) exited]
[Thread 0x7ffeaa1fe640 (LWP 23882) exited]
[Thread 0x7ffea89fb640 (LWP 23885) exited]
[Thread 0x7ffea91fc640 (LWP 23884) exited]
[Thread 0x7ffea99fd640 (LWP 23883) exited]
[Thread 0x7ffea71f8640 (LWP 23888) exited]
[Thread 0x7ffea79f9640 (LWP 23887) exited]
[Thread 0x7ffea29ef640 (LWP 23897) exited]
[Thread 0x7ffea01ea640 (LWP 23902) exited]
[Thread 0x7ffea41f2640 (LWP 23894) exited]
[Thread 0x7ffea49f3640 (LWP 23893) exited]
[Thread 0x7ffe9f9e9640 (LWP 23903) exited]
[Thread 0x7ffe9d9e5640 (LWP 23907) exited]
[Thread 0x7ffea09eb640 (LWP 23901) exited]
[Thread 0x7ffe9d1e4640 (LWP 23908) exited]
[Thread 0x7ffe9e1e6640 (LWP 23906) exited]
[Thread 0x7ffe9e9e7640 (LWP 23905) exited]
[Thread 0x7ffe9f1e8640 (LWP 23904) exited]
[Thread 0x7ffea11ec640 (LWP 23900) exited]
[Thread 0x7ffea19ed640 (LWP 23899) exited]
[Thread 0x7ffea21ee640 (LWP 23898) exited]
[Thread 0x7ffea39f1640 (LWP 23895) exited]
[Thread 0x7ffea51f4640 (LWP 23892) exited]
[Thread 0x7ffea59f5640 (LWP 23891) exited]
[Thread 0x7ffea61f6640 (LWP 23890) exited]
[Thread 0x7ffea69f7640 (LWP 23889) exited]
[Thread 0x7ffea81fa640 (LWP 23886) exited]
[Thread 0x7ffea31f0640 (LWP 23896) exited]
[Thread 0x7ffe9c9e3640 (LWP 23909) exited]
[Detaching after fork from child process 23913]
Testing kernel execution for all configurations...
============================================================

Warmup run...
[New Thread 0x7ffe9c9e3640 (LWP 23921)]

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
[Detaching after vfork from child process 23923]
[Detaching after vfork from child process 23924]
[Detaching after vfork from child process 23925]
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
  Mean: 23736832.30 ns (23736.83 μs, 23.737 ms)
  Std:  4499.28 ns (4.50 μs)
  Err:  2597.66 ns (2.60 μs)
  Best: 23731744.77 ns (23731.74 μs, 23.732 ms)
  Worst: 23740287.78 ns (23740.29 μs, 23.740 ms)
  Relative error: 0.011%
  📊 vs Speed of Light: 23736.83 μs / 8.622 μs = 2753.05x slower
     Best: 23731.74 μs / 8.622 μs = 2752.46x slower

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
  kernel_ptr=0x7ffea80a8cd0
  grid=(32,8,1)
  block=(320,1,1)
  shared_bytes=189440
✓ Rank-3 L4, L8 kernel launched successfully!
^C
Thread 1 "python3" received signal SIGINT, Interrupt.
[Switching focus to CUDA kernel 0, grid 161, cluster (0,0,0), block (0,0,0), thread (0,0,0), device 0, sm 142, warp 2, lane 0]
0x00007ffea15f9a70 in void fp4_gemv_rank3_cluster<128, 256, 320>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, CUtensorMap_st const*, CUtensorMap_st const*, CUtensorMap_st const*, CUtensorMap_st const*, __half*, int, int, int, int)<<<(32,8,1),(320,1,1)>>> ()
(cuda-gdb) info cuda thread
Unrecognized option: 'thread'.
(cuda-gdb) info cuda threads
  BlockIdx ThreadIdx To BlockIdx To ThreadIdx Count                 PC Filename  Line 
Kernel 0
*  (0,0,0)   (0,0,0)     (0,0,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
   (1,0,0)   (0,0,0)     (1,0,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
   (2,0,0)   (0,0,0)     (2,0,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
   (3,0,0)   (0,0,0)     (3,0,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
   (8,0,0)   (0,0,0)     (8,0,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
   (9,0,0)   (0,0,0)     (9,0,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (12,0,0)   (0,0,0)    (12,0,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (13,0,0)   (0,0,0)    (13,0,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (14,0,0)   (0,0,0)    (14,0,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (15,0,0)   (0,0,0)    (15,0,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (16,0,0)   (0,0,0)    (16,0,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (17,0,0)   (0,0,0)    (17,0,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (18,0,0)   (0,0,0)    (18,0,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (19,0,0)   (0,0,0)    (19,0,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (24,0,0)   (0,0,0)    (24,0,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (25,0,0)   (0,0,0)    (25,0,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
   (0,1,0)   (0,0,0)     (0,1,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
   (1,1,0)   (0,0,0)     (1,1,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
   (6,1,0)   (0,0,0)     (6,1,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
   (7,1,0)   (0,0,0)     (7,1,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (10,1,0)   (0,0,0)    (10,1,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (11,1,0)   (0,0,0)    (11,1,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (14,1,0)   (0,0,0)    (14,1,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (15,1,0)   (0,0,0)    (15,1,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (16,1,0)   (0,0,0)    (16,1,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (17,1,0)   (0,0,0)    (17,1,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (28,1,0)   (0,0,0)    (28,1,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (29,1,0)   (0,0,0)    (29,1,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
   (8,2,0)   (0,0,0)     (8,2,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
   (9,2,0)   (0,0,0)     (9,2,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (14,2,0)   (0,0,0)    (14,2,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (15,2,0)   (0,0,0)    (15,2,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (16,2,0)   (0,0,0)    (16,2,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (17,2,0)   (0,0,0)    (17,2,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (28,2,0)   (0,0,0)    (28,2,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (29,2,0)   (0,0,0)    (29,2,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (30,2,0)   (0,0,0)    (30,2,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (31,2,0)   (0,0,0)    (31,2,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
   (6,3,0)   (0,0,0)     (6,3,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
   (7,3,0)   (0,0,0)     (7,3,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (10,3,0)   (0,0,0)    (10,3,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (11,3,0)   (0,0,0)    (11,3,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (12,3,0)   (0,0,0)    (12,3,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (13,3,0)   (0,0,0)    (13,3,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (24,3,0)   (0,0,0)    (24,3,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (25,3,0)   (0,0,0)    (25,3,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
   (2,4,0)   (0,0,0)     (2,4,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
   (3,4,0)   (0,0,0)     (3,4,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
   (4,4,0)   (0,0,0)     (4,4,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
   (5,4,0)   (0,0,0)     (5,4,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (16,4,0)   (0,0,0)    (16,4,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (17,4,0)   (0,0,0)    (17,4,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (22,4,0)   (0,0,0)    (22,4,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (23,4,0)   (0,0,0)    (23,4,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (16,5,0)   (0,0,0)    (16,5,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (17,5,0)   (0,0,0)    (17,5,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
   (8,6,0)   (0,0,0)     (8,6,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
--Type <RET> for more, q to quit, c to continue without paging--c  
   (9,6,0)   (0,0,0)     (9,6,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (10,6,0)   (0,0,0)    (10,6,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (11,6,0)   (0,0,0)    (11,6,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (12,6,0)   (0,0,0)    (12,6,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (13,6,0)   (0,0,0)    (13,6,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (28,6,0)   (0,0,0)    (28,6,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (29,6,0)   (0,0,0)    (29,6,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
   (4,7,0)   (0,0,0)     (4,7,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
   (5,7,0)   (0,0,0)     (5,7,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (10,7,0)   (0,0,0)    (10,7,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (11,7,0)   (0,0,0)    (11,7,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (12,7,0)   (0,0,0)    (12,7,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (13,7,0)   (0,0,0)    (13,7,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (18,7,0)   (0,0,0)    (18,7,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (19,7,0)   (0,0,0)    (19,7,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (22,7,0)   (0,0,0)    (22,7,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (23,7,0)   (0,0,0)    (23,7,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
  (26,7,0)   (0,0,0)    (26,7,0)    (319,0,0)   320 0x00007ffea15f9a70              0 
  (27,7,0)   (0,0,0)    (27,7,0)    (319,0,0)   320 0x00007ffea15f61f0              0 
(cuda-gdb) cuda kernel block thread
kernel 0, block (0,0,0), thread (0,0,0)
(cuda-gdb) backtrace
#0  0x00007ffea15f9a70 in void fp4_gemv_rank3_cluster<128, 256, 320>(unsigned char const*, unsigned char const*, unsigned char const*, unsigned char const*, CUtensorMap_st const*, CUtensorMap_st const*, CUtensorMap_st const*, CUtensorMap_st const*, __half*, int, int, int, int)<<<(32,8,1),(320,1,1)>>> ()
(cuda-gdb) thread apply all bt

Thread 33 (Thread 0x7ffe9c9e3640 (LWP 23921) "cuda-EvtHandlr"):
#0  0x00007ffff7d6dc3f in poll () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007ffee96d3e27 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007ffee97c35f7 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007ffee96bfbb3 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007ffff7ce9ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#5  0x00007ffff7d7b8c0 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 32 (Thread 0x7ffe93fff640 (LWP 23912) "python3"):
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

Thread 31 (Thread 0x7ffe995dc640 (LWP 23911) "cuda00001400006"):
#0  0x00007ffff7d6dc3f in poll () from /lib/x86_64-linux-gnu/libc.so.6
#1  0x00007ffee96d3e27 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#2  0x00007ffee97c35f7 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#3  0x00007ffee96bfbb3 in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
#4  0x00007ffff7ce9ac3 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
#5  0x00007ffff7d7b8c0 in ?? () from /lib/x86_64-linux-gnu/libc.so.6

Thread 1 (Thread 0x7ffff7c54000 (LWP 23875) "python3"):
#0  0x00007ffee9818bae in ?? () from /lib/x86_64-linux-gnu/libcuda.so.1
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
#13 0x00007ffea80a846d in launch_fp4_gemv_optimized(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long) () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#14 0x00007ffea8098155 in void std::__invoke_impl<void, void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>(std::__invoke_other, void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor&&, at::Tensor&&, at::Tensor&&, at::Tensor&&, at::Tensor&&, long&&, long&&, long&&, long&&) () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#15 0x00007ffea8091d32 in std::__invoke_result<void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>::type std::__invoke<void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>(void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor&&, at::Tensor&&, at::Tensor&&, at::Tensor&&, at::Tensor&&, long&&, long&&, long&&, long&&) () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#16 0x00007ffea808a631 in std::invoke_result<void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>::type std::invoke<void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>(void (* const&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), at::Tensor&&, at::Tensor&&, at::Tensor&&, at::Tensor&&, at::Tensor&&, long&&, long&&, long&&, long&&) () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#17 0x00007ffea8082bc1 in torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}::operator()(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long) const () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
--Type <RET> for more, q to quit, c to continue without paging--c
#18 0x00007ffea80a2b83 in void pybind11::detail::argument_loader<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>::call_impl<void, torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}&, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, pybind11::detail::void_type>(torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}&, std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, pybind11::detail::void_type&&) && () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#19 0x00007ffea809e10c in std::enable_if<std::is_void<void>::value, pybind11::detail::void_type>::type pybind11::detail::argument_loader<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long>::call<void, pybind11::detail::void_type, torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}&>(torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}&) && () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#20 0x00007ffea8098362 in pybind11::cpp_function::initialize<torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}, void, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long, pybind11::name, pybind11::scope, pybind11::sibling, char [26]>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), void (*)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), pybind11::name const&, pybind11::scope const&, pybind11::sibling const&, char const (&) [26])::{lambda(pybind11::detail::function_call&)#3}::operator()(pybind11::detail::function_call&) const () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#21 0x00007ffea809887c in pybind11::cpp_function::initialize<torch::detail::wrap_pybind_function_impl_<void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul, false>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), std::integer_sequence<unsigned long, 0ul, 1ul, 2ul, 3ul, 4ul, 5ul, 6ul, 7ul, 8ul>, std::integral_constant<bool, false>)::{lambda(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long)#1}, void, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long, pybind11::name, pybind11::scope, pybind11::sibling, char [26]>(void (&)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), void (*)(at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, long, long, long, long), pybind11::name const&, pybind11::scope const&, pybind11::sibling const&, char const (&) [26])::{lambda(pybind11::detail::function_call&)#3}::_FUN(pybind11::detail::function_call&) () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
#22 0x00007ffea80809a2 in pybind11::cpp_function::dispatcher(_object*, _object*, _object*) () from /root/.cache/torch_extensions/py310_cu128/nvfp4_gemv_sm100_ptx/nvfp4_gemv_sm100_ptx.so
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