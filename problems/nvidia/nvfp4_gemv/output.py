## Benchmarks:
```
❌ k: 16384; l: 1; m: 7168; seed: 1111 failed testing:
mismatch found! custom implementation doesn't match reference:
Number of mismatched elements: 7118
ERROR AT (0, 0, 0): 0.0 4668.0
ERROR AT (1, 0 ...

❌ k: 7168; l: 8; m: 4096; seed: 1111 failed testing:
mismatch found! custom implementation doesn't match reference:
Number of mismatched elements: 32591
ERROR AT (0, 0, 0): 0.0 2244.0
ERROR AT (0, ...

❌ k: 2048; l: 4; m: 7168; seed: 1111 failed testing:
mismatch found! custom implementation doesn't match reference:
Number of mismatched elements: 28604
ERROR AT (0, 0, 0): 0.0 504.75
ERROR AT (0, ...
```

## Program stdout:
```
================================================================================
BEFORE PERMUTE:
  A shape: torch.Size([7168, 8192, 1]), dtype: torch.float4_e2m1fn_x2
  B shape: torch.Size([128, 8192, 1]), dtype: torch.float4_e2m1fn_x2
  C shape: torch.Size([7168, 1, 1]), dtype: torch.float16
  SFA shape: torch.Size([7168, 1024, 1]), dtype: torch.float8_e4m3fn
  SFB shape: torch.Size([128, 1024, 1]), dtype: torch.float8_e4m3fn
  M=7168, K=16384, L=1
================================================================================
DEBUG: After permute+clone, a.shape = torch.Size([1, 7168, 8192]), expected [1, 7168, 8192]
DEBUG: After permute+clone, b.shape = torch.Size([1, 128, 8192])

AFTER PERMUTE (BEFORE KERNEL):
  a_bytes shape: torch.Size([1, 7168, 8192]), dtype: torch.uint8
  b_bytes shape: torch.Size([1, 128, 8192]), dtype: torch.uint8
  c shape: torch.Size([1, 7168, 1]), dtype: torch.float16
  sfa_bytes shape: torch.Size([1, 7168, 1024]), dtype: torch.uint8
  sfb_bytes shape: torch.Size([1, 128, 1024]), dtype: torch.uint8
  Kernel params: M=7168, K=16384, L=1
================================================================================
[1/3] c++ -MMD -MF main.o.d -DTORCH_EXTENSION_NAME=nvfp4_gemv_sm100_ptx -DTORCH_API_INCLUDE_EXTENSION_H -isystem /usr/local/lib/python3.10/dist-...
[2/3] /usr/local/cuda-13.0/bin/nvcc --generate-dependencies-with-compile --dependency-output cuda.cuda.o.d -DTORCH_EXTENSION_NAME=nvfp4_gemv_sm1...
/home/runner/.cache/torch_extensions/py310_cu130/nvfp4_gemv_sm100_ptx/cuda.cu(404): warning #177-D: variable "n_col_blocks" was declared but not used
  const int n_col_blocks = K_scales / 4;
/home/runner/.cache/torch_extensions/py310_cu130/nvfp4_gemv_sm100_ptx/cuda.cu(90): warning #177-D: variable "n_col_blocks" was declared but never used
  const int n_col_blocks = K_scales / 4;

Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"

ptxas info    : Overriding maximum register limit 256 for '_Z18fp4_gemv_naive_onePKhS0_S0_S0_P6__halfiii' with 128 of maxrregcount option
ptxas warning : Local memory used for function '_Z22fp4_gemv_sm100_ptx_mmaILi64ELi128ELi256EEvPKhS1_S1_S1_P6__halfiii', size of stack frame: 64 bytes
ptxas info    : 311 bytes gmem
ptxas info    : Compiling entry function '_Z22fp4_gemv_sm100_ptx_mmaILi64ELi128ELi256EEvPKhS1_S1_S1_P6__halfiii' for 'sm_100'
ptxas info    : Function properties for _Z22fp4_gemv_sm100_ptx_mmaILi64ELi128ELi256EEvPKhS1_S1_S1_P6__halfiii
    64 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 48 registers, used 1 barriers, 64 bytes cumulative stack size, 19456 bytes smem
ptxas info    : Compile time = 21.139 ms
ptxas info    : Compiling entry function '_Z18fp4_gemv_naive_onePKhS0_S0_S0_P6__halfiii' for 'sm_100'
ptxas info    : Function properties for _Z18fp4_gemv_naive_onePKhS0_S0_S0_P6__halfiii
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 36 registers, used 0 barriers
ptxas info    : Compile time = 7.178 ms
[3/3] c++ main.o cuda.cuda.o -shared -lcuda -L/usr/local/lib/python3.10/dist-packages/torch/lib -lc10 -lc10_cuda -ltorch_cpu -ltorch_cuda -l...

[KERNEL A] batch=0 m=1 k=0,1   | packed=0x00 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=1 k=2,3   | packed=0x02 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=1 k=4,5   | packed=0x01 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=1 k=6,7   | packed=0x00 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=1 k=8,9   | packed=0x00 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=1 k=10,11 | packed=0x02 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=1 k=12,13 | packed=0x02 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=1 k=14,15 | packed=0x02 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=1 k=16,17 | packed=0x01 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,0.500000)
[KERNEL A] batch=0 m=1 k=18,19 | packed=0x01 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,0.500000)
[KERNEL A] batch=0 m=1 k=20,21 | packed=0x03 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,1.500000) scaled=(0.000000,1.500000)
[KERNEL A] batch=0 m=1 k=22,23 | packed=0x02 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,1.000000)
[KERNEL A] batch=0 m=1 k=24,25 | packed=0x00 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=1 k=26,27 | packed=0x00 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=1 k=28,29 | packed=0x00 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=1 k=30,31 | packed=0x02 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,1.000000)

[KERNEL A] batch=0 m=2 k=0,1   | packed=0x02 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=2 k=2,3   | packed=0x00 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=2 k=4,5   | packed=0x03 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,1.500000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=2 k=6,7   | packed=0x03 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,1.500000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=2 k=8,9   | packed=0x00 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=2 k=10,11 | packed=0x02 scale_byte=0.000000 scale=0.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=2 k=12,13 | packed=0x03 scale_byte=0.000000 scale=0.000000 | fp4_raw=(0.000000,1.500000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=2 k=14,15 | packed=0x02 scale_byte=0.000000 scale=0.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=2 k=16,17 | packed=0x01 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,1.000000)
[KERNEL A] batch=0 m=2 k=18,19 | packed=0x00 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=2 k=20,21 | packed=0x01 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,1.000000)
[KERNEL A] batch=0 m=2 k=22,23 | packed=0x02 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,2.000000)
[KERNEL A] batch=0 m=2 k=24,25 | packed=0x02 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,2.000000)
[KERNEL A] batch=0 m=2 k=26,27 | packed=0x00 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=2 k=28,29 | packed=0x01 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,1.000000)
[KERNEL A] batch=0 m=2 k=30,31 | packed=0x02 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,2.000000)

[KERNEL A] batch=0 m=0 k=0,1   | packed=0x03 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,1.500000) scaled=(0.000000,3.000000)
[KERNEL A] batch=0 m=0 k=2,3   | packed=0x00 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=0 k=4,5   | packed=0x00 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=0 k=6,7   | packed=0x03 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,1.500000) scaled=(0.000000,3.000000)
[KERNEL A] batch=0 m=0 k=8,9   | packed=0x01 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,1.000000)
[KERNEL A] batch=0 m=0 k=10,11 | packed=0x02 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,2.000000)
[KERNEL A] batch=0 m=0 k=12,13 | packed=0x01 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,1.000000)
[KERNEL A] batch=0 m=0 k=14,15 | packed=0x03 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,1.500000) scaled=(0.000000,3.000000)
[KERNEL A] batch=0 m=0 k=16,17 | packed=0x00 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=0 k=18,19 | packed=0x03 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,1.500000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=0 k=20,21 | packed=0x03 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,1.500000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=0 k=22,23 | packed=0x01 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=0 k=24,25 | packed=0x00 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=0 k=26,27 | packed=0x03 scale_byte=0x00 scale=0.000000 | fp4_raw=(0.000000,1.500000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=0 k=28,29 | packed=0x02 scale_byte=0.000000 scale=0.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,0.000000)
[KERNEL A] batch=0 m=0 k=30,31 | packed=0x02 scale_byte=0.000000 scale=0.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,0.000000)

[KERNEL B] batch=0 k=0,1   | packed=0x03 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,1.500000) scaled=(0.000000,3.000000)
[KERNEL B] batch=0 k=2,3   | packed=0x01 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,1.000000)
[KERNEL B] batch=0 k=4,5   | packed=0x01 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,1.000000)
[KERNEL B] batch=0 k=6,7   | packed=0x00 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL B] batch=0 k=8,9   | packed=0x01 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,1.000000)
[KERNEL B] batch=0 k=10,11 | packed=0x01 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,1.000000)
[KERNEL B] batch=0 k=12,13 | packed=0x01 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,1.000000)
[KERNEL B] batch=0 k=14,15 | packed=0x01 scale_byte=0x40 scale=2.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,1.000000)
[KERNEL B] batch=0 k=16,17 | packed=0x02 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,1.000000) scaled=(0.000000,1.000000)
[KERNEL B] batch=0 k=18,19 | packed=0x03 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,1.500000) scaled=(0.000000,1.500000)
[KERNEL B] batch=0 k=20,21 | packed=0x00 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL B] batch=0 k=22,23 | packed=0x03 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,1.500000) scaled=(0.000000,1.500000)
[KERNEL B] batch=0 k=24,25 | packed=0x01 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,0.500000) scaled=(0.000000,0.500000)
[KERNEL B] batch=0 k=26,27 | packed=0x00 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL B] batch=0 k=28,29 | packed=0x00 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)
[KERNEL B] batch=0 k=30,31 | packed=0x00 scale_byte=0x38 scale=1.000000 | fp4_raw=(0.000000,0.000000) scaled=(0.000000,0.000000)

[... 6587 lines omitted]
```
'`````
