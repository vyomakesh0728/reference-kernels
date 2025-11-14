Profiling k: 16384; l: 1; m: 7168; seed: 1111 (1/2):
    void fp4_gemv_sm100_ptx_mma<64, 128, 256>(const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, int, int, int) (112, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0

      Table Name : GPU Throughput
      ---------------- ----------- ------------
      Metric Name      Metric Unit Metric Value
      ---------------- ----------- ------------
      Memory [%]                 %         4.37
      Compute (SM) [%]           %        10.09
      ---------------- ----------- ------------

      Table Name : Pipe Utilization (% of active cycles)
      Table Description : Pipeline utilization based on the number of cycles the pipeline was active. This takes the rates of different instructions executing on the pipeline into account. For an instruction requiring 4 cycles to complete execution, the counter is increased by 1 for 4 cycles. Use this to understand the pipeline utilization for the time it was active.
      -------------------- ----------- ------------
      Metric Name          Metric Unit Metric Value
      -------------------- ----------- ------------
      ALU                            %         9.24
      FMA                            %         5.00
      Shared (FP64+Tensor)           %         0.55
      Tensor (All)                   %         0.55
      Tensor (FP)                    %         0.55
      TMEM (Tensor Memory)           %            0
      FP64                           %            0
      TC                             %            0
      Tensor (DP)                    %            0
      Tensor (INT)                   %            0
      TMA                            %            0
      -------------------- ----------- ------------

      Table Name : Warp State (All Cycles)
      ------------------------ ----------- ------------
Profiling k: 16384; l: 1; m: 7168; seed: 1111 (2/2):
      Metric Name              Metric Unit Metric Value
      ------------------------ ----------- ------------
      Stall Long Scoreboard           inst        10.24
      Stall Wait                      inst         2.09
      Stall Short Scoreboard          inst         1.23
      Selected                        inst            1
      Stall Barrier                   inst         0.88
      Stall Branch Resolving          inst         0.28
      Stall Not Selected              inst         0.07
      Stall No Instruction            inst         0.06
      Stall Math Pipe Throttle        inst         0.05
      Stall Dispatch Stall            inst         0.00
      Stall Drain                     inst         0.00
      Stall LG Throttle               inst            0
      Stall Membar                    inst            0
      Stall MIO Throttle              inst            0
      Stall Misc                      inst            0
      Stall Sleeping                  inst            0
      Stall Tex Throttle              inst            0
      ------------------------ ----------- ------------
Profiling k: 7168; l: 8; m: 4096; seed: 1111 (1/4):
    void fp4_gemv_sm100_ptx_mma<64, 128, 256>(const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, int, int, int) (64, 8, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0

      Table Name : GPU Throughput
      ---------------- ----------- ------------
      Metric Name      Metric Unit Metric Value
      ---------------- ----------- ------------
      Memory [%]                 %        19.25
      Compute (SM) [%]           %        43.08
      ---------------- ----------- ------------

      Table Name : Pipe Utilization (% of active cycles)
      Table Description : Pipeline utilization based on the number of cycles the pipeline was active. This takes the rates of different instructions executing on the pipeline into account. For an instruction requiring 4 cycles to complete execution, the counter is increased by 1 for 4 cycles. Use this to understand the pipeline utilization for the time it was active.
      -------------------- ----------- ------------
      Metric Name          Metric Unit Metric Value
      -------------------- ----------- ------------
      ALU                            %        30.34
      FMA                            %        16.40
      Shared (FP64+Tensor)           %         1.81
      Tensor (All)                   %         1.81
      Tensor (FP)                    %         1.81
      TMEM (Tensor Memory)           %            0
      FP64                           %            0
      TC                             %            0
      Tensor (DP)                    %            0
      Tensor (INT)                   %            0
      TMA                            %            0
      -------------------- ----------- ------------

      Table Name : Warp State (All Cycles)
      ------------------------ ----------- ------------
Profiling k: 7168; l: 8; m: 4096; seed: 1111 (2/4):
      Metric Name              Metric Unit Metric Value
      ------------------------ ----------- ------------
      Stall Long Scoreboard           inst         9.81
      Stall Wait                      inst         2.12
      Stall Short Scoreboard          inst         1.30
      Stall Barrier                   inst         1.19
      Selected                        inst            1
      Stall Not Selected              inst         0.45
      Stall Branch Resolving          inst         0.29
      Stall Math Pipe Throttle        inst         0.27
      Stall No Instruction            inst         0.07
      Stall Dispatch Stall            inst         0.02
      Stall Drain                     inst         0.00
      Stall LG Throttle               inst            0
      Stall Membar                    inst            0
      Stall MIO Throttle              inst         0.00
      Stall Misc                      inst         0.00
      Stall Sleeping                  inst            0
      Stall Tex Throttle              inst            0
      ------------------------ ----------- ------------

    void at::elementwise_kernel<128, 4, void at::gpu_kernel_impl_nocast<at::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() lambda() (instance 10)]::operator ()() lambda(c10::Half) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) (64, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 10.0

      Table Name : GPU Throughput
      ---------------- ----------- ------------
      Metric Name      Metric Unit Metric Value
      ---------------- ----------- ------------
      Memory [%]                 %         2.39
      Compute (SM) [%]           %         0.73
      ---------------- ----------- ------------

      Table Name : Pipe Utilization (% of active cycles)
Profiling k: 7168; l: 8; m: 4096; seed: 1111 (3/4):
      Table Description : Pipeline utilization based on the number of cycles the pipeline was active. This takes the rates of different instructions executing on the pipeline into account. For an instruction requiring 4 cycles to complete execution, the counter is increased by 1 for 4 cycles. Use this to understand the pipeline utilization for the time it was active.
      -------------------- ----------- ------------
      Metric Name          Metric Unit Metric Value
      -------------------- ----------- ------------
      ALU                            %         3.01
      FMA                            %         1.30
      TMEM (Tensor Memory)           %            0
      FP64                           %            0
      Shared (FP64+Tensor)           %            0
      TC                             %            0
      Tensor (All)                   %            0
      Tensor (DP)                    %            0
      Tensor (FP)                    %            0
      Tensor (INT)                   %            0
      TMA                            %            0
      -------------------- ----------- ------------

      Table Name : Warp State (All Cycles)
      ------------------------ ----------- ------------
      Metric Name              Metric Unit Metric Value
      ------------------------ ----------- ------------
      Stall Long Scoreboard           inst        11.74
      Stall No Instruction            inst         4.72
      Stall Short Scoreboard          inst         3.50
      Stall Wait                      inst         2.16
      Selected                        inst            1
      Stall Branch Resolving          inst         0.24
      Stall Drain                     inst         0.13
      Stall Dispatch Stall            inst         0.10
      Stall Barrier                   inst            0
Profiling k: 7168; l: 8; m: 4096; seed: 1111 (4/4):
      Stall LG Throttle               inst            0
      Stall Math Pipe Throttle        inst            0
      Stall Membar                    inst            0
      Stall MIO Throttle              inst            0
      Stall Misc                      inst            0
      Stall Not Selected              inst            0
      Stall Sleeping                  inst            0
      Stall Tex Throttle              inst            0
      ------------------------ ----------- ------------
Profiling k: 2048; l: 4; m: 7168; seed: 1111 (1/4):
    void fp4_gemv_sm100_ptx_mma<64, 128, 256>(const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, int, int, int) (112, 4, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.0

      Table Name : GPU Throughput
      ---------------- ----------- ------------
      Metric Name      Metric Unit Metric Value
      ---------------- ----------- ------------
      Memory [%]                 %        16.62
      Compute (SM) [%]           %        37.29
      ---------------- ----------- ------------

      Table Name : Pipe Utilization (% of active cycles)
      Table Description : Pipeline utilization based on the number of cycles the pipeline was active. This takes the rates of different instructions executing on the pipeline into account. For an instruction requiring 4 cycles to complete execution, the counter is increased by 1 for 4 cycles. Use this to understand the pipeline utilization for the time it was active.
      -------------------- ----------- ------------
      Metric Name          Metric Unit Metric Value
      -------------------- ----------- ------------
      ALU                            %        27.72
      FMA                            %        14.96
      Shared (FP64+Tensor)           %         1.65
      Tensor (All)                   %         1.65
      Tensor (FP)                    %         1.65
      TMEM (Tensor Memory)           %            0
      FP64                           %            0
      TC                             %            0
      Tensor (DP)                    %            0
      Tensor (INT)                   %            0
      TMA                            %            0
      -------------------- ----------- ------------

      Table Name : Warp State (All Cycles)
      ------------------------ ----------- ------------
Profiling k: 2048; l: 4; m: 7168; seed: 1111 (2/4):
      Metric Name              Metric Unit Metric Value
      ------------------------ ----------- ------------
      Stall Long Scoreboard           inst         9.45
      Stall Wait                      inst         2.11
      Stall Short Scoreboard          inst         1.36
      Selected                        inst            1
      Stall Barrier                   inst         0.86
      Stall Not Selected              inst         0.41
      Stall Branch Resolving          inst         0.29
      Stall Math Pipe Throttle        inst         0.24
      Stall No Instruction            inst         0.09
      Stall Dispatch Stall            inst         0.01
      Stall Drain                     inst         0.00
      Stall LG Throttle               inst            0
      Stall Membar                    inst            0
      Stall MIO Throttle              inst         0.00
      Stall Misc                      inst         0.00
      Stall Sleeping                  inst            0
      Stall Tex Throttle              inst            0
      ------------------------ ----------- ------------

    void at::elementwise_kernel<128, 4, void at::gpu_kernel_impl_nocast<at::direct_copy_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 3)]::operator ()() lambda() (instance 10)]::operator ()() lambda(c10::Half) (instance 1)]>(at::TensorIteratorBase &, const T1 &)::[lambda(int) (instance 1)]>(int, T3) (56, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 10.0

      Table Name : GPU Throughput
      ---------------- ----------- ------------
      Metric Name      Metric Unit Metric Value
      ---------------- ----------- ------------
      Memory [%]                 %         2.35
      Compute (SM) [%]           %         0.62
      ---------------- ----------- ------------

      Table Name : Pipe Utilization (% of active cycles)
Profiling k: 2048; l: 4; m: 7168; seed: 1111 (3/4):
      Table Description : Pipeline utilization based on the number of cycles the pipeline was active. This takes the rates of different instructions executing on the pipeline into account. For an instruction requiring 4 cycles to complete execution, the counter is increased by 1 for 4 cycles. Use this to understand the pipeline utilization for the time it was active.
      -------------------- ----------- ------------
      Metric Name          Metric Unit Metric Value
      -------------------- ----------- ------------
      ALU                            %         2.75
      FMA                            %         1.19
      TMEM (Tensor Memory)           %            0
      FP64                           %            0
      Shared (FP64+Tensor)           %            0
      TC                             %            0
      Tensor (All)                   %            0
      Tensor (DP)                    %            0
      Tensor (FP)                    %            0
      Tensor (INT)                   %            0
      TMA                            %            0
      -------------------- ----------- ------------

      Table Name : Warp State (All Cycles)
      ------------------------ ----------- ------------
      Metric Name              Metric Unit Metric Value
      ------------------------ ----------- ------------
      Stall Long Scoreboard           inst        14.58
      Stall Short Scoreboard          inst         4.56
      Stall No Instruction            inst         4.44
      Stall Wait                      inst         2.16
      Selected                        inst            1
      Stall Branch Resolving          inst         0.24
      Stall Drain                     inst         0.13
      Stall Dispatch Stall            inst         0.10
      Stall Barrier                   inst            0
Profiling k: 2048; l: 4; m: 7168; seed: 1111 (4/4):
      Stall LG Throttle               inst            0
      Stall Math Pipe Throttle        inst            0
      Stall Membar                    inst            0
      Stall MIO Throttle              inst            0
      Stall Misc                      inst            0
      Stall Not Selected              inst            0
      Stall Sleeping                  inst            0
      Stall Tex Throttle              inst            0
      ------------------------ ----------- ------------
