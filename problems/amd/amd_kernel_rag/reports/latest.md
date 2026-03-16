# AMD Kernel RAG Evaluation

- Queries: 24
- Recall@5: 0.5417
- Recall@20: 0.7083
- MRR@10: 0.3358

## Top Misses

### q02-workitem-id

- Query: `Which builtin gives lane or workitem id x on AMDGPU? __builtin_amdgcn_workitem_id_x details`
- Why hard: The answer hides in a large builtin table and can be diluted by nearby builtin families.
- Top retrieved chunks:
  - .agent-loop/handrolled/mxfp4_mm/round-0034-mfma_scale_exact_m32/submission.py::__builtin_amdgcn_workitem_id_x:325-362 (/Users/v/reference-kernels/problems/amd/.agent-loop/handrolled/mxfp4_mm/round-0034-mfma_scale_exact_m32/submission.py)
  - .agent-loop/handrolled/mxfp4_mm/round-0035-mfma_scale_exact_m32/submission.py::__builtin_amdgcn_workitem_id_x:325-362 (/Users/v/reference-kernels/problems/amd/.agent-loop/handrolled/mxfp4_mm/round-0035-mfma_scale_exact_m32/submission.py)
  - .agent-loop/handrolled/mxfp4_mm/round-0035-mfma_scale_exact_m32_launch_bounds/submission.py::__builtin_amdgcn_workitem_id_x:326-363 (/Users/v/reference-kernels/problems/amd/.agent-loop/handrolled/mxfp4_mm/round-0035-mfma_scale_exact_m32_launch_bounds/submission.py)
  - .agent-loop/handrolled/mxfp4_mm/round-0036-mfma_scale_exact_m32_launch_bounds/submission.py::__builtin_amdgcn_workitem_id_x:326-363 (/Users/v/reference-kernels/problems/amd/.agent-loop/handrolled/mxfp4_mm/round-0036-mfma_scale_exact_m32_launch_bounds/submission.py)
  - .agent-loop/handrolled/mxfp4_mm/round-0037-mfma_scale_exact_m32/submission.py::__builtin_amdgcn_workitem_id_x:325-362 (/Users/v/reference-kernels/problems/amd/.agent-loop/handrolled/mxfp4_mm/round-0037-mfma_scale_exact_m32/submission.py)

### q12-gcnasm-opcode

- Query: `gcnasm opcode examples and AMD assembly encoding reference`
- Why hard: The repo mixes implementation and examples, so broad lexical queries can land on tooling code instead of the useful examples.
- Top retrieved chunks:
  - https://llvm.org/docs/AMDGPUUsage.html::VALU¶:26822-26868 (https://llvm.org/docs/AMDGPUUsage.html)
  - https://hazyresearch.stanford.edu/blog/2025-11-09-hk::Climbing out of the CUDA moat: Introducing HipKittens:45-53 (https://hazyresearch.stanford.edu/blog/2025-11-09-hk)
  - llvm/test/CodeGen/AMDGPU/ran-out-of-registers-errors.ll:1-63 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/llvm/test/CodeGen/AMDGPU/ran-out-of-registers-errors.ll)
  - llvm/test/CodeGen/AMDGPU/branch-relaxation-inst-size-gfx1250.mir::instruction 12 bytes (4-byte opcode + 8-byte literal) instead of 8 bytes:2-2 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/llvm/test/CodeGen/AMDGPU/branch-relaxation-inst-size-gfx1250.mir)
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html::Summary#:607-608 (https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html)

### q16-fp8-mm-builtin-example

- Query: `Local fp8-mm HIP example with __builtin_amdgcn_mfma_f32_16x16x32_bf16`
- Why hard: This must retrieve a concrete local code example instead of a more abstract builtin definition.
- Top retrieved chunks:
  - .agent-loop/handrolled/mxfp4_mm/round-0024-mfma_exact_m16/submission.py::__builtin_amdgcn_mfma_f32_16x16x32_bf16:135-156 (/Users/v/reference-kernels/problems/amd/.agent-loop/handrolled/mxfp4_mm/round-0024-mfma_exact_m16/submission.py)
  - .agent-loop/handrolled/mxfp4_mm/round-0025-mfma_exact_m16_launch_bounds/submission.py::__builtin_amdgcn_mfma_f32_16x16x32_bf16:135-156 (/Users/v/reference-kernels/problems/amd/.agent-loop/handrolled/mxfp4_mm/round-0025-mfma_exact_m16_launch_bounds/submission.py)
  - agent_loop/handroll.py::__builtin_amdgcn_mfma_f32_16x16x32_bf16:874-895 (/Users/v/reference-kernels/problems/amd/agent_loop/handroll.py)
  - clang/test/SemaOpenCL/builtins-amdgcn-error-gfx950-param.cl::__builtin_amdgcn_mfma_f32_16x16x32_bf16:67-71 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/clang/test/SemaOpenCL/builtins-amdgcn-error-gfx950-param.cl)
  - clang/test/CodeGenOpenCL/builtins-amdgcn-mfma.cl::__builtin_amdgcn_mfma_f32_16x16x32_bf16:468-469 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/clang/test/CodeGenOpenCL/builtins-amdgcn-mfma.cl)

### q18-raw-ptr-buffer-load

- Query: `Where is llvm.amdgcn.raw.ptr.buffer.load defined?`
- Why hard: The intrinsic name contains dots in the query and underscores in the source.
- Top retrieved chunks:
  - https://llvm.org/docs/AMDGPUUsage.html::llvm.amdgcn.raw.ptr.buffer.load:2262-2262 (https://llvm.org/docs/AMDGPUUsage.html)
  - llvm/test/CodeGen/AMDGPU/GlobalISel/llvm.amdgcn.make.buffer.rsrc.ll::llvm.amdgcn.raw.ptr.buffer.load:583-583 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/llvm/test/CodeGen/AMDGPU/GlobalISel/llvm.amdgcn.make.buffer.rsrc.ll)
  - llvm/test/CodeGen/AMDGPU/llvm.amdgcn.make.buffer.rsrc.ll::llvm.amdgcn.raw.ptr.buffer.load:528-528 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/llvm/test/CodeGen/AMDGPU/llvm.amdgcn.make.buffer.rsrc.ll)
  - llvm/test/CodeGen/AMDGPU/GlobalISel/llvm.amdgcn.make.buffer.rsrc.ll::llvm.amdgcn.raw.ptr.buffer.load:96-98 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/llvm/test/CodeGen/AMDGPU/GlobalISel/llvm.amdgcn.make.buffer.rsrc.ll)
  - llvm/test/CodeGen/AMDGPU/llvm.amdgcn.make.buffer.rsrc.ll::llvm.amdgcn.raw.ptr.buffer.load:88-90 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/llvm/test/CodeGen/AMDGPU/llvm.amdgcn.make.buffer.rsrc.ll)

### q19-wavefront-size

- Query: `AMDGPU usage docs wavefront size and wave32 wave64 details`
- Why hard: Wavefront text appears in many sections, so chunk locality matters.
- Top retrieved chunks:
  - llvm/test/CodeGen/AMDGPU/GlobalISel/legalize-brcond.mir::used.:254-290 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/llvm/test/CodeGen/AMDGPU/GlobalISel/legalize-brcond.mir)
  - llvm/test/CodeGen/AMDGPU/GlobalISel/legalize-brcond.mir::used.:313-347 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/llvm/test/CodeGen/AMDGPU/GlobalISel/legalize-brcond.mir)
  - llvm/test/CodeGen/AMDGPU/attr-amdgpu-flat-work-group-size-vgpr-limit.ll:545-594 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/llvm/test/CodeGen/AMDGPU/attr-amdgpu-flat-work-group-size-vgpr-limit.ll)
  - llvm/test/CodeGen/AMDGPU/GlobalISel/bool-legalization.ll::bb1:107-142 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/llvm/test/CodeGen/AMDGPU/GlobalISel/bool-legalization.ll)
  - llvm/test/CodeGen/AMDGPU/skip-if-dead.ll::llvm.amdgcn.kill:8-38 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/llvm/test/CodeGen/AMDGPU/skip-if-dead.ll)

### q20-gcnasm-mfma

- Query: `Does gcnasm include MFMA or matrix instruction examples?`
- Why hard: The repo may mention MFMA sparsely, so this exposes source coverage gaps quickly.
- Top retrieved chunks:
  - https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html::Matrix Core Instructions#:163-196 (https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html)
  - https://llvm.org/docs/AMDGPUUsage.html::Instruction Examples¶:26744-26744 (https://llvm.org/docs/AMDGPUUsage.html)
  - https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr::HipKittens memory access patterns:32-50 (https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr)
  - llvm/test/CodeGen/AMDGPU/inflate-reg-class-vgpr-mfma-to-av-with-load-source.mir::of a simple copy.:1087-1093 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/llvm/test/CodeGen/AMDGPU/inflate-reg-class-vgpr-mfma-to-av-with-load-source.mir)
  - llvm/test/CodeGen/AMDGPU/inflate-reg-class-vgpr-mfma-to-av-with-load-source.mir::of used lanes instead of a simple copy,:1169-1175 (https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/llvm/test/CodeGen/AMDGPU/inflate-reg-class-vgpr-mfma-to-av-with-load-source.mir)

### q21-kernel-mutator-scaled-mfma

- Query: `Our agent_loop kernel mutator focus on replacing tiled inner loop with CDNA4 scaled MFMA intrinsics`
- Why hard: This is long local phrasing that should hit a very specific strategy string.
- Top retrieved chunks:
  - skills/optimization-skill/references/cdna4-gemm-ladder.md::Order:8-40 (/Users/v/reference-kernels/problems/amd/skills/optimization-skill/references/cdna4-gemm-ladder.md)
  - .agent-loop/handrolled/mxfp4_mm/round-0051-deaiter_exact_m64_scaled_split16/plan.json:1-9 (/Users/v/reference-kernels/problems/amd/.agent-loop/handrolled/mxfp4_mm/round-0051-deaiter_exact_m64_scaled_split16/plan.json)
  - skills/amd-kernel-speedrun/SKILL.md::Phase 2 Optimization Ladder:56-78 (/Users/v/reference-kernels/problems/amd/skills/amd-kernel-speedrun/SKILL.md)
  - agent_loop/llm_mutator.py::_source_inspirations:135-185 (/Users/v/reference-kernels/problems/amd/agent_loop/llm_mutator.py)
  - .agent-loop/problems/mxfp4_mm/manual/hip_reference_tiled/submission.py::AGENT_LOOP_META: {"attempt": 2, "generator": {"kind": "manual_reference"}, "gpu": "MI355X", "leaderboard": "amd-mxfp4-mm", "policy_profile": {"family": "hip_explore", "name": "hip_contract_reference"}, "problem": "mxfp4_mm"}:1-62 (/Users/v/reference-kernels/problems/amd/.agent-loop/problems/mxfp4_mm/manual/hip_reference_tiled/submission.py)
