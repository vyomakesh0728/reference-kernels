# Example Query Outputs

These examples were generated from the current local index in sparse-only mode because no Mixedbread API key was present in this workspace at build time.

## Query

`__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4 operand order`

Top grounded hits:

1. `clang/test/SemaOpenCL/builtins-amdgcn-error-gfx950-param.cl::__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4:48-52`
   Source: `https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/clang/test/SemaOpenCL/builtins-amdgcn-error-gfx950-param.cl`
2. `clang/test/CodeGenOpenCL/builtins-amdgcn-mfma.cl::__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4:449-450`
   Source: `https://github.com/llvm/llvm-project/blob/696e82db339ce6dc907378bd977ec7857cc892e9/clang/test/CodeGenOpenCL/builtins-amdgcn-mfma.cl`

Why this is useful:

- the Sema test gives operand positions by showing which arguments must be compile-time constants
- the CodeGenOpenCL test provides a concrete callsite the agents can pattern-match against

## Query

`Which local session summary mentions exact m64 direct native scaled MFMA memory fault benchmark failures?`

Top grounded hit:

1. `session-summary.md::Highest-Signal Failed Probes:43-56`
   Source: `/Users/v/reference-kernels/problems/amd/session-summary.md`

Quoted evidence:

- `Same GPU memory fault even with scale_a = 127 and scale_b = 127.`
- `Same fault before any JSON phase markers flush, so the crash is still inside or immediately around the direct scaled-MFMA feed/launch path.`

Why this is useful:

- it surfaces the exact local failure memory we want future agents to reuse before guessing at operand or launch changes

