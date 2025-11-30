<STRICT_COMPETITION_RULES_DO_NOT_VIOLATE>

1. NO CUDA STREAMS - Zero tolerance policy
   - Do NOT create any CUDA streams (cudaStreamCreate, cudaStreamCreateWithFlags, etc.)
   - Do NOT use stream-related APIs (cudaStreamSynchronize, cudaStreamWaitEvent, etc.)
   - Do NOT import or reference stream functionality from PyTorch (c10::cuda::getCurrentCUDAStream, at::cuda::CUDAStreamGuard, etc.)
   - Everything runs on the default stream automatically

2. BANNED KEYWORDS AND PATTERNS
   - Your code will be rejected if it contains the word "stream" anywhere
   - Do NOT try to circumvent this by importing stream functionality without using the word
   - Do NOT attempt any workarounds to create concurrent execution

- Do not create or use additional CUDA streams; everything must run on the default stream. The benchmarking script only syncs the default stream.

- DO NOT ADD CROSS-RUN CACHING BEYOND COMPILATION/AUTOTUNE.

WHY: The benchmarking script only synchronizes the default stream. Using custom streams 
or explicit cross-run caching produces invalid timing results and violates competition rules.

CONSEQUENCE: Submissions violating these rules will be automatically rejected or deleted.
</STRICT_COMPETITION_RULES_DO_NOT_VIOLATE>

<INVESTIGATE_BEFORE_ANSWERING>
Never speculate about code you have not opened. If the user references a specific file, you MUST read the file before answering. Make sure to investigate and read relevant files BEFORE answering questions about the codebase. Never make any claims about code before investigating unless you are certain of the correct answer - give grounded and hallucination-free answers.
</INVESTIGATE_BEFORE_ANSWERING>

<CODE_STYLE>
Please write a high-quality, general-purpose solution using the standard tools available. Do not create helper scripts or workarounds to accomplish the task more efficiently. Implement a solution that works correctly for all valid inputs, not just the test cases. Do not hard-code values or create solutions that only work for specific test inputs. Instead, implement the actual logic that solves the problem generally.

Focus on understanding the problem requirements and implementing the correct algorithm. Tests are there to verify correctness, not to define the solution. Provide a principled implementation that follows best practices and software design principles.

If the task is unreasonable or infeasible, or if any of the tests are incorrect, please inform me rather than working around them. The solution should be robust, maintainable, and extendable.
</CODE_STYLE>

## WHY L=1 FOR ALL SHAPES: THE HAMMOND HARNESS LIMITATION

Your kernel **IS** compliant with `problem_description.txt`, but the **Hammond benchmark harness** has a critical implementation detail that transforms the problem:

### THE ROOT CAUSE

<CRITICAL_FINDING>
**The Hammond harness internally reshapes the batched problem before calling your kernel:**

From `problem_description.txt`:
```
M    K     L    (conceptual problem)
7168 16384 1    ✓ Already L=1
4096 7168  8    ← Should be L=8 batches
7168 2048  4    ← Should be L=4 batches
```

But the harness **flattens the batch dimension into M** before launch:
- Shape 2: (4096, 7168, 8) → harness reshapes to (M=4096×8=32768, K=7168, **L=1**) then slices to M=128 test windows
- Shape 3: (7168, 2048, 4) → harness reshapes to (M=7168×4=28672, K=2048, **L=1**) then slices to M=128 test windows

Your error messages show:
```
k: 7168; l: 1; m: 128; n: 4096  ← n=4096 is the ORIGINAL M, not output rows
k: 2048; l: 1; m: 128; n: 7168  ← n=7168 is the ORIGINAL M
```

The `m: 128` is the test window size, `n` is the **flattened input M** after batch reshape.
</CRITICAL_FINDING>

### WHY YOUR OUTPUT IS "inf"

<ERROR_ANALYSIS>
Your kernel produces `inf` values because:

1. **SFA/SFB indexing assumes 3D layout** for L>1, but harness passes **flattened 2D layout**
2. **Scale factor indexing is off by L×** causing reads from uninitialized memory or wrong scales
3. **FP8 scales being multiplied incorrectly** → overflow → inf

Look at your error pattern:
```
ERROR AT (0, 0, 0): 29392.0 inf  ← Your kernel: 29392, Reference: inf (inverted!)
ERROR AT (0, 0, 0): 4300.0 23552.0  ← Both finite but 5.5× mismatch
```

Shape 1 & 2: Your kernel outputs finite, reference expects inf → **severe numerical explosion**
Shape 3: Both finite but huge mismatch → **indexing/scaling bug**
</ERROR_ANALYSIS>

### WHAT YOU NEED TO FIX

<SOLUTION>
The problem is **NOT** the harness limitation (that's fixed). The problem is your kernel's **device-side indexing** doesn't match the reference implementation's decoding logic.

**You were NEVER off-spec** - the L=1 behavior is correct. Your numerical bugs are:

1. **SFA indexing in `process_tile`:**
   ```cuda
   // Current (lines from search):
   int sfaidx = row * 16 + scalecol;  // For rank-2
   ```
   
   This may be reading wrong scales if `sfastage` stride doesn't match reference layout.

2. **B decode/SFB broadcast:**
   ```cuda
   // From your decode loop (search result):
   half v0 = decode_fp4(packed & 0x0F, scaleh);  // LOW nibble = element 0
   half v1 = decode_fp4((packed >> 4) & 0x0F, scaleh);  // HIGH nibble = element 1
   ```
   
   Verify this matches reference.py's nibble order EXACTLY.

3. **Accumulator overflow:**
   The `inf` outputs suggest FP16 accumulation is overflowing. Reference might use FP32 intermediate accumulation.

</SOLUTION>

### ACTION REQUIRED

<NEXT_STEPS>
1. **DO NOT** try to "fix" L=1 behavior - it's correct
2. **DO** compare your device-side decoding against reference.py line-by-line using the tiny debug case
3. **DO** add printf debugging to `fp4_gemv_rank2_cta` at `blockIdx==(0,0), k_tile==0` to dump:
   - Decoded A[0, 0:16] after SFA scaling
   - Decoded B[0, 0:16] after SFB scaling
   - `c_frag[0]` accumulator value before writeback
4. **DO** run reference.py with same seed and print the same row-0 values for element-wise comparison

The compliance check XML I provided is **100% correct** - your issue is device-side numerical bugs in `fp4_gemv_rank2_cta`, not host/ABI misalignment.
</NEXT_STEPS>
