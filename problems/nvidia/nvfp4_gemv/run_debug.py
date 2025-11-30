import torch
import reference
import submission
import sys

def run_debug():
    print("Running debug comparison...")
    M = 128
    K = 256
    L = 1
    seed = 0
    
    print(f"Generating input with M={M}, K={K}, L={L}, seed={seed}")
    # generate_input returns: (a, b, sfa, sfb, sfa_permuted, sfb_permuted, c)
    # Note: reference.py generate_input returns 7 items.
    inputs = reference.generate_input(M, K, L, seed)
    a, b, sfa, sfb, sfa_permuted, sfb_permuted, c_ref = inputs
    
    # Run reference kernel (prints debug info)
    print("\n" + "="*80)
    print("RUNNING REFERENCE KERNEL")
    print("="*80)
    c_ref_out = reference.ref_kernel(inputs)
    
    # Run submission kernel
    print("\n" + "="*80)
    print("RUNNING SUBMISSION KERNEL")
    print("="*80)
    # submission.fp4_gemv expects (a, b, sfa, sfb, c)
    # It seems submission.py's fp4_gemv might have a different signature or expects specific inputs.
    # Let's check submission.py's fp4_gemv signature.
    # It takes `data: input_t` which is a tuple.
    # In submission.py: def fp4_gemv(data: input_t) -> output_t:
    # input_t is likely (a, b, sfa, sfb, sfa_permuted, sfb_permuted, c) based on reference.py
    # But submission.py might ignore some.
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    try:
        c_submit = submission.custom_kernel(inputs)
    except Exception as e:
        print(f"Error running submission kernel: {e}")
        import traceback
        traceback.print_exc()
        return
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f"Kernel execution time: {elapsed_time_ms * 1000:.2f} us")

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    # Compare outputs
    c_ref_cpu = c_ref_out.cpu()
    c_submit_cpu = c_submit.cpu()
    
    print(f"Ref shape: {c_ref_cpu.shape}")
    print(f"Sub shape: {c_submit_cpu.shape}")
    
    if torch.allclose(c_ref_cpu, c_submit_cpu, atol=1e-3, rtol=1e-3):
        print("Outputs MATCH!")
    else:
        print("Outputs MISMATCH!")
        diff = (c_ref_cpu - c_submit_cpu).abs()
        max_diff = diff.max().item()
        print(f"Max diff: {max_diff}")
        print(f"Ref[0,0,0]: {c_ref_cpu[0,0,0].item()}")
        print(f"Sub[0,0,0]: {c_submit_cpu[0,0,0].item()}")

if __name__ == "__main__":
    run_debug()
