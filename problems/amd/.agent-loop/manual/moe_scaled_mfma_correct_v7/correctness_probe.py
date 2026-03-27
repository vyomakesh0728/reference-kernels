#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
LIVE_PROBLEM_DIR = REPO_ROOT.parent / "amd_202602" / "moe-mxfp4"
DEFAULT_SOURCE = REPO_ROOT / ".agent-loop" / "manual" / "moe_scaled_mfma_correct_v7" / "submission.py"


def _load_module(module_name: str, source_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _relative_histogram(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, int]:
    diff = (candidate - reference).abs().to(torch.float32)
    denom = torch.maximum(reference.abs().to(torch.float32), torch.full_like(diff, 1e-6))
    rel = (diff / denom).reshape(-1)
    bins = [
        ("<=1e-4", rel <= 1e-4),
        ("(1e-4,1e-3]", (rel > 1e-4) & (rel <= 1e-3)),
        ("(1e-3,1e-2]", (rel > 1e-3) & (rel <= 1e-2)),
        ("(1e-2,1e-1]", (rel > 1e-2) & (rel <= 1e-1)),
        ("(1e-1,1]", (rel > 1e-1) & (rel <= 1.0)),
        (">1", rel > 1.0),
    ]
    return {label: int(mask.sum().item()) for label, mask in bins}


def _error_metrics(reference: torch.Tensor, candidate: torch.Tensor, atol: float, rtol: float) -> dict[str, object]:
    ref = reference.to(torch.float32)
    got = candidate.to(torch.float32)
    abs_err = (got - ref).abs()
    denom = torch.maximum(ref.abs(), torch.full_like(abs_err, 1e-6))
    rel_err = abs_err / denom
    mismatches = ~torch.isclose(got, ref, atol=atol, rtol=rtol)
    flat_abs = abs_err.reshape(-1)
    worst_val, worst_idx = torch.max(flat_abs, dim=0)
    worst_idx = int(worst_idx.item())
    cols = reference.shape[-1]
    worst_row = worst_idx // cols
    worst_col = worst_idx % cols
    tol_1 = abs_err <= (0.5 + 0.05 * ref.abs())
    tol_2 = abs_err <= (1.0 + 0.10 * ref.abs())
    return {
        "shape": list(reference.shape),
        "mismatch_count": int(mismatches.sum().item()),
        "mae": float(abs_err.mean().item()),
        "max_abs_error": float(worst_val.item()),
        "max_rel_error": float(rel_err.max().item()),
        "worst_index": [worst_row, worst_col],
        "reference_at_worst": float(ref.reshape(-1)[worst_idx].item()),
        "candidate_at_worst": float(got.reshape(-1)[worst_idx].item()),
        "tolerance_hits": {
            "atol_0.5_rtol_0.05": int(tol_1.sum().item()),
            "atol_1.0_rtol_0.10": int(tol_2.sum().item()),
        },
        "relative_error_histogram": _relative_histogram(reference, candidate),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare full scaled-MFMA MoE candidate against the AITER fused_moe reference.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dhidden", type=int, default=4096)
    parser.add_argument("--dexpert", type=int, default=1536)
    parser.add_argument("--nroutedexperts", type=int, default=64)
    parser.add_argument("--nsharedexperts", type=int, default=1)
    parser.add_argument("--nexpertspertoken", type=int, default=6)
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=81934)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--rtol", type=float, default=2e-2)
    args = parser.parse_args()

    sys.path.insert(0, str(LIVE_PROBLEM_DIR))
    candidate = _load_module("moe_scaled_mfma_probe_submission", args.source)
    reference = _load_module("moe_scaled_mfma_probe_reference", LIVE_PROBLEM_DIR / "reference.py")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm device is required for this probe")

    data = reference.generate_input(
        dhidden=args.dhidden,
        dexpert=args.dexpert,
        nroutedexperts=args.nroutedexperts,
        nexpertspertoken=args.nexpertspertoken,
        nsharedexperts=args.nsharedexperts,
        bs=args.bs,
        seed=args.seed,
    )

    with torch.no_grad():
        ref = reference.ref_kernel(data)
        got = candidate.custom_kernel(data)

    payload = {
        "source": str(args.source),
        "shape": {
            "dhidden": args.dhidden,
            "dexpert": args.dexpert,
            "nroutedexperts": args.nroutedexperts,
            "nsharedexperts": args.nsharedexperts,
            "nexpertspertoken": args.nexpertspertoken,
            "bs": args.bs,
            "seed": args.seed,
        },
        "tolerance": {"atol": args.atol, "rtol": args.rtol},
        "candidate_vs_reference": _error_metrics(ref, got, atol=args.atol, rtol=args.rtol),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
