#!/usr/bin/env python3
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
LIVE_PROBLEM_DIR = REPO_ROOT.parent / "amd_202602" / "moe-mxfp4"
DEFAULT_SOURCE = REPO_ROOT / ".agent-loop" / "manual" / "dispatch_pack_sparse256_v6" / "submission.py"


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
        ("(1e-1,1]", (rel > 1e-1) & (rel <= 1)),
        (">1", rel > 1),
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
    return {
        "shape": list(reference.shape),
        "mismatch_count": int(mismatches.sum().item()),
        "mae": float(abs_err.mean().item()),
        "max_abs_error": float(worst_val.item()),
        "max_rel_error": float(rel_err.max().item()),
        "worst_index": [worst_row, worst_col],
        "reference_at_worst": float(ref.reshape(-1)[worst_idx].item()),
        "candidate_at_worst": float(got.reshape(-1)[worst_idx].item()),
        "relative_error_histogram": _relative_histogram(reference, candidate),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare dispatch-pack split path against the exact two-stage path.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dhidden", type=int, default=7168)
    parser.add_argument("--dexpert", type=int, default=256)
    parser.add_argument("--nroutedexperts", type=int, default=256)
    parser.add_argument("--nsharedexperts", type=int, default=1)
    parser.add_argument("--nexpertspertoken", type=int, default=8)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=9371)
    parser.add_argument("--atol", type=float, default=1e-2)
    parser.add_argument("--rtol", type=float, default=1e-2)
    args = parser.parse_args()

    sys.path.insert(0, str(LIVE_PROBLEM_DIR))
    candidate = _load_module("dispatch_pack_probe_submission", args.source)
    reference = _load_module("dispatch_pack_probe_reference", LIVE_PROBLEM_DIR / "reference.py")

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
    (
        hidden_states,
        gate_up_weight,
        down_weight,
        gate_up_weight_scale,
        down_weight_scale,
        gate_up_weight_shuffled,
        down_weight_shuffled,
        gate_up_weight_scale_shuffled,
        down_weight_scale_shuffled,
        topk_weights,
        topk_ids,
        config,
    ) = data

    with torch.no_grad():
        exact = candidate._green_two_stage_exact(
            hidden_states,
            gate_up_weight_shuffled,
            down_weight_shuffled,
            gate_up_weight_scale_shuffled,
            down_weight_scale_shuffled,
            topk_weights,
            topk_ids,
            config,
        )
        split = candidate._dispatch_split_shared_path(
            hidden_states,
            gate_up_weight,
            down_weight,
            gate_up_weight_scale,
            down_weight_scale,
            gate_up_weight_shuffled,
            down_weight_shuffled,
            gate_up_weight_scale_shuffled,
            down_weight_scale_shuffled,
            topk_weights,
            topk_ids,
            config,
        )

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
        "split_vs_exact": _error_metrics(exact, split, atol=args.atol, rtol=args.rtol),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
