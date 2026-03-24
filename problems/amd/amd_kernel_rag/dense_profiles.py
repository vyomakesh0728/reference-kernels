from __future__ import annotations

from dataclasses import dataclass

from .chunking import Chunk


DEFAULT_DENSE_PROFILE = "mxfp4-mm-free"
VALID_DENSE_PROFILES = ("mxfp4-mm-free", "full", "none")

_MXFP4_MM_FREE_SOURCE_ALLOWLIST = {
    "amd-local-docs",
    "llvm-amdgpu-usage-html",
    "rocm-cdna4-gemm-blog",
    "rocm-matrix-cores-cdna-blog",
    "rocm-gemm-optimization-blog",
    "rocm-matrix-cores-blog",
    "amd-aocl-small-matrices-blog",
    "rocm-ai-blogs-index",
    "hazy-hk-blog",
    "hazy-amd-brr-blog",
}

_MXFP4_MM_FREE_LLVM_DISPLAY_PATHS = {
    "clang/include/clang/Basic/BuiltinsAMDGPU.td",
    "llvm/include/llvm/IR/IntrinsicsAMDGPU.td",
}


@dataclass(frozen=True)
class DenseSelection:
    profile: str
    chunks: list[Chunk]
    chunk_count: int
    text_chars: int
    sources: list[str]
    coverage_ratio: float


def select_dense_chunks(chunks: list[Chunk], *, profile: str) -> DenseSelection:
    normalized = profile.strip().lower()
    if normalized not in VALID_DENSE_PROFILES:
        raise ValueError(f"unknown dense profile: {profile}")

    if normalized == "none":
        selected: list[Chunk] = []
    elif normalized == "full":
        selected = list(chunks)
    else:
        selected = [
            chunk
            for chunk in chunks
            if chunk.source_id in _MXFP4_MM_FREE_SOURCE_ALLOWLIST
            or (
                chunk.source_id == "llvm-project-amdgpu"
                and chunk.display_path in _MXFP4_MM_FREE_LLVM_DISPLAY_PATHS
            )
        ]

    source_ids = sorted({chunk.source_id for chunk in selected})
    chunk_count = len(selected)
    total_chunks = len(chunks)
    text_chars = sum(len(chunk.text) for chunk in selected)
    coverage_ratio = (chunk_count / total_chunks) if total_chunks else 0.0
    return DenseSelection(
        profile=normalized,
        chunks=selected,
        chunk_count=chunk_count,
        text_chars=text_chars,
        sources=source_ids,
        coverage_ratio=coverage_ratio,
    )
