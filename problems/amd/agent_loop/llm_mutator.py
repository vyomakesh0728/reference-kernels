from __future__ import annotations

import ast
import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import py_compile
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from .config import AppConfig, load_config
from .kernel_mutator import (
    META_RE,
    candidate_attempt,
    choose_policy_profile,
    choose_variant,
    history_entries,
    load_context,
    load_parent_meta,
    render_submission,
)


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)
CUSTOM_KERNEL_RE = re.compile(
    r"def\s+custom_kernel\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:\n(?P<body>(?:[ \t]+.*(?:\n|$)|\s*\n)*)",
    re.MULTILINE,
)
TOP_LEVEL_TRANSCRIPT_MARKERS = (
    "OpenAI Codex v",
    "workdir:",
    "model:",
    "provider:",
    "approval:",
    "sandbox:",
    "reasoning effort:",
    "reasoning summaries:",
    "session id:",
    "tokens used",
    "deprecated:",
    "mcp:",
    "codex",
    "user",
    "assistant",
)
PYTHON_START_MARKERS = (
    "#!POPCORN",
    "from __future__ import annotations",
    "import ",
    "from ",
    "def custom_kernel",
)
CODEX_USAGE_LIMIT_RE = re.compile(r"you've hit your usage limit", re.IGNORECASE)
CODEX_TRY_AGAIN_RE = re.compile(r"try again at\s+([0-9]{1,2}:[0-9]{2}\s*[AP]M)", re.IGNORECASE)
REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_SKILLS_ROOT = REPO_ROOT / "skills"
AOT_GUIDANCE = [
    "Use an Atom-of-Thoughts style internal process: treat the current parent submission as the current Markov state, not as a transcript to copy verbatim.",
    "Preserve an answer-equivalence invariant: every next state must keep the exact live contract, semantics, and correctness behavior of the original task.",
    "Decomposition stage: internally split the kernel problem into 2-5 self-contained atomic bottlenecks such as contract correctness, quant/dequant flow, memory layout, tile geometry, or launch configuration.",
    "Choose exactly one atomic bottleneck for this transition. Do not change multiple independent bottlenecks in the same candidate.",
    "Contraction stage: output one full next-state submission.py that is self-contained, contract-faithful, and simpler to verify than a broad rewrite.",
    "Require monotonic complexity reduction: the next state should reduce open decisions or isolate one clearer bottleneck instead of expanding scope.",
    "Reflective refinement: before finalizing, internally verify that the new state preserves the live contract and meaningfully improves the chosen bottleneck. If not, revise once and keep the smaller change.",
    "Minimize historical dependence: use the parent source, compact history digest, and knowledge summary, but do not rely on long free-form transcripts.",
    "Terminate at the current atomic frontier: if the right next move is still correctness repair, keep the edit correctness-first instead of pretending to optimize throughput.",
    "Operate like an AutoKernel experiment: make one focused kernel edit per iteration so the outer loop can explicitly keep or revert it.",
]


def _family_label(desired_family: str | None) -> str:
    if desired_family == "hip_explore":
        return "HIP"
    if desired_family == "kernel_explore":
        return "Kernel"
    return "GPU"


def _default_skill_name(desired_family: str | None) -> str:
    del desired_family
    return str(LOCAL_SKILLS_ROOT / "amd-kernel-speedrun" / "SKILL.md")


def _relevant_skill_paths(problem_key: str, desired_family: str | None) -> list[str]:
    skill_paths = [str(LOCAL_SKILLS_ROOT / "amd-kernel-speedrun" / "SKILL.md")]
    if problem_key == "mxfp4_mm":
        skill_paths.append(str(LOCAL_SKILLS_ROOT / "amd-live-reference-correctness" / "SKILL.md"))
    if desired_family == "hip_explore":
        skill_paths.append(str(LOCAL_SKILLS_ROOT / "optimization-skill" / "SKILL.md"))

    seen: set[str] = set()
    ordered: list[str] = []
    for path in skill_paths:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _experiment_protocol(desired_family: str | None) -> list[str]:
    family = _family_label(desired_family)
    protocol = [
        "AutoKernel discipline: make one atomic kernel edit, let the outer loop keep or revert it, and do not hide multiple independent changes in one round.",
        "KernelAgent discipline: leave a clear artifact trail for this round so the parent, prompt, diff, result, and summary all describe the same focused experiment.",
        "ADRS discipline: optimize only against the evaluator contract and measured result, not vague intuition or unverifiable assembly fantasies.",
    ]
    if desired_family == "hip_explore":
        protocol.extend(
            [
                "Stay in one compilation path: Python submission.py + load_inline + HIP C++ on gfx950.",
                "Remove inherited Triton scaffold completely; the final HIP candidate should not keep import triton, @triton.jit, or unused Triton helper code.",
                "Competition purity rule: do not use alternate streams, async overlap tricks, event/timing hacks, background work, or benchmark-specific hacks that can distort measured microseconds.",
                "Do not include the lowercase token `stream` anywhere in the final submission source; KernelBot rejects such submissions before execution.",
                "Benchmark-fidelity rule: optimize for sustained steady-state throughput under the evaluator, not bursty best-case timings. Do not insert sleeps, idle gaps, launch batching tricks, or duty-cycle manipulations that make microseconds look better without improving real kernel throughput.",
                "Benchmark-fidelity rule: prefer changes that should survive warmed-up, repeated evaluation and robust aggregation; treat sub-10us-style vanity chasing and single-run noise as suspect.",
                f"Treat this as a {family} microkernel experiment over one lever at a time: tile shape, LDS movement, double buffering, swizzle, or scaled MFMA replacement.",
            ]
        )
    elif desired_family == "kernel_explore":
        protocol.extend(
            [
                "Keep the real hot path on the generated kernel and avoid dead scaffold.",
                "Treat this as one generated-kernel scheduling or dataflow experiment at a time, not a broad rewrite.",
            ]
        )
    return protocol


def _source_inspirations(problem_key: str, desired_family: str | None) -> list[dict[str, str]]:
    inspirations = [
        {
            "name": "AutoKernel",
            "kind": "repo",
            "idea": "one-kernel keep/revert loop with explicit experiment discipline",
        },
        {
            "name": "KernelAgent",
            "kind": "repo",
            "idea": "artifact-rich per-run directories and evaluator-driven iteration",
        },
        {
            "name": "ADRS/OpenEvolve",
            "kind": "repo",
            "idea": "treat the evaluator as the ground truth and iterate through compact experiments",
        },
        {
            "name": "Atom of Thoughts",
            "kind": "paper",
            "idea": "make one small Markov transition over the parent kernel rather than rewriting everything",
        },
    ]
    if problem_key == "mxfp4_mm" and desired_family == "hip_explore":
        inspirations.extend(
            [
                {
                    "name": "AMD CDNA4 GEMM Optimization Blog",
                    "kind": "blog",
                    "idea": "optimize LDS tiling, double buffering, swizzle, and global-to-LDS movement on CDNA4",
                },
                {
                    "name": "AMD CDNA4 ISA",
                    "kind": "pdf",
                    "idea": "scaled MFMA instructions on gfx950 are the long-term target for MXFP4 throughput",
                },
                {
                    "name": "StandardKernel High-Fidelity GPU Benchmarking",
                    "kind": "blog",
                    "idea": "favor steady-state, contention-free, non-bursty measurements and do not mistake measurement noise for kernel improvement",
                },
                {
                    "name": "ROCm 7.1 load_inline gist",
                    "kind": "gist",
                    "idea": "assume clang++ plus torch 2.10.0+rocm7.1 style load_inline environment targeting gfx950",
                },
            ]
        )
    return inspirations


def _write_experiment_plan(
    context: dict[str, object],
    *,
    policy_profile: dict[str, object],
    variant: dict[str, object],
) -> None:
    candidate_dir = context.get("candidate_dir")
    if not isinstance(candidate_dir, str) or not candidate_dir:
        return
    problem = context.get("problem")
    if not isinstance(problem, dict):
        return
    desired_family = context.get("desired_family")
    if not isinstance(desired_family, str):
        desired_family = None
    variant_name = variant.get("variant_name") if isinstance(variant.get("variant_name"), str) else ""
    strategy = variant.get("strategy") if isinstance(variant.get("strategy"), str) else ""
    focus = policy_profile.get("focus") if isinstance(policy_profile.get("focus"), str) else ""
    hypothesis = f"{focus}; variant={variant_name}; strategy={strategy}".strip("; ")
    payload = {
        "problem": problem.get("key"),
        "leaderboard": problem.get("leaderboard"),
        "gpu": problem.get("gpu"),
        "desired_family": desired_family,
        "attempt": context.get("attempt"),
        "parent_candidate_id": (context.get("parent") or {}).get("candidate_id") if isinstance(context.get("parent"), dict) else None,
        "policy_profile": policy_profile,
        "variant": variant,
        "hypothesis": hypothesis,
        "edit_budget": context.get("edit_budget"),
        "protocol": _experiment_protocol(desired_family),
        "sources": _source_inspirations(str(problem.get("key")), desired_family),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    path = Path(candidate_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "experiment.plan.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _codex_usage_limit_reset_time(text: str) -> datetime | None:
    if not CODEX_USAGE_LIMIT_RE.search(text):
        return None
    match = CODEX_TRY_AGAIN_RE.search(text)
    if not match:
        return None
    now = datetime.now().astimezone()
    try:
        clock = datetime.strptime(match.group(1).upper(), "%I:%M %p")
    except ValueError:
        return None
    reset_at = now.replace(
        hour=clock.hour,
        minute=clock.minute,
        second=0,
        microsecond=0,
    )
    if reset_at <= now:
        reset_at += timedelta(days=1)
    return reset_at


def _write_codex_wait_artifact(workspace_dir: Path, *, reset_at: datetime | None, stderr: str) -> None:
    payload = {
        "event": "codex_usage_limit",
        "detected_at": datetime.now().astimezone().isoformat(),
        "reset_at": reset_at.isoformat() if reset_at is not None else None,
        "message_tail": "\n".join(stderr.strip().splitlines()[-10:]),
    }
    (workspace_dir / "codex_usage_limit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _compact_command_failure(stderr: str, stdout: str) -> str:
    text = stderr.strip() or stdout.strip()
    if not text:
        return "rc=1"
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= 24:
        return "\n".join(lines)
    return "\n".join(lines[-24:])


def _extract_meta(source: str) -> dict[str, object]:
    match = META_RE.search(source)
    if not match:
        return {}
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}


def _replace_meta(source: str, meta: dict[str, object]) -> str:
    meta_line = f"# AGENT_LOOP_META: {json.dumps(meta, sort_keys=True)}"
    if META_RE.search(source):
        return META_RE.sub(lambda _: meta_line, source, count=1)

    lines = source.splitlines()
    insert_at = 0
    while insert_at < len(lines) and lines[insert_at].startswith("#!POPCORN"):
        insert_at += 1
    lines.insert(insert_at, meta_line)
    return "\n".join(lines)


def _extract_code(text: str) -> str:
    match = CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    stripped = text.strip()
    lines = stripped.splitlines()
    start_index = 0
    for index, line in enumerate(lines):
        candidate = line.lstrip()
        if any(candidate.startswith(marker) for marker in PYTHON_START_MARKERS):
            start_index = index
            break
    stripped = "\n".join(lines[start_index:]).strip()

    lines = stripped.splitlines()
    cleaned: list[str] = []
    for line in lines:
        if cleaned and not line.startswith((" ", "\t")):
            if any(line.startswith(marker) for marker in TOP_LEVEL_TRANSCRIPT_MARKERS):
                break
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _canonicalize_source(raw_source: str, seed_source: str, meta: dict[str, object]) -> str:
    seed_lines = seed_source.splitlines()
    canonical_headers = [line for line in seed_lines[:2] if line.startswith("#!POPCORN")]
    body = _extract_code(raw_source)
    body = META_RE.sub("", body).strip()

    body_lines = body.splitlines()
    while body_lines and body_lines[0].startswith("#!POPCORN"):
        body_lines.pop(0)
    while body_lines and body_lines[0].startswith("# AGENT_LOOP_META:"):
        body_lines.pop(0)

    final_lines = canonical_headers + [f"# AGENT_LOOP_META: {json.dumps(meta, sort_keys=True)}"]
    if body_lines:
        final_lines.extend(body_lines)
    return "\n".join(final_lines).strip() + "\n"


def _build_history_digest(history: list[dict[str, object]], limit: int = 6) -> str:
    entries: list[str] = []
    for entry in history[:limit]:
        meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
        critique = entry.get("critique") if isinstance(entry.get("critique"), dict) else {}
        variant = meta.get("variant") if isinstance(meta, dict) else {}
        policy_profile = meta.get("policy_profile") if isinstance(meta, dict) else {}
        pieces = [
            f"candidate={entry.get('candidate_id')}",
            f"status={entry.get('status')}",
        ]
        if isinstance(entry.get("score"), (int, float)):
            pieces.append(f"objective={float(entry['score'])}")
        if isinstance(variant, dict) and variant.get("variant_name"):
            pieces.append(f"variant={variant['variant_name']}")
        if isinstance(policy_profile, dict) and policy_profile.get("name"):
            pieces.append(f"policy={policy_profile['name']}")
        if critique.get("policy_signal"):
            pieces.append(f"signal={critique['policy_signal']}")
        if critique.get("summary"):
            pieces.append(f"summary={critique['summary']}")
        entries.append(" | ".join(str(piece) for piece in pieces))
    return "\n".join(entries) if entries else "(no prior history)"


def _repeated_failure_hint(history: list[dict[str, object]], limit: int = 8) -> str | None:
    counts: dict[str, int] = {}
    for entry in history[:limit]:
        critique = entry.get("critique") if isinstance(entry.get("critique"), dict) else {}
        summary = critique.get("summary")
        if isinstance(summary, str) and summary:
            counts[summary] = counts.get(summary, 0) + 1
    if not counts:
        return None
    summary, count = max(counts.items(), key=lambda item: item[1])
    if count < 2:
        return None
    return f"Recent repeated failure pattern ({count}x): {summary}"


def _history_has_ok_family(history: list[dict[str, object]], *, family: str) -> bool:
    for entry in history:
        if entry.get("status") != "ok":
            continue
        meta = entry.get("meta")
        if not isinstance(meta, dict):
            continue
        variant = meta.get("variant")
        if isinstance(variant, dict) and variant.get("family") == family:
            return True
    return False


def _problem_specific_guidance(
    problem_key: str,
    desired_family: str | None,
    history: list[dict[str, object]],
) -> list[str]:
    guidance: list[str] = []
    if desired_family == "kernel_explore":
        guidance.append(
            "The real hot path in custom_kernel must execute generated-kernel logic. Do not leave dead scaffold while delegating compute to an AITER anchor."
        )
    elif desired_family == "hip_explore":
        guidance.append(
            "The real hot path in custom_kernel must compile and execute HIP through torch.utils.cpp_extension.load_inline on MI355X/gfx950. Do not route the hot path through an AITER anchor."
        )

    repeated_failure = _repeated_failure_hint(history)
    if repeated_failure:
        guidance.append(repeated_failure)
    hip_ok_seen = _history_has_ok_family(history, family="hip_explore") if desired_family == "hip_explore" else False

    if problem_key == "mxfp4_mm":
        if desired_family == "hip_explore":
            guidance.extend(
                [
                    "Target gfx950 explicitly in both PYTORCH_ROCM_ARCH and --offload-arch.",
                    "Keep the optimization path in one language and one compiler pipeline: Python submission.py + load_inline + HIP C++ only.",
                    "Assume a ROCm 7.1-style runner with clang++ and torch 2.10.0+rocm7.1 semantics when choosing HIP/load_inline code patterns.",
                    "Preserve the shuffled MXFP4 contract semantics even in HIP mode. Before the first HIP correctness pass, a slow reference-oracle path is allowed: requantize logical A and B from the raw bf16 inputs with QuantType.per_1x32, dequantize them logically into float32, and use HIP only for the final float32-accumulate matmul before casting the output back to bf16.",
                    "Even in that reference-oracle phase, do not drop or delete B_q, B_shuffle, or B_scale_sh. Keep them sanity-checked in custom_kernel so the candidate still respects the live contract inputs.",
                    "Stay in contract-repair mode until one HIP candidate passes the remote correctness checks; do not pivot into memory-hierarchy tuning or scaled MFMA speculation before that.",
                    "After the first HIP correctness pass, follow the CDNA4 ladder in order: vectorized ingress, direct global-to-LDS movement when possible, bank-conflict-aware LDS layout/swizzle, software pipelining with double buffering, then scaled MFMA.",
                    "The long-term matrix-core targets are CDNA4 scaled MFMA instructions, especially V_MFMA_SCALE_F32_16X16X128_F8F6F4 and V_MFMA_SCALE_F32_32X32X64_F8F6F4.",
                    "When benchmarking candidate kernels, optimize for stable repeated performance under warmed-up evaluation rather than best-case burst timing; do not use sleep/gap tricks or anything that changes duty cycle without improving sustained throughput.",
                    "Make one focused HIP edit per iteration: memory hierarchy, tile shape, launch shape, or MFMA inner loop replacement, not all at once.",
                ]
            )
            if not hip_ok_seen:
                guidance.append(
                    "No HIP candidate has passed remote correctness yet. Lock this round to contract repair only and keep the variant within correctness-first hip_shared_bf16 seeds."
                )
        else:
            guidance.extend(
                [
                    "Preserve the shuffled MXFP4 contract exactly: quantize A with shuffle=True and consume B_shuffle/B_scale_sh consistently.",
                    "If the shuffled-B interpretation is still unstable, prefer a correctness-first kernel seed that requantizes both A and B from the original bf16 tensors, dequantizes them plainly, and only then runs the generated matmul path.",
                    "Do not change both semantic layout and tiling in the same candidate when correctness is failing repeatedly; fix contract/dataflow first, then tune tiles.",
                    "Prefer a minimal generated-kernel matmul path that is actually called from custom_kernel over a larger submission that still routes the hot path through aiter.gemm_a4w4.",
                ]
            )
    elif problem_key == "moe_mxfp4":
        guidance.extend(
            [
                "If correctness is unstable, keep the routing/topk contract fixed and only rewrite one expert-compute stage at a time.",
                "Avoid leaving the hot path on fused_moe while presenting side-code as the hot path.",
            ]
        )
    elif problem_key == "mixed_mla":
        guidance.extend(
            [
                "Bias toward latency-first rewrites for q_seq_len=1 decode, and keep the hot path on actual generated-kernel decode/attention code rather than mla_decode_fwd.",
                "Prefer small, correctness-preserving reductions in memory movement over broad rewrites that preserve the same slow structure.",
            ]
        )

    return guidance


def _build_prompt(
    config: AppConfig,
    context: dict[str, object],
    parent_source: str,
    seed_source: str,
    policy_profile: dict[str, object],
    variant: dict[str, object],
    *,
    workspace_edit: bool = False,
) -> tuple[str, str]:
    problem = context["problem"]
    history = history_entries(context)
    knowledge_path = Path(str(context["knowledge_path"]))
    knowledge_text = knowledge_path.read_text(encoding="utf-8") if knowledge_path.exists() else "{}"
    rag_context_path_raw = context.get("rag_context_path")
    rag_context_text = ""
    if isinstance(rag_context_path_raw, str):
        rag_context_path = Path(rag_context_path_raw)
        if rag_context_path.exists():
            rag_context_text = rag_context_path.read_text(encoding="utf-8")
    desired_family = context.get("desired_family")
    if not isinstance(desired_family, str):
        desired_family = None
    edit_budget = context.get("edit_budget")
    if not isinstance(edit_budget, dict):
        edit_budget = {}
    max_changed_lines = edit_budget.get("max_changed_lines")
    max_edit_hunks = edit_budget.get("max_edit_hunks")
    budget_lines: list[str] = []
    if isinstance(max_changed_lines, int):
        budget_lines.append(f"- keep the diff at or under {max_changed_lines} changed lines")
    if isinstance(max_edit_hunks, int):
        budget_lines.append(f"- keep the diff at or under {max_edit_hunks} unified-diff hunks")
    if budget_lines:
        budget_text = "\n".join(budget_lines)
    else:
        budget_text = "- no explicit edit budget provided"
    skill_paths = _relevant_skill_paths(str(problem["key"]), desired_family)

    family_label = _family_label(desired_family)
    purity_clause = (
        "Reject any hidden or cross-call cache, illegal_cache pattern, pointer replay, output replay, "
        "stale-tensor reuse, pointer-address keyed memoization, shape/stride/seed keyed memoization, "
        "or any scheme that reuses prior outputs or intermediates instead of recomputing from the current input. "
        "Every output must be derived from the current call's tensors only."
    )
    purity_requirements = [
        "- reject any hidden or cross-call cache, illegal_cache pattern, pointer replay, output replay, or stale-output reuse",
        "- do not cache or memoize by data_ptr(), storage pointer, shape tuple, stride tuple, seed, or any other proxy for prior outputs",
        "- do not keep global/module/static/LRU tensors or intermediates that can survive across calls or evaluations",
        "- every output must be recomputed from the current input tensors within the current call",
    ]
    if workspace_edit:
        system_prompt = (
            f"You are an expert AMD MI355X {family_label} kernel engineer working on a real competition submission. "
            "Edit the existing submission.py file in place inside the current isolated workspace. "
            "Preserve the live kernel contract, keep the two #!POPCORN header lines, define custom_kernel(data), "
            f"and {purity_clause} Favor correctness first, then speed. "
            "Use Codex helper agents only for internal analysis, then finish with a short plain-text summary of the one focused edit you made."
        )
        delivery_requirements = [
            "- edit the existing submission.py in place",
            "- do not create alternate code files or markdown fences",
            "- keep the final change localized and preserve untouched code where possible",
        ]
    else:
        system_prompt = (
            f"You are an expert AMD MI355X {family_label} kernel engineer working on a real competition submission. "
            "Generate exactly one Python submission file. Return only raw Python source, no markdown fences. "
            "Preserve the live kernel contract, keep the two #!POPCORN header lines, define custom_kernel(data), "
            f"and {purity_clause} Favor correctness first, then speed. "
            "Use Codex helper agents only for internal analysis, then return a single final submission.py."
        )
        delivery_requirements = [
            "- return only Python source",
            "- keep the #!POPCORN header lines valid",
        ]

    user_prompt = f"""
Problem key: {problem["key"]}
Leaderboard: {problem["leaderboard"]}
GPU: {problem["gpu"]}
Objective: minimize ranked geometric mean nanoseconds
Desired family: {context.get("desired_family")}

Selected policy profile:
{json.dumps(policy_profile, indent=2, sort_keys=True)}

Selected seed variant:
{json.dumps(variant, indent=2, sort_keys=True)}

Recent history digest:
{_build_history_digest(history)}

Knowledge memory:
{knowledge_text}

Retrieved reference context:
{rag_context_text or "(no retrieval index or no relevant grounded context was available for this round)"}

Current parent submission:
{parent_source}

Seed submission to improve:
{seed_source}

Write a new full submission.py candidate for this problem.
Requirements:
{chr(10).join(delivery_requirements)}
- preserve the exact data contract for the problem
- keep the real hot path in the requested family rather than calling back into an anchor op
- do not add fake speedups, persistent benchmark caches, or hidden state across evaluations
- purity rule: every output must be recomputed from the current input within the current call
- reject any idea that depends on pointer identity, allocator stability, stale outputs, or replaying prior computation
{chr(10).join(purity_requirements)}
- prefer a smaller, correct kernel over a large broken rewrite
- preserve untouched code where possible and keep the edit localized

Read these repo-local skill files before editing:
{chr(10).join(f"- {path}" for path in skill_paths)}

Focused edit budget:
{budget_text}

Experiment protocol:
{chr(10).join(f"- {line}" for line in _experiment_protocol(desired_family))}

Atom-of-Thoughts operating rules:
{chr(10).join(f"- {line}" for line in AOT_GUIDANCE)}

Problem-specific guidance:
{chr(10).join(f"- {line}" for line in _problem_specific_guidance(problem["key"], desired_family, history))}
"""
    _write_prompt_artifacts(context, system_prompt.strip(), user_prompt.strip())
    return system_prompt.strip(), user_prompt.strip()


def _response_text(payload: dict[str, object]) -> str:
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct

    chunks: list[str] = []
    output = payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                text = part.get("text")
                if part_type in {"output_text", "text"} and isinstance(text, str):
                    chunks.append(text)
    return "\n".join(chunk for chunk in chunks if chunk.strip())


def _post_json(
    url: str,
    payload: dict[str, object],
    headers: dict[str, str],
) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace").strip()
        detail = body or exc.reason
        raise RuntimeError(f"http {exc.code}: {detail}") from exc


def _anthropic_response_text(payload: dict[str, object]) -> str:
    chunks: list[str] = []
    content = payload.get("content")
    if isinstance(content, list):
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and isinstance(item.get("text"), str):
                chunks.append(item["text"])
    return "\n".join(chunk for chunk in chunks if chunk.strip())


def _openrouter_response_text(payload: dict[str, object]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return ""
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            if chunks:
                return "\n".join(chunks)
    return ""


def _call_openai_responses(
    config: AppConfig,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, dict[str, object]]:
    api_key = os.environ.get(config.llm.api_key_env_var)
    if not api_key:
        raise RuntimeError(
            f"missing API key env var {config.llm.api_key_env_var} for LLM mutator"
        )

    payload = {
        "model": config.llm.model,
        "input": [
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": user_prompt}],
            },
        ],
        "max_output_tokens": config.llm.max_output_tokens,
    }
    if config.llm.reasoning_effort:
        payload["reasoning"] = {"effort": config.llm.reasoning_effort}

    response_payload = _post_json(
        config.llm.api_url,
        payload,
        {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    text = _response_text(response_payload)
    if not text.strip():
        raise RuntimeError("LLM response did not include any output text")
    return text, response_payload


def _call_anthropic_messages(
    config: AppConfig,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, dict[str, object]]:
    api_key = os.environ.get(config.llm.anthropic_api_key_env_var)
    if not api_key:
        raise RuntimeError(
            f"missing API key env var {config.llm.anthropic_api_key_env_var} for Anthropic mutator"
        )

    payload = {
        "model": config.llm.model,
        "max_tokens": config.llm.max_output_tokens,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    }
    response_payload = _post_json(
        config.llm.anthropic_api_url,
        payload,
        {
            "x-api-key": api_key,
            "anthropic-version": config.llm.anthropic_version,
            "Content-Type": "application/json",
        },
    )
    text = _anthropic_response_text(response_payload)
    if not text.strip():
        raise RuntimeError("Anthropic response did not include any text content")
    return text, response_payload


def _call_openrouter_chat(
    config: AppConfig,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, dict[str, object]]:
    api_key = os.environ.get(config.llm.openrouter_api_key_env_var)
    if not api_key:
        raise RuntimeError(
            f"missing API key env var {config.llm.openrouter_api_key_env_var} for OpenRouter mutator"
        )

    payload = {
        "model": config.llm.model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "max_tokens": config.llm.max_output_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if config.llm.openrouter_http_referer:
        headers["HTTP-Referer"] = config.llm.openrouter_http_referer
    if config.llm.openrouter_title:
        headers["X-Title"] = config.llm.openrouter_title

    response_payload = _post_json(
        config.llm.openrouter_api_url,
        payload,
        headers,
    )
    text = _openrouter_response_text(response_payload)
    if not text.strip():
        raise RuntimeError("OpenRouter response did not include any message content")
    return text, response_payload


def _call_codex_exec(
    config: AppConfig,
    system_prompt: str,
    user_prompt: str,
    *,
    workspace_dir: Path,
) -> tuple[str, dict[str, object]]:
    codex_path = shutil.which(config.llm.codex_cli)
    if not codex_path:
        raise RuntimeError(f"codex CLI '{config.llm.codex_cli}' was not found in PATH")

    orchestration_bits = [
        "You are running inside a repo-local autonomous kernel search loop.",
        "Edit ./submission.py in place inside the current isolated workspace.",
        "Modify only ./submission.py and keep all other files unchanged.",
        "Your final message should be a short plain-text note describing the single focused edit you made.",
        f"Read {_default_skill_name(None)} for the competition workflow, problem-contract rules, and current search discipline.",
        "Also follow any additional repo-local skill files explicitly referenced in the USER prompt.",
    ]
    if config.llm.codex_use_plan:
        orchestration_bits.append(
            "Use a concise internal plan before writing code, and if it helps, use helper agents for parallel analysis."
        )
        orchestration_bits.append(
            f"Keep helper-agent fanout to at most {config.llm.codex_parallel_agents}."
        )

    prompt = "\n\n".join(
        [
            f"SYSTEM:\n{system_prompt}",
            f"EXECUTION:\n{' '.join(orchestration_bits)}",
            f"USER:\n{user_prompt}",
        ]
    )

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as handle:
        output_path = Path(handle.name)

    command = [
        codex_path,
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--color",
        "never",
        "-C",
        str(workspace_dir),
        "-s",
        "workspace-write",
        "-o",
        str(output_path),
    ]
    if config.llm.codex_use_plan and config.llm.codex_parallel_agents > 1:
        command.extend(["--enable", "multi_agent"])
    if config.llm.codex_model:
        command.extend(["-m", config.llm.codex_model])
    if config.llm.codex_profile:
        command.extend(["-p", config.llm.codex_profile])
    command.append("-")

    try:
        while True:
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    cwd=str(workspace_dir),
                    text=True,
                    capture_output=True,
                    check=False,
                    timeout=config.llm.codex_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                timeout_seconds = config.llm.codex_timeout_seconds
                if timeout_seconds is None:
                    raise
                raise RuntimeError(
                    f"codex exec timed out after {timeout_seconds:g}s"
                ) from exc

            stderr = completed.stderr or ""
            stdout = completed.stdout or ""
            reset_at = _codex_usage_limit_reset_time(stderr) or _codex_usage_limit_reset_time(stdout)
            if completed.returncode != 0 and reset_at is not None:
                _write_codex_wait_artifact(workspace_dir, reset_at=reset_at, stderr=stderr or stdout)
                sleep_seconds = max((reset_at - datetime.now().astimezone()).total_seconds() + 5.0, 5.0)
                time.sleep(sleep_seconds)
                continue

            if completed.returncode != 0:
                detail = _compact_command_failure(stderr, stdout)
                raise RuntimeError(f"codex exec failed: {detail}")

            text = output_path.read_text(encoding="utf-8").strip()
            if not text:
                raise RuntimeError("codex exec did not write a final message")
            edited_submission = workspace_dir / "submission.py"
            if not edited_submission.exists():
                raise RuntimeError("codex exec did not leave a submission.py in the workspace")
            edited_text = edited_submission.read_text(encoding="utf-8")
            return text, {
                "provider": "codex_cli",
                "command": command,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "workspace_dir": str(workspace_dir),
                "edited_submission": edited_text,
            }
    finally:
        output_path.unlink(missing_ok=True)


def _compile_check(path: Path) -> None:
    py_compile.compile(str(path), doraise=True)


def _compact_reason(reason: str, limit: int = 240) -> str:
    compact = " ".join(reason.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _write_prompt_artifacts(
    context: dict[str, object],
    system_prompt: str,
    user_prompt: str,
) -> None:
    candidate_dir = context.get("candidate_dir")
    if not isinstance(candidate_dir, str) or not candidate_dir:
        return
    path = Path(candidate_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / "prompt.system.txt").write_text(system_prompt + "\n", encoding="utf-8")
    (path / "prompt.user.txt").write_text(user_prompt + "\n", encoding="utf-8")


def _write_generation_artifacts(
    context: dict[str, object],
    *,
    provider: str,
    raw_text: str,
    response_payload: dict[str, object] | None = None,
) -> None:
    candidate_dir = context.get("candidate_dir")
    if not isinstance(candidate_dir, str) or not candidate_dir:
        return
    path = Path(candidate_dir)
    path.mkdir(parents=True, exist_ok=True)
    (path / f"{provider}.raw.txt").write_text(raw_text, encoding="utf-8")
    if response_payload is not None:
        try:
            serialized = json.dumps(response_payload, indent=2, sort_keys=True)
        except TypeError:
            serialized = json.dumps({"unserializable": True}, indent=2, sort_keys=True)
        (path / f"{provider}.response.json").write_text(serialized + "\n", encoding="utf-8")


def _custom_kernel_body(source: str) -> str | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None

    if tree is not None:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "custom_kernel":
                if not node.body:
                    return ""
                segments: list[str] = []
                for stmt in node.body:
                    segment = ast.get_source_segment(source, stmt)
                    if isinstance(segment, str) and segment.strip():
                        segments.append(segment)
                if segments:
                    return "\n".join(segments)
                lines = source.splitlines()
                start = node.body[0].lineno - 1
                end_lineno = getattr(node.body[-1], "end_lineno", None) or node.body[-1].lineno
                return "\n".join(lines[start:end_lineno])

    match = CUSTOM_KERNEL_RE.search(source)
    if not match:
        return None
    return match.group("body")


def _validate_hot_path(
    source: str,
    problem_key: str,
    desired_family: str | None,
) -> None:
    if desired_family not in {"kernel_explore", "hip_explore"}:
        return
    body = _custom_kernel_body(source)
    if body is None:
        raise RuntimeError("output did not include a readable custom_kernel body")
    forbidden = {
        "mxfp4_mm": ["aiter.gemm_a4w4("],
        "moe_mxfp4": ["fused_moe("],
        "mixed_mla": ["mla_decode_fwd("],
    }
    for token in forbidden.get(problem_key, []):
        if token in body:
            family = "HIP" if desired_family == "hip_explore" else "Kernel"
            raise RuntimeError(f"hot path remained on anchor op {token} instead of {family}")
    if desired_family == "hip_explore":
        if "load_inline(" not in source and "load_inline(" not in body:
            raise RuntimeError("HIP family candidate did not use load_inline")
        if "gfx950" not in source:
            raise RuntimeError("HIP family candidate did not target gfx950")
        if problem_key == "mxfp4_mm":
            if "QuantType.per_1x32" not in source:
                raise RuntimeError("mxfp4_mm HIP candidate did not use QuantType.per_1x32")
            sanitized_body = re.sub(r"^\s*del[^\n]*$", "", body, flags=re.MULTILINE)
            if "b_shuffle" not in sanitized_body or "b_scale_sh" not in sanitized_body or "b_q" not in sanitized_body:
                raise RuntimeError("mxfp4_mm HIP candidate did not keep the live contract tensors visible in custom_kernel")
            if re.search(r"^\s*del[^\n]*\bb_shuffle\b", body, flags=re.MULTILINE):
                raise RuntimeError("mxfp4_mm HIP candidate illegally discarded required contract tensor 'b_shuffle'")
            if re.search(r"^\s*del[^\n]*\bb_scale_sh\b", body, flags=re.MULTILINE):
                raise RuntimeError("mxfp4_mm HIP candidate illegally discarded required contract tensor 'b_scale_sh'")
            if re.search(r"^\s*del[^\n]*\bb_q\b", body, flags=re.MULTILINE):
                raise RuntimeError("mxfp4_mm HIP candidate illegally discarded required contract tensor 'b_q'")
            if "shuffle=True" not in source and "shuffle=False" not in source:
                raise RuntimeError("mxfp4_mm HIP candidate did not make its quantization mode explicit")
        lower_source = source.lower()
        forbidden_purity_tokens = (
            "stream",
            "cudaevent",
            "hipevent",
            "record_event",
            "wait_event",
            "torch.cuda.synchronize",
            "hipdevicesynchronize",
            "cudaeventsynchronize",
            "usleep(",
            "sleep(",
        )
        for token in forbidden_purity_tokens:
            if token in lower_source:
                raise RuntimeError(
                    f"HIP family candidate contained forbidden purity token {token!r}"
                )
        forbidden_hip_scaffold = (
            "import triton",
            "@triton.jit",
            "triton.language",
            "triton.cdiv(",
        )
        for token in forbidden_hip_scaffold:
            if token in source:
                raise RuntimeError(
                    f"HIP family candidate still contained Triton scaffold token {token!r}"
                )


def _fallback_source(
    seed_source: str,
    seed_meta: dict[str, object],
    reason: str,
) -> str:
    meta = dict(seed_meta)
    meta["generator"] = {
        "kind": "template_fallback",
        "reason": _compact_reason(reason),
    }
    return _replace_meta(seed_source, meta) + ("" if seed_source.endswith("\n") else "\n")


def _llm_source(
    config: AppConfig,
    context: dict[str, object],
    parent_source: str,
    seed_source: str,
    seed_meta: dict[str, object],
    policy_profile: dict[str, object],
    variant: dict[str, object],
) -> str:
    system_prompt, user_prompt = _build_prompt(
        config,
        context,
        parent_source,
        seed_source,
        policy_profile,
        variant,
    )
    raw_text, response_payload = _call_openai_responses(config, system_prompt, user_prompt)
    _write_generation_artifacts(context, provider="openai", raw_text=raw_text, response_payload=response_payload)
    meta = dict(seed_meta)
    meta["generator"] = {
        "kind": "llm",
        "provider": config.llm.provider,
        "model": config.llm.model,
    }
    candidate_source = _canonicalize_source(raw_text, seed_source, meta)
    if "def custom_kernel" not in candidate_source:
        raise RuntimeError("LLM output did not define custom_kernel")
    desired_family = context.get("desired_family")
    if not isinstance(desired_family, str):
        desired_family = None
    _validate_hot_path(candidate_source, str(context["problem"]["key"]), desired_family)
    return candidate_source


def _anthropic_source(
    config: AppConfig,
    context: dict[str, object],
    parent_source: str,
    seed_source: str,
    seed_meta: dict[str, object],
    policy_profile: dict[str, object],
    variant: dict[str, object],
) -> str:
    system_prompt, user_prompt = _build_prompt(
        config,
        context,
        parent_source,
        seed_source,
        policy_profile,
        variant,
    )
    raw_text, response_payload = _call_anthropic_messages(config, system_prompt, user_prompt)
    _write_generation_artifacts(context, provider="anthropic", raw_text=raw_text, response_payload=response_payload)
    meta = dict(seed_meta)
    meta["generator"] = {
        "kind": "llm",
        "provider": "anthropic",
        "model": config.llm.model,
    }
    candidate_source = _canonicalize_source(raw_text, seed_source, meta)
    if "def custom_kernel" not in candidate_source:
        raise RuntimeError("Anthropic output did not define custom_kernel")
    desired_family = context.get("desired_family")
    if not isinstance(desired_family, str):
        desired_family = None
    _validate_hot_path(candidate_source, str(context["problem"]["key"]), desired_family)
    return candidate_source


def _openrouter_source(
    config: AppConfig,
    context: dict[str, object],
    parent_source: str,
    seed_source: str,
    seed_meta: dict[str, object],
    policy_profile: dict[str, object],
    variant: dict[str, object],
) -> str:
    system_prompt, user_prompt = _build_prompt(
        config,
        context,
        parent_source,
        seed_source,
        policy_profile,
        variant,
    )
    raw_text, response_payload = _call_openrouter_chat(config, system_prompt, user_prompt)
    _write_generation_artifacts(context, provider="openrouter", raw_text=raw_text, response_payload=response_payload)
    meta = dict(seed_meta)
    meta["generator"] = {
        "kind": "llm",
        "provider": "openrouter",
        "model": config.llm.model,
    }
    candidate_source = _canonicalize_source(raw_text, seed_source, meta)
    if "def custom_kernel" not in candidate_source:
        raise RuntimeError("OpenRouter output did not define custom_kernel")
    desired_family = context.get("desired_family")
    if not isinstance(desired_family, str):
        desired_family = None
    _validate_hot_path(candidate_source, str(context["problem"]["key"]), desired_family)
    return candidate_source


def _codex_source(
    config: AppConfig,
    context: dict[str, object],
    parent_source: str,
    seed_source: str,
    seed_meta: dict[str, object],
    policy_profile: dict[str, object],
    variant: dict[str, object],
) -> str:
    system_prompt, user_prompt = _build_prompt(
        config,
        context,
        parent_source,
        seed_source,
        policy_profile,
        variant,
        workspace_edit=True,
    )
    candidate_dir_raw = context.get("candidate_dir")
    if not isinstance(candidate_dir_raw, str) or not candidate_dir_raw:
        raise RuntimeError("context did not include candidate_dir for codex workspace edit mode")
    workspace_dir = Path(candidate_dir_raw)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    workspace_submission = workspace_dir / "submission.py"
    desired_family = context.get("desired_family")
    if not isinstance(desired_family, str):
        desired_family = None
    base_source = seed_source if desired_family == "hip_explore" else parent_source
    workspace_submission.write_text(base_source, encoding="utf-8")

    raw_text, response_payload = _call_codex_exec(
        config,
        system_prompt,
        user_prompt,
        workspace_dir=workspace_dir,
    )
    _write_generation_artifacts(context, provider="codex_cli", raw_text=raw_text, response_payload=response_payload)
    meta = dict(seed_meta)
    meta["generator"] = {
        "kind": "llm",
        "provider": "codex_cli",
        "model": config.llm.codex_model or "(codex default)",
        "use_plan": config.llm.codex_use_plan,
        "parallel_agents": config.llm.codex_parallel_agents,
        "mode": "workspace_edit",
    }
    edited_text = response_payload.get("edited_submission")
    if not isinstance(edited_text, str) or not edited_text.strip():
        raise RuntimeError("codex workspace edit did not produce readable submission.py contents")
    candidate_source = _canonicalize_source(edited_text, seed_source, meta)
    if "def custom_kernel" not in candidate_source:
        raise RuntimeError("Codex output did not define custom_kernel")
    _validate_hot_path(candidate_source, str(context["problem"]["key"]), desired_family)
    return candidate_source


def _generate_source(
    config: AppConfig,
    context: dict[str, object],
    parent_path: Path,
) -> tuple[str, dict[str, object], dict[str, object]]:
    problem_key = str(context["problem"]["key"])
    attempt = candidate_attempt(context)
    context["attempt"] = attempt
    history = history_entries(context)
    parent_meta = load_parent_meta(parent_path)
    desired_family = context.get("desired_family")
    if not isinstance(desired_family, str):
        desired_family = None

    policy_profile = choose_policy_profile(
        problem_key,
        attempt,
        parent_meta,
        history,
        desired_family=desired_family,
    )
    variant_index, variant = choose_variant(
        problem_key,
        attempt,
        parent_meta,
        history,
        policy_profile=policy_profile,
        desired_family=desired_family,
    )
    seed_source = render_submission(
        problem_key,
        variant_index,
        variant,
        context,
        attempt,
        policy_profile=policy_profile,
    )
    seed_meta = _extract_meta(seed_source)
    parent_source = parent_path.read_text(encoding="utf-8")
    _write_experiment_plan(context, policy_profile=policy_profile, variant=variant)

    if not config.llm.enabled:
        return _fallback_source(seed_source, seed_meta, "llm_disabled"), policy_profile, variant

    providers_to_try: list[str]
    provider = config.llm.provider
    if provider == "auto":
        providers_to_try = []
        if os.environ.get(config.llm.api_key_env_var):
            providers_to_try.append("openai")
        providers_to_try.append("codex_cli")
    else:
        providers_to_try = [provider]

    last_error: Exception | None = None
    for active_provider in providers_to_try:
        try:
            if active_provider == "openai":
                source = _llm_source(
                    config,
                    context,
                    parent_source,
                    seed_source,
                    seed_meta,
                    policy_profile,
                    variant,
                )
                return source, policy_profile, variant
            if active_provider == "anthropic":
                source = _anthropic_source(
                    config,
                    context,
                    parent_source,
                    seed_source,
                    seed_meta,
                    policy_profile,
                    variant,
                )
                return source, policy_profile, variant
            if active_provider == "openrouter":
                source = _openrouter_source(
                    config,
                    context,
                    parent_source,
                    seed_source,
                    seed_meta,
                    policy_profile,
                    variant,
                )
                return source, policy_profile, variant
            if active_provider == "codex_cli":
                source = _codex_source(
                    config,
                    context,
                    parent_source,
                    seed_source,
                    seed_meta,
                    policy_profile,
                    variant,
                )
                return source, policy_profile, variant
            raise RuntimeError(f"unsupported llm provider: {active_provider}")
        except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            continue

    if not config.llm.fallback_to_seed and last_error is not None:
        raise last_error
    reason = str(last_error) if last_error is not None else f"unsupported_provider:{provider}"
    return _fallback_source(seed_source, seed_meta, reason), policy_profile, variant


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="agent_loop.toml")
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    context = load_context(Path(args.context))
    context["candidate_dir"] = str(Path(args.context).resolve().parent)
    source, _, _ = _generate_source(config, context, Path(args.parent))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(source, encoding="utf-8")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(source)
        temp_path = Path(handle.name)
    try:
        _compile_check(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
