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
from .triton_mutator import (
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


def _problem_specific_guidance(
    problem_key: str,
    desired_family: str | None,
    history: list[dict[str, object]],
) -> list[str]:
    guidance: list[str] = []
    if desired_family == "triton_explore":
        guidance.append(
            "The real hot path in custom_kernel must execute Triton-based logic. Do not leave Triton as dead scaffold while delegating compute to an AITER anchor."
        )

    repeated_failure = _repeated_failure_hint(history)
    if repeated_failure:
        guidance.append(repeated_failure)

    if problem_key == "mxfp4_mm":
        guidance.extend(
            [
                "Preserve the shuffled MXFP4 contract exactly: quantize A with shuffle=True and consume B_shuffle/B_scale_sh consistently.",
                "If the shuffled-B interpretation is still unstable, prefer a correctness-first Triton seed that requantizes both A and B from the original bf16 tensors, dequantizes them plainly, and only then runs Triton matmul.",
                "Do not change both semantic layout and tiling in the same candidate when correctness is failing repeatedly; fix contract/dataflow first, then tune tiles.",
                "Prefer a minimal Triton matmul path that is actually called from custom_kernel over a larger submission that still routes the hot path through aiter.gemm_a4w4.",
            ]
        )
    elif problem_key == "moe_mxfp4":
        guidance.extend(
            [
                "If correctness is unstable, keep the routing/topk contract fixed and only rewrite one expert-compute stage at a time.",
                "Avoid leaving the hot path on fused_moe while presenting Triton as unused side code.",
            ]
        )
    elif problem_key == "mixed_mla":
        guidance.extend(
            [
                "Bias toward latency-first rewrites for q_seq_len=1 decode, and keep the hot path on actual Triton decode/attention code rather than mla_decode_fwd.",
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

    if workspace_edit:
        system_prompt = (
            "You are an expert AMD MI355X Triton kernel engineer working on a real competition submission. "
            "Edit the existing submission.py file in place inside the current isolated workspace. "
            "Preserve the live kernel contract, keep the two #!POPCORN header lines, define custom_kernel(data), "
            "and do not use hidden cross-call caches or benchmark-cheating tricks. Favor correctness first, then speed. "
            "Use Codex helper agents only for internal analysis, then finish with a short plain-text summary of the one focused edit you made."
        )
        delivery_requirements = [
            "- edit the existing submission.py in place",
            "- do not create alternate code files or markdown fences",
            "- keep the final change localized and preserve untouched code where possible",
        ]
    else:
        system_prompt = (
            "You are an expert AMD MI355X Triton kernel engineer working on a real competition submission. "
            "Generate exactly one Python submission file. Return only raw Python source, no markdown fences. "
            "Preserve the live kernel contract, keep the two #!POPCORN header lines, define custom_kernel(data), "
            "and do not use hidden cross-call caches or benchmark-cheating tricks. Favor correctness first, then speed. "
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

Current parent submission:
{parent_source}

Seed Triton submission to improve:
{seed_source}

Write a new full submission.py candidate for this problem.
Requirements:
{chr(10).join(delivery_requirements)}
- preserve the exact data contract for the problem
- include Triton code when optimizing Triton paths
- do not add fake speedups, persistent benchmark caches, or hidden state across evaluations
- prefer a smaller, correct kernel over a large broken rewrite
- preserve untouched code where possible and keep the edit localized

Focused edit budget:
{budget_text}

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
        "Use $amd-triton-speedrun for the competition workflow, problem-contract rules, and current search discipline.",
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
    if desired_family != "triton_explore":
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
            raise RuntimeError(f"hot path remained on anchor op {token} instead of Triton")


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
    workspace_submission.write_text(parent_source, encoding="utf-8")

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
    desired_family = context.get("desired_family")
    if not isinstance(desired_family, str):
        desired_family = None
    _validate_hot_path(candidate_source, str(context["problem"]["key"]), desired_family)
    return candidate_source


def _generate_source(
    config: AppConfig,
    context: dict[str, object],
    parent_path: Path,
) -> tuple[str, dict[str, object], dict[str, object]]:
    problem_key = str(context["problem"]["key"])
    attempt = candidate_attempt(context)
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

    if not config.llm.fallback_to_triton and last_error is not None:
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
