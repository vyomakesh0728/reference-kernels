from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import py_compile
import re
import tempfile
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
        return META_RE.sub(meta_line, source, count=1)

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
    return text.strip()


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


def _build_prompt(
    config: AppConfig,
    context: dict[str, object],
    parent_source: str,
    seed_source: str,
    policy_profile: dict[str, object],
    variant: dict[str, object],
) -> tuple[str, str]:
    problem = context["problem"]
    history = history_entries(context)
    knowledge_path = Path(str(context["knowledge_path"]))
    knowledge_text = knowledge_path.read_text(encoding="utf-8") if knowledge_path.exists() else "{}"

    system_prompt = (
        "You are an expert AMD MI355X Triton kernel engineer working on a real competition submission. "
        "Generate exactly one Python submission file. Return only raw Python source, no markdown fences. "
        "Preserve the live kernel contract, keep the two #!POPCORN header lines, define custom_kernel(data), "
        "and do not use hidden cross-call caches or benchmark-cheating tricks. Favor correctness first, then speed."
    )

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
- return only Python source
- keep the #!POPCORN header lines valid
- preserve the exact data contract for the problem
- include Triton code when optimizing Triton paths
- do not add fake speedups, persistent benchmark caches, or hidden state across evaluations
- prefer a smaller, correct kernel over a large broken rewrite
"""
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

    request = urllib.request.Request(
        config.llm.api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        response_payload = json.loads(response.read().decode("utf-8"))
    text = _response_text(response_payload)
    if not text.strip():
        raise RuntimeError("LLM response did not include any output text")
    return text, response_payload


def _compile_check(path: Path) -> None:
    py_compile.compile(str(path), doraise=True)


def _fallback_source(
    seed_source: str,
    seed_meta: dict[str, object],
    reason: str,
) -> str:
    meta = dict(seed_meta)
    meta["generator"] = {
        "kind": "template_fallback",
        "reason": reason,
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
    raw_text, _ = _call_openai_responses(config, system_prompt, user_prompt)
    meta = dict(seed_meta)
    meta["generator"] = {
        "kind": "llm",
        "provider": config.llm.provider,
        "model": config.llm.model,
    }
    candidate_source = _canonicalize_source(raw_text, seed_source, meta)
    if "def custom_kernel" not in candidate_source:
        raise RuntimeError("LLM output did not define custom_kernel")
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

    if config.llm.provider != "openai":
        if config.llm.fallback_to_triton:
            return _fallback_source(seed_source, seed_meta, f"unsupported_provider:{config.llm.provider}"), policy_profile, variant
        raise RuntimeError(f"unsupported llm provider: {config.llm.provider}")

    try:
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
    except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
        if not config.llm.fallback_to_triton:
            raise
        return _fallback_source(seed_source, seed_meta, str(exc)), policy_profile, variant


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="agent_loop.toml")
    parser.add_argument("--parent", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    context = load_context(Path(args.context))
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
