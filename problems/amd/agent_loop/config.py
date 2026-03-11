from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class WorkspaceConfig:
    root: Path
    popcorn_cli: str
    api_url: str | None
    default_gpu: str
    default_mode: str
    leaderboard_reference_seconds: float | None
    leaderboard_timeout_seconds: float | None
    promote_on_improve: bool
    improvement_epsilon: float


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool
    provider: str
    model: str
    api_url: str
    api_key_env_var: str
    anthropic_api_url: str
    anthropic_api_key_env_var: str
    anthropic_version: str
    openrouter_api_url: str
    openrouter_api_key_env_var: str
    openrouter_http_referer: str | None
    openrouter_title: str | None
    reasoning_effort: str
    max_output_tokens: int
    fallback_to_triton: bool
    codex_cli: str
    codex_model: str | None
    codex_profile: str | None
    codex_sandbox: str
    codex_use_plan: bool
    codex_parallel_agents: int


@dataclass(frozen=True)
class ProblemConfig:
    key: str
    submission_path: Path
    leaderboard: str
    gpu: str
    mode: str
    goal: str
    objective: str
    mutator_command: str | None

    @property
    def lower_is_better(self) -> bool:
        return self.goal == "min"


@dataclass(frozen=True)
class AppConfig:
    config_path: Path
    repo_root: Path
    workspace: WorkspaceConfig
    llm: LLMConfig
    problems: dict[str, ProblemConfig]

    def require_problem(self, key: str) -> ProblemConfig:
        try:
            return self.problems[key]
        except KeyError as exc:
            known = ", ".join(sorted(self.problems))
            raise KeyError(f"unknown problem '{key}' (known: {known})") from exc


def _require_table(data: dict, key: str) -> dict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"missing [{key}] table in config")
    return value


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    repo_root = config_path.parent
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))

    workspace_raw = _require_table(raw, "workspace")
    llm_raw = _require_table(raw, "llm")
    problems_raw = _require_table(raw, "problems")

    workspace = WorkspaceConfig(
        root=(repo_root / workspace_raw.get("root", ".agent-loop")).resolve(),
        popcorn_cli=str(workspace_raw.get("popcorn_cli", "popcorn-cli")),
        api_url=str(workspace_raw["api_url"]) if workspace_raw.get("api_url") else None,
        default_gpu=str(workspace_raw.get("default_gpu", "MI300")),
        default_mode=str(workspace_raw.get("default_mode", "benchmark")),
        leaderboard_reference_seconds=(
            float(workspace_raw["leaderboard_reference_seconds"])
            if workspace_raw.get("leaderboard_reference_seconds") is not None
            else None
        ),
        leaderboard_timeout_seconds=(
            float(workspace_raw["leaderboard_timeout_seconds"])
            if workspace_raw.get("leaderboard_timeout_seconds") is not None
            else None
        ),
        promote_on_improve=bool(workspace_raw.get("promote_on_improve", True)),
        improvement_epsilon=float(workspace_raw.get("improvement_epsilon", 0.0)),
    )

    llm = LLMConfig(
        enabled=bool(llm_raw.get("enabled", False)),
        provider=str(llm_raw.get("provider", "auto")),
        model=str(llm_raw.get("model", "gpt-5-mini")),
        api_url=str(llm_raw.get("api_url", "https://api.openai.com/v1/responses")),
        api_key_env_var=str(llm_raw.get("api_key_env_var", "OPENAI_API_KEY")),
        anthropic_api_url=str(
            llm_raw.get("anthropic_api_url", "https://api.anthropic.com/v1/messages")
        ),
        anthropic_api_key_env_var=str(
            llm_raw.get("anthropic_api_key_env_var", "ANTHROPIC_API_KEY")
        ),
        anthropic_version=str(llm_raw.get("anthropic_version", "2023-06-01")),
        openrouter_api_url=str(
            llm_raw.get("openrouter_api_url", "https://openrouter.ai/api/v1/chat/completions")
        ),
        openrouter_api_key_env_var=str(
            llm_raw.get("openrouter_api_key_env_var", "OPENROUTER_API_KEY")
        ),
        openrouter_http_referer=(
            str(llm_raw["openrouter_http_referer"])
            if llm_raw.get("openrouter_http_referer")
            else None
        ),
        openrouter_title=(
            str(llm_raw["openrouter_title"])
            if llm_raw.get("openrouter_title")
            else None
        ),
        reasoning_effort=str(llm_raw.get("reasoning_effort", "medium")),
        max_output_tokens=int(llm_raw.get("max_output_tokens", 12000)),
        fallback_to_triton=bool(llm_raw.get("fallback_to_triton", True)),
        codex_cli=str(llm_raw.get("codex_cli", "codex")),
        codex_model=(
            str(llm_raw["codex_model"]) if llm_raw.get("codex_model") else None
        ),
        codex_profile=(
            str(llm_raw["codex_profile"]) if llm_raw.get("codex_profile") else None
        ),
        codex_sandbox=str(llm_raw.get("codex_sandbox", "read-only")),
        codex_use_plan=bool(llm_raw.get("codex_use_plan", True)),
        codex_parallel_agents=int(llm_raw.get("codex_parallel_agents", 3)),
    )

    problems: dict[str, ProblemConfig] = {}
    for key, value in problems_raw.items():
        if not isinstance(value, dict):
            raise ValueError(f"[problems.{key}] must be a table")
        submission_raw = value.get("submission_path")
        leaderboard = value.get("leaderboard")
        if not submission_raw or not leaderboard:
            raise ValueError(
                f"[problems.{key}] must define submission_path and leaderboard"
            )
        problems[key] = ProblemConfig(
            key=key,
            submission_path=(repo_root / str(submission_raw)).resolve(),
            leaderboard=str(leaderboard),
            gpu=str(value.get("gpu", workspace.default_gpu)),
            mode=str(value.get("mode", workspace.default_mode)),
            goal=str(value.get("goal", "min")),
            objective=str(value.get("objective", "geom_mean_ns")),
            mutator_command=(
                str(value["mutator_command"]) if value.get("mutator_command") else None
            ),
        )

    return AppConfig(
        config_path=config_path,
        repo_root=repo_root,
        workspace=workspace,
        llm=llm,
        problems=problems,
    )
