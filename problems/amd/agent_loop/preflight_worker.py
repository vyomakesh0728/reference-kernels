from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass, field
import importlib.util
import inspect
import json
import os
from pathlib import Path
import py_compile
import hashlib
import sys
from typing import Any

from .config import AppConfig, load_config


PROBLEM_DIR_BY_KEY = {
    "mxfp4_mm": Path("amd_202602/mxfp4-mm"),
    "moe_mxfp4": Path("amd_202602/moe-mxfp4"),
    "mixed_mla": Path("amd_202602/mixed-mla"),
}


@dataclass
class PreflightCheck:
    name: str
    status: str
    detail: str


@dataclass
class PreflightReport:
    status: str
    runtime: str
    purity_status: str
    checks: list[PreflightCheck] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "runtime": self.runtime,
            "purity_status": self.purity_status,
            "checks": [asdict(item) for item in self.checks],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreflightReport":
        checks = [
            PreflightCheck(
                name=str(item.get("name", "")),
                status=str(item.get("status", "")),
                detail=str(item.get("detail", "")),
            )
            for item in payload.get("checks", [])
            if isinstance(item, dict)
        ]
        return cls(
            status=str(payload.get("status", "pending")),
            runtime=str(payload.get("runtime", "unknown")),
            purity_status=str(payload.get("purity_status", "pending")),
            checks=checks,
            notes=[str(item) for item in payload.get("notes", [])],
        )


class PurityVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.issues: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"data_ptr", "storage", "storage_offset"}:
            self.issues.append(f"python-level pointer/storage access via .{func.attr}() at line {node.lineno}")
        if isinstance(func, ast.Name) and func.id in {"cache", "lru_cache"}:
            self.issues.append(f"cache decorator/helper `{func.id}` referenced at line {node.lineno}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr in {"_tensor_cache_key", "illegal_cache"}:
            self.issues.append(f"suspicious cache helper `{node.attr}` referenced at line {node.lineno}")
        self.generic_visit(node)


def run_host_preflight(
    *,
    repo_root: Path,
    config_path: Path,
    problem_key: str,
    source_path: Path,
    compile_jit: bool,
    runtime_label: str,
    static_only: bool = False,
) -> PreflightReport:
    config = load_config(config_path)
    report = PreflightReport(status="ok", runtime=runtime_label, purity_status="ok")
    source_text = source_path.read_text(encoding="utf-8")

    try:
        py_compile.compile(str(source_path), doraise=True)
        report.checks.append(PreflightCheck("py_compile", "ok", "syntax check passed"))
    except py_compile.PyCompileError as exc:
        report.checks.append(PreflightCheck("py_compile", "fail", str(exc)))
        report.status = "fail"
        report.purity_status = "fail"
        return report

    try:
        module_ast = ast.parse(source_text, filename=str(source_path))
        report.checks.append(PreflightCheck("ast_parse", "ok", "AST parse passed"))
    except SyntaxError as exc:
        report.checks.append(PreflightCheck("ast_parse", "fail", f"{exc.msg} at line {exc.lineno}"))
        report.status = "fail"
        report.purity_status = "fail"
        return report

    visitor = PurityVisitor()
    visitor.visit(module_ast)
    if visitor.issues:
        report.checks.append(
            PreflightCheck("purity_scan", "fail", "; ".join(visitor.issues))
        )
        report.status = "fail"
        report.purity_status = "fail"
        return report
    report.checks.append(PreflightCheck("purity_scan", "ok", "no python-level pointer/cache replay patterns found"))

    custom_kernel_fn = None
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == "custom_kernel":
            custom_kernel_fn = node
            break
    if custom_kernel_fn is None:
        report.checks.append(PreflightCheck("custom_kernel", "fail", "custom_kernel(data) is missing"))
        report.status = "fail"
        return report
    if len(custom_kernel_fn.args.args) != 1:
        report.checks.append(PreflightCheck("custom_kernel", "fail", "custom_kernel must take exactly one argument"))
        report.status = "fail"
        return report
    report.checks.append(PreflightCheck("custom_kernel", "ok", "custom_kernel(data) found"))

    gate_checks = _check_mxfp4_shape_gates(module_ast) if problem_key == "mxfp4_mm" else []
    for check in gate_checks:
        report.checks.append(check)
        if check.status == "fail":
            report.status = "fail"
            return report

    if static_only:
        report.checks.append(
            PreflightCheck(
                "module_import",
                "skip",
                "static-only preflight skipped import/JIT; remote cluster will validate runtime dependencies",
            )
        )
        report.notes.append("static-only preflight mode enabled")
        report.status = "warn"
        return report

    import_report = _import_submission(
        config=config,
        problem_key=problem_key,
        source_path=source_path,
        compile_jit=compile_jit,
    )
    report.checks.extend(import_report.checks)
    if import_report.status != "ok":
        report.status = import_report.status
    if import_report.purity_status != "ok":
        report.purity_status = import_report.purity_status
    report.notes.extend(import_report.notes)
    return report


def _problem_dir(config: AppConfig, problem_key: str) -> Path:
    try:
        relative = PROBLEM_DIR_BY_KEY[problem_key]
    except KeyError as exc:
        raise KeyError(f"no preflight problem dir mapping for {problem_key!r}") from exc
    return (config.repo_root.parent / relative).resolve()


def _import_submission(
    *,
    config: AppConfig,
    problem_key: str,
    source_path: Path,
    compile_jit: bool,
) -> PreflightReport:
    report = PreflightReport(status="ok", runtime="python", purity_status="ok")
    problem_dir = _problem_dir(config, problem_key)
    repo_root = config.repo_root
    sys_path_backup = list(sys.path)
    env_arch = os.environ.get("PYTORCH_ROCM_ARCH")
    digest = hashlib.sha1(str(source_path.resolve()).encode("utf-8")).hexdigest()[:10]
    module_name = f"preflight_candidate_{source_path.stem}_{digest}"
    try:
        sys.path.insert(0, str(problem_dir))
        sys.path.insert(0, str(repo_root))
        sys.path.insert(0, str(source_path.parent))
        os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx950")
        spec = importlib.util.spec_from_file_location(module_name, source_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to construct import spec for submission")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        report.checks.append(PreflightCheck("module_import", "ok", "submission import succeeded"))
        signature = inspect.signature(module.custom_kernel)
        if len(signature.parameters) != 1:
            raise RuntimeError("custom_kernel signature drifted at import time")
        if compile_jit and hasattr(module, "_module"):
            module._module()
            report.checks.append(PreflightCheck("jit_build", "ok", "_module() JIT build succeeded"))
        elif compile_jit:
            report.checks.append(PreflightCheck("jit_build", "warn", "_module() helper missing; skipped JIT build"))
        else:
            report.checks.append(PreflightCheck("jit_build", "skip", "JIT build skipped"))
    except Exception as exc:  # noqa: BLE001
        report.checks.append(PreflightCheck("module_import", "fail", str(exc)))
        report.status = "fail"
    finally:
        sys.modules.pop(module_name, None)
        sys.path[:] = sys_path_backup
        if env_arch is None:
            os.environ.pop("PYTORCH_ROCM_ARCH", None)
        else:
            os.environ["PYTORCH_ROCM_ARCH"] = env_arch
    return report


def _check_mxfp4_shape_gates(module_ast: ast.AST) -> list[PreflightCheck]:
    aliases: set[str] = set()
    gate_hits = {
        "wide_lane_gate": False,
        "m16_lane_gate": False,
        "thin_lane_gate": False,
    }

    for node in ast.walk(module_ast):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if _is_m_dim_expr(node.value, aliases):
                aliases.add(node.targets[0].id)
            continue
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not _is_m_dim_expr(left, aliases):
            continue
        if len(node.ops) != 1 or len(node.comparators) != 1:
            continue
        op = node.ops[0]
        right = node.comparators[0]
        if isinstance(op, ast.GtE) and _int_literal(right) == 32:
            gate_hits["wide_lane_gate"] = True
        elif isinstance(op, ast.Eq) and _int_literal(right) == 16:
            gate_hits["m16_lane_gate"] = True
        elif isinstance(op, ast.In) and _literal_int_set(right) == {4, 8}:
            gate_hits["thin_lane_gate"] = True

    details = {
        "wide_lane_gate": "expected an m-dimension gate equivalent to `a.shape[0] >= 32`",
        "m16_lane_gate": "expected an m-dimension gate equivalent to `a.shape[0] == 16`",
        "thin_lane_gate": "expected an m-dimension gate equivalent to `a.shape[0] in (4, 8)`",
    }
    return [
        PreflightCheck(name, "ok" if hit else "fail", "shape gate detected" if hit else details[name])
        for name, hit in gate_hits.items()
    ]


def _is_m_dim_expr(node: ast.AST, aliases: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in aliases
    if not isinstance(node, ast.Subscript):
        return False
    value = node.value
    if not isinstance(value, ast.Attribute) or value.attr != "shape":
        return False
    if not isinstance(value.value, ast.Name) or value.value.id != "a":
        return False
    return _int_literal(node.slice) == 0


def _int_literal(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    return None


def _literal_int_set(node: ast.AST) -> set[int] | None:
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return None
    values = {_int_literal(item) for item in node.elts}
    if None in values:
        return None
    return {int(item) for item in values if item is not None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent_loop.preflight_worker")
    parser.add_argument("--config", default="agent_loop.toml")
    parser.add_argument("--problem", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--compile-jit", action="store_true")
    parser.add_argument("--static-only", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    report = run_host_preflight(
        repo_root=config.repo_root,
        config_path=config_path,
        problem_key=args.problem,
        source_path=Path(args.source).expanduser().resolve(),
        compile_jit=bool(args.compile_jit),
        runtime_label="container",
        static_only=bool(args.static_only),
    )
    print(json.dumps(report.to_dict(), sort_keys=True))
    return 0 if report.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
