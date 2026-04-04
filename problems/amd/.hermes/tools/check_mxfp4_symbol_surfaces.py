#!/usr/bin/env python3
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


DECL_RE = re.compile(r"^\s*void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)
MODULE_CALL_RE = re.compile(r"(?:\bmod\b|_module\(\))\.([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _load_literal_assign(tree: ast.Module, name: str) -> object:
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise RuntimeError(f"could not find literal assignment for {name}")


def _eval_literal_expr(tree: ast.Module, expr: ast.AST) -> object:
    if isinstance(expr, ast.Name):
        return _load_literal_assign(tree, expr.id)
    return ast.literal_eval(expr)


def _load_export_list(tree: ast.Module) -> list[str]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id != "load_inline":
            continue
        if isinstance(func, ast.Attribute) and func.attr != "load_inline":
            continue
        for keyword in node.keywords:
            if keyword.arg == "functions":
                return _eval_literal_expr(tree, keyword.value)
    raise RuntimeError("could not find load_inline(..., functions=[...]) export list")


def _sorted_names(names: set[str]) -> list[str]:
    return sorted(names, key=lambda value: (value.count("_"), value))


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "/Users/v/reference-kernels/problems/amd/fp8-mm/submission.py"
    )
    source = target.read_text()
    tree = ast.parse(source, filename=str(target))

    cpp_wrapper = _load_literal_assign(tree, "CPP_WRAPPER")
    hip_src = _load_literal_assign(tree, "HIP_SRC")
    exports = set(_load_export_list(tree))

    wrapper_decls = set(DECL_RE.findall(cpp_wrapper))
    hip_defs = set(DECL_RE.findall(hip_src))
    module_calls = set(MODULE_CALL_RE.findall(source))

    hard_failures: list[tuple[str, set[str]]] = []
    warnings: list[tuple[str, set[str]]] = []

    if exports - wrapper_decls:
        hard_failures.append(("exports missing from CPP_WRAPPER", exports - wrapper_decls))
    if exports - hip_defs:
        hard_failures.append(("exports missing from HIP_SRC definitions", exports - hip_defs))
    if module_calls - exports:
        hard_failures.append(("Python module calls missing from load_inline export list", module_calls - exports))

    if wrapper_decls - exports:
        warnings.append(("CPP_WRAPPER declarations not exported", wrapper_decls - exports))
    if module_calls - wrapper_decls:
        warnings.append(("Python module calls missing from CPP_WRAPPER", module_calls - wrapper_decls))
    if module_calls - hip_defs:
        warnings.append(("Python module calls missing from HIP_SRC definitions", module_calls - hip_defs))

    print(f"checked: {target}")
    print(f"exports={len(exports)} wrapper_decls={len(wrapper_decls)} hip_defs={len(hip_defs)} module_calls={len(module_calls)}")

    for label, names in warnings:
        if names:
            print(f"warning: {label}: {', '.join(_sorted_names(names))}")

    if hard_failures:
        for label, names in hard_failures:
            print(f"error: {label}: {', '.join(_sorted_names(names))}")
        return 1

    print("ok: exported symbol surfaces are internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
