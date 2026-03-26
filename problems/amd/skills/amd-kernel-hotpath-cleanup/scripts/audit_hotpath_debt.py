#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import re
from collections import defaultdict, deque
from pathlib import Path


SERVED_PHASES = {
    "path_direct_m32_multiples32",
    "path_direct_m16",
    "path_direct_m4_m8_thin",
}

FALLBACK_PHASE_PREFIXES = (
    "path_fallback_",
    "path_default_",
    "path_mfma_medium",
    "path_medium_",
)

AITER_TOKENS = ("aiter", "fp4_utils", "shuffle_weight", "QuantType")


def source_segment(text: str, node: ast.AST) -> str:
    segment = ast.get_source_segment(text, node)
    return segment or ""


def iter_calls(node: ast.AST) -> list[str]:
    names: list[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            fn = sub.func
            if isinstance(fn, ast.Name):
                names.append(fn.id)
    return names


def iter_names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            out.add(sub.id)
    return out


def first_phase_labels(node: ast.AST) -> set[str]:
    labels: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        if not isinstance(sub.func, ast.Name) or sub.func.id != "_phase":
            continue
        if not sub.args:
            continue
        arg0 = sub.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            labels.add(arg0.value)
    return labels


def reachability(roots: set[str], graph: dict[str, set[str]]) -> set[str]:
    seen: set[str] = set()
    q = deque(sorted(roots))
    while q:
        name = q.popleft()
        if name in seen:
            continue
        seen.add(name)
        for nxt in graph.get(name, set()):
            if nxt not in seen:
                q.append(nxt)
    return seen


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit AMD kernel hot-path debt.")
    parser.add_argument("--submission", required=True, help="Path to submission.py")
    args = parser.parse_args()

    path = Path(args.submission)
    text = path.read_text()
    tree = ast.parse(text)

    imported: dict[str, tuple[str, int]] = {}
    import_order: list[str] = []
    func_nodes: dict[str, list[ast.FunctionDef]] = defaultdict(list)
    top_level_funcs: dict[str, ast.FunctionDef] = {}

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split(".")[0]
                imported[name] = (alias.name, node.lineno)
                import_order.append(name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                name = alias.asname or alias.name
                imported[name] = (f"{module}:{alias.name}", node.lineno)
                import_order.append(name)
        elif isinstance(node, ast.FunctionDef):
            func_nodes[node.name].append(node)
            top_level_funcs[node.name] = node

    used_names = iter_names(tree)
    unused_imports = [
        (name, imported[name][0], imported[name][1])
        for name in import_order
        if name in imported and name not in used_names
    ]

    duplicate_functions = {
        name: [node.lineno for node in nodes]
        for name, nodes in func_nodes.items()
        if len(nodes) > 1
    }

    graph: dict[str, set[str]] = {}
    for name, node in top_level_funcs.items():
        graph[name] = {callee for callee in iter_calls(node) if callee in top_level_funcs}

    custom = top_level_funcs.get("custom_kernel")
    served_roots: set[str] = set()
    fallback_roots: set[str] = set()
    if custom is not None:
        for node in custom.body:
            if not isinstance(node, ast.If):
                continue
            labels = first_phase_labels(node)
            branch_calls = {callee for callee in iter_calls(node) if callee in top_level_funcs}
            if labels & SERVED_PHASES:
                served_roots |= branch_calls
            elif any(any(label.startswith(prefix) for prefix in FALLBACK_PHASE_PREFIXES) for label in labels):
                fallback_roots |= branch_calls
            elif "pre_reference_oracle_inputs" in labels or "regime_selected" in labels:
                fallback_roots |= branch_calls

    served_funcs = reachability(served_roots, graph)
    fallback_funcs = reachability(fallback_roots, graph)

    definition_only: list[tuple[str, int]] = []
    for name, node in top_level_funcs.items():
        if name == "custom_kernel":
            continue
        refs = len(re.findall(rf"\b{re.escape(name)}\b", text))
        if refs == 1:
            definition_only.append((name, node.lineno))

    fallback_only = sorted((fallback_funcs - served_funcs) & set(top_level_funcs))
    served_only = sorted((served_funcs - fallback_funcs) & set(top_level_funcs))

    served_aiter_refs: list[tuple[str, int, list[str]]] = []
    for name in sorted(served_funcs):
        node = top_level_funcs[name]
        segment = source_segment(text, node)
        hits = [token for token in AITER_TOKENS if token in segment]
        if hits:
            served_aiter_refs.append((name, node.lineno, hits))

    print(f"# Hotpath debt audit: {path}")
    print()
    print("## Unused imports")
    if unused_imports:
        for name, origin, lineno in unused_imports:
            print(f"- line {lineno}: {name} from {origin}")
    else:
        print("- none")

    print()
    print("## Duplicate helper definitions")
    if duplicate_functions:
        for name, lines in sorted(duplicate_functions.items()):
            joined = ", ".join(str(x) for x in lines)
            print(f"- {name}: lines {joined}")
    else:
        print("- none")

    print()
    print("## Definition-only helpers")
    if definition_only:
        for name, lineno in sorted(definition_only, key=lambda x: x[1]):
            print(f"- line {lineno}: {name}")
    else:
        print("- none")

    print()
    print("## Served-shape-only helpers")
    if served_only:
        for name in served_only:
            print(f"- {name}")
    else:
        print("- none")

    print()
    print("## Fallback-only helpers")
    if fallback_only:
        for name in fallback_only:
            print(f"- {name}")
    else:
        print("- none")

    print()
    print("## Remaining aiter/fp4_utils/shuffle_weight use on served shapes")
    if served_aiter_refs:
        for name, lineno, hits in served_aiter_refs:
            print(f"- line {lineno}: {name} -> {', '.join(hits)}")
    else:
        print("- none")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
