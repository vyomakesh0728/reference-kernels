from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha1, sha256
from html.parser import HTMLParser
from pathlib import Path
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request


TEXT_SUFFIXES = {
    ".c",
    ".cc",
    ".cl",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".html",
    ".inc",
    ".json",
    ".ll",
    ".md",
    ".mir",
    ".py",
    ".rst",
    ".s",
    ".td",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_PATTERNS = (
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
)


@dataclass(frozen=True)
class SourceSpec:
    id: str
    kind: str
    url: str | None = None
    repo: str | None = None
    ref: str | None = None
    paths: list[str] | None = None
    root: str | None = None
    patterns: list[str] | None = None
    exclude: list[str] | None = None


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    source_id: str
    kind: str
    title: str
    canonical_uri: str
    display_path: str
    file_path: str
    commit: str | None
    content_hash: str
    text: str


def default_manifest(workspace_root: Path) -> list[SourceSpec]:
    workspace = workspace_root.resolve()
    return [
        SourceSpec(
            id="llvm-project-amdgpu",
            kind="git",
            repo="https://github.com/llvm/llvm-project.git",
            ref="main",
            paths=[
                "clang/include/clang/Basic/BuiltinsAMDGPU.td",
                "llvm/include/llvm/IR/IntrinsicsAMDGPU.td",
                "llvm/test/CodeGen/AMDGPU",
                "clang/test/CodeGenOpenCL",
                "clang/test/SemaOpenCL",
            ],
        ),
        SourceSpec(
            id="llvm-amdgpu-usage-html",
            kind="web",
            url="https://llvm.org/docs/AMDGPUUsage.html",
        ),
        SourceSpec(
            id="gcnasm",
            kind="git",
            repo="https://github.com/carlushuang/gcnasm.git",
            ref="master",
            paths=["."],
        ),
        SourceSpec(
            id="rocm-cdna4-gemm-blog",
            kind="web",
            url="https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html",
        ),
        SourceSpec(
            id="hazy-hk-blog",
            kind="web",
            url="https://hazyresearch.stanford.edu/blog/2025-11-09-hk",
        ),
        SourceSpec(
            id="hazy-amd-brr-blog",
            kind="web",
            url="https://hazyresearch.stanford.edu/blog/2025-11-09-amd-brr",
        ),
        SourceSpec(
            id="amd-local-docs",
            kind="local",
            root=str(workspace),
            patterns=[
                "session-summary.md",
                "*.md",
                "agent_loop/**/*.py",
                "fp8-mm/**/*.py",
                "mla-decode/**/*.py",
                "moe/**/*.py",
                "skills/**/*.md",
                "skills/**/*.py",
                ".agent-loop/problems/*/manual/**/*.py",
                ".agent-loop/handrolled/**/*.py",
                ".agent-loop/handrolled/**/*.json",
            ],
            exclude=[
                "**/__pycache__/**",
                "**/*.png",
                "**/*.pdf",
                ".agent-loop/**/evaluation/**",
                ".agent-loop/**/stdout.txt",
                ".agent-loop/**/stderr.txt",
                ".agent-loop/**/result.txt",
                ".agent-loop/**/parsed_metrics.json",
                ".agent-loop/**/response_payload.json",
            ],
        ),
    ]


def write_default_manifest(path: Path, workspace_root: Path) -> Path:
    path = path.resolve()
    payload = {"sources": [asdict(spec) for spec in default_manifest(workspace_root)]}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_manifest(path: Path | None, workspace_root: Path) -> list[SourceSpec]:
    if path is None:
        return default_manifest(workspace_root)
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw.get("sources")
    if not isinstance(items, list):
        raise ValueError("manifest must contain a top-level 'sources' array")
    specs: list[SourceSpec] = []
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("manifest entries must be JSON objects")
        specs.append(SourceSpec(**item))
    return specs


def sync_sources(
    specs: list[SourceSpec],
    *,
    cache_dir: Path,
    refresh: bool = False,
) -> list[SourceDocument]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    documents: list[SourceDocument] = []
    for spec in specs:
        if spec.kind == "git":
            documents.extend(_sync_git_source(spec, cache_dir=cache_dir, refresh=refresh))
        elif spec.kind == "web":
            documents.extend(_sync_web_source(spec, cache_dir=cache_dir, refresh=refresh))
        elif spec.kind == "local":
            documents.extend(_sync_local_source(spec))
        else:
            raise ValueError(f"unsupported source kind: {spec.kind}")
    return documents


def _sync_git_source(spec: SourceSpec, *, cache_dir: Path, refresh: bool) -> list[SourceDocument]:
    if not spec.repo:
        raise ValueError(f"git source '{spec.id}' is missing repo")
    repo_dir = cache_dir / "git" / _safe_id(spec.id)
    ref = spec.ref or "main"
    sparse_paths = spec.paths or ["."]
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                spec.repo,
                str(repo_dir),
            ]
        )
    _run(["git", "-C", str(repo_dir), "sparse-checkout", "set", "--no-cone", *sparse_paths])
    if refresh:
        _run(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", ref])
        _run(["git", "-C", str(repo_dir), "checkout", "--force", "FETCH_HEAD"])
    else:
        _run(["git", "-C", str(repo_dir), "checkout", "--force", ref], check=False)
    commit = _capture(["git", "-C", str(repo_dir), "rev-parse", "HEAD"]).strip() or None

    base_repo = spec.repo[:-4] if spec.repo.endswith(".git") else spec.repo
    docs: list[SourceDocument] = []
    for rel in sparse_paths:
        target = repo_dir / rel
        if target.is_file():
            doc = _document_from_file(
                source_id=spec.id,
                kind="git",
                path=target,
                root=repo_dir,
                canonical_uri=f"{base_repo}/blob/{commit}/{_rel_posix(target, repo_dir)}",
                commit=commit,
            )
            if doc is not None:
                docs.append(doc)
            continue
        if not target.exists():
            continue
        for path in _iter_text_files(target):
            doc = _document_from_file(
                source_id=spec.id,
                kind="git",
                path=path,
                root=repo_dir,
                canonical_uri=f"{base_repo}/blob/{commit}/{_rel_posix(path, repo_dir)}",
                commit=commit,
            )
            if doc is not None:
                docs.append(doc)
    return docs


def _sync_web_source(spec: SourceSpec, *, cache_dir: Path, refresh: bool) -> list[SourceDocument]:
    if not spec.url:
        raise ValueError(f"web source '{spec.id}' is missing url")
    web_dir = cache_dir / "web"
    web_dir.mkdir(parents=True, exist_ok=True)
    parsed = urllib.parse.urlparse(spec.url)
    ext = Path(parsed.path).suffix or ".html"
    cached = web_dir / f"{_safe_id(spec.id)}{ext}"
    meta_path = web_dir / f"{_safe_id(spec.id)}.json"
    if refresh or not cached.exists():
        req = urllib.request.Request(
            spec.url,
            headers={"User-Agent": "amd-kernel-rag/1.0 (+https://github.com/llvm/llvm-project)"},
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            payload = response.read()
            cached.write_bytes(payload)
            headers = dict(response.headers.items())
        meta_path.write_text(
            json.dumps(
                {
                    "url": spec.url,
                    "headers": headers,
                    "sha256": sha256(payload).hexdigest(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    raw = cached.read_bytes()
    content_hash = sha256(raw).hexdigest()
    text = _decode_web_payload(raw, suffix=cached.suffix)
    title = _web_title(text=text, url=spec.url)
    return [
        SourceDocument(
            doc_id=_stable_id(spec.id, spec.url, content_hash),
            source_id=spec.id,
            kind="web",
            title=title,
            canonical_uri=spec.url,
            display_path=spec.url,
            file_path=str(cached),
            commit=None,
            content_hash=content_hash,
            text=text,
        )
    ]


def _sync_local_source(spec: SourceSpec) -> list[SourceDocument]:
    root = Path(spec.root or ".").expanduser().resolve()
    patterns = spec.patterns or ["**/*"]
    exclude = tuple(spec.exclude or [])
    commit = _git_head(root)
    seen: set[Path] = set()
    docs: list[SourceDocument] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.exists() or path.is_dir():
                continue
            if any(part in SKIP_PATTERNS for part in path.parts):
                continue
            if exclude and any(path.match(item) for item in exclude):
                continue
            if path in seen or not _is_text_candidate(path):
                continue
            seen.add(path)
            doc = _document_from_file(
                source_id=spec.id,
                kind="local",
                path=path,
                root=root,
                canonical_uri=str(path),
                commit=commit,
            )
            if doc is not None:
                docs.append(doc)
    return sorted(docs, key=lambda item: item.display_path)


def _document_from_file(
    *,
    source_id: str,
    kind: str,
    path: Path,
    root: Path,
    canonical_uri: str,
    commit: str | None,
) -> SourceDocument | None:
    if not _is_text_candidate(path):
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return None
    content_hash = sha256(raw).hexdigest()
    rel = _rel_posix(path, root)
    title = path.name
    return SourceDocument(
        doc_id=_stable_id(source_id, rel, content_hash),
        source_id=source_id,
        kind=kind,
        title=title,
        canonical_uri=canonical_uri,
        display_path=rel,
        file_path=str(path),
        commit=commit,
        content_hash=content_hash,
        text=text,
    )


def _iter_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_PATTERNS for part in path.parts):
            continue
        if _is_text_candidate(path):
            files.append(path)
    return sorted(files)


def _is_text_candidate(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    if path.name in {"README", "LICENSE"}:
        return True
    return False


def _decode_web_payload(raw: bytes, *, suffix: str) -> str:
    text = raw.decode("utf-8", errors="replace")
    if suffix.lower() in {".html", ".htm"} or "<html" in text.lower():
        return _extract_html_text(text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in {"br", "div", "p", "section", "article", "table", "tr", "pre"}:
            self._chunks.append("\n")
        if tag == "li":
            self._chunks.append("\n- ")
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            self._chunks.append("\n" + ("#" * level) + " ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag in {"p", "div", "section", "article", "table", "tr", "pre"}:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        self._chunks.append(data)

    def text(self) -> str:
        joined = "".join(self._chunks)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        lines = [line.rstrip() for line in joined.splitlines()]
        return "\n".join(line for line in lines if line.strip())


def _extract_html_text(text: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(text)
    return parser.text()


def _web_title(*, text: str, url: str) -> str:
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    parsed = urllib.parse.urlparse(url)
    return Path(parsed.path).name or url


def _stable_id(*parts: str) -> str:
    digest = sha1("::".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-").lower() or "source"


def _rel_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _capture(command: list[str]) -> str:
    proc = subprocess.run(command, capture_output=True, check=True, text=True)
    return proc.stdout


def _run(command: list[str], *, check: bool = True) -> None:
    subprocess.run(
        command,
        check=check,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _git_head(root: Path) -> str | None:
    probe = root
    if probe.is_file():
        probe = probe.parent
    while True:
        if (probe / ".git").exists():
            try:
                return _capture(["git", "-C", str(probe), "rev-parse", "HEAD"]).strip() or None
            except (OSError, subprocess.CalledProcessError):
                return None
        if probe.parent == probe:
            return None
        probe = probe.parent
