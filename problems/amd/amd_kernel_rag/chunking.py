from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
import re

from .sources import SourceDocument


MAX_BLOCK_LINES = 80
OVERLAP_LINES = 12

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLEGEN_RE = re.compile(r"^\s*(defm?|class|multiclass)\s+([A-Za-z0-9_.$]+)\b")
PY_DEF_RE = re.compile(r"^\s*(def|class)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
CHECK_LABEL_RE = re.compile(r"^\s*//\s*CHECK-LABEL:\s*(.+?)\s*$")
ASM_LABEL_RE = re.compile(r"^\s*([A-Za-z_.$][A-Za-z0-9_.$]*)\s*:\s*$")
C_FUNC_RE = re.compile(
    r"^\s*(?:template\s*<.*>\s*)?(?:[A-Za-z_][\w:\s<>\*&,\[\]]+\s+)?([A-Za-z_][A-Za-z0-9_:]*)\s*\([^;]*\)\s*(?:\{|$)"
)
EXACT_TOKEN_RE = re.compile(
    r"__builtin_amdgcn_[A-Za-z0-9_]+"
    r"|llvm\.amdgcn\.[A-Za-z0-9_.]+"
    r"|v_[A-Za-z0-9_]+"
    r"|s_[A-Za-z0-9_]+"
    r"|buffer_[A-Za-z0-9_]+"
    r"|flat_[A-Za-z0-9_]+"
    r"|global_[A-Za-z0-9_]+"
    r"|ds_[A-Za-z0-9_]+"
    r"|image_[A-Za-z0-9_]+"
    r"|gfx\d+"
    r"|[A-Za-z][A-Za-z0-9_]*\d+x\d+x\d+[A-Za-z0-9_]*"
    r"|mfma_[A-Za-z0-9_]+",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
SPLIT_RE = re.compile(r"[^A-Za-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source_id: str
    canonical_uri: str
    display_path: str
    file_path: str
    symbol: str | None
    chunk_kind: str
    start_line: int
    end_line: int
    commit: str | None
    content_hash: str
    text: str
    search_text: str
    exact_tokens: list[str]


def chunk_document(document: SourceDocument) -> list[Chunk]:
    lines = document.text.splitlines()
    if not lines:
        return []
    anchors = _find_anchors(lines, document)
    blocks = _blocks_from_anchors(lines, anchors)
    chunks: list[Chunk] = []
    for block in blocks:
        chunks.extend(_chunk_block(document, lines, **block))
    return chunks


def extract_exact_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for match in EXACT_TOKEN_RE.finditer(text):
        token = match.group(0).strip().lower()
        if token and token not in seen:
            seen.add(token)
            tokens.append(token)
    return tokens


def query_exact_tokens(query: str) -> list[str]:
    return extract_exact_tokens(query)


def normalized_query_terms(query: str) -> list[str]:
    tokens = extract_exact_tokens(query)
    words = [item.lower() for item in WORD_RE.findall(query)]
    expanded: list[str] = []
    seen: set[str] = set()
    for token in tokens + words:
        for term in _expand_term(token):
            if len(term) < 2 or term in seen:
                continue
            seen.add(term)
            expanded.append(term)
    return expanded


def citation_label(chunk: Chunk) -> str:
    symbol = f"::{chunk.symbol}" if chunk.symbol else ""
    return f"{chunk.display_path}{symbol}:{chunk.start_line}-{chunk.end_line}"


def _find_anchors(lines: list[str], document: SourceDocument) -> list[dict[str, object]]:
    anchors: list[dict[str, object]] = []
    for index, line in enumerate(lines):
        symbol: str | None = None
        kind: str | None = None
        if match := HEADING_RE.match(line):
            symbol = match.group(2).strip()
            kind = "heading"
        elif match := TABLEGEN_RE.match(line):
            symbol = match.group(2)
            kind = "tablegen"
        elif match := PY_DEF_RE.match(line):
            symbol = match.group(2)
            kind = "python"
        elif match := CHECK_LABEL_RE.match(line):
            symbol = match.group(1).strip()
            kind = "check-label"
        elif match := ASM_LABEL_RE.match(line):
            symbol = match.group(1)
            kind = "asm-label"
        elif match := C_FUNC_RE.match(line):
            symbol = match.group(1)
            kind = "function"
        elif "__builtin_amdgcn_" in line or "llvm.amdgcn." in line:
            symbol = extract_exact_tokens(line)[0] if extract_exact_tokens(line) else None
            kind = "token-anchor"

        if not symbol:
            continue
        anchors.append(
            {
                "anchor_line": index,
                "symbol": symbol,
                "chunk_kind": kind,
                "start_line": _expand_to_comment_context(lines, index),
            }
        )

    if not anchors:
        suffix = Path(document.display_path).suffix.lower()
        if suffix in {".md", ".rst", ".txt", ".html"}:
            return [{"anchor_line": 0, "symbol": document.title, "chunk_kind": "document", "start_line": 0}]
    deduped: list[dict[str, object]] = []
    seen_lines: set[int] = set()
    for anchor in anchors:
        line_number = int(anchor["anchor_line"])
        if line_number in seen_lines:
            continue
        seen_lines.add(line_number)
        deduped.append(anchor)
    return deduped


def _expand_to_comment_context(lines: list[str], index: int) -> int:
    lower = index
    budget = 6
    while lower > 0 and budget > 0:
        candidate = lines[lower - 1].strip()
        if not candidate:
            lower -= 1
            budget -= 1
            continue
        if candidate.startswith(("//", "#", "/*", "*", ";")):
            lower -= 1
            budget -= 1
            continue
        break
    return lower


def _blocks_from_anchors(lines: list[str], anchors: list[dict[str, object]]) -> list[dict[str, object]]:
    if not anchors:
        return [{"start": 0, "end": len(lines) - 1, "symbol": None, "chunk_kind": "window"}]
    blocks: list[dict[str, object]] = []
    for index, anchor in enumerate(anchors):
        start = int(anchor["start_line"])
        next_start = len(lines)
        if index + 1 < len(anchors):
            next_start = int(anchors[index + 1]["start_line"])
        end = max(start, next_start - 1)
        blocks.append(
            {
                "start": start,
                "end": end,
                "symbol": anchor.get("symbol"),
                "chunk_kind": str(anchor["chunk_kind"]),
            }
        )
    return blocks


def _chunk_block(
    document: SourceDocument,
    lines: list[str],
    *,
    start: int,
    end: int,
    symbol: str | None,
    chunk_kind: str,
) -> list[Chunk]:
    start = max(0, start)
    end = min(len(lines) - 1, end)
    if end < start:
        return []
    if (end - start + 1) <= MAX_BLOCK_LINES:
        return [_make_chunk(document, lines, start, end, symbol=symbol, chunk_kind=chunk_kind)]

    chunks: list[Chunk] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + MAX_BLOCK_LINES - 1)
        chunks.append(_make_chunk(document, lines, cursor, chunk_end, symbol=symbol, chunk_kind=chunk_kind))
        if chunk_end >= end:
            break
        cursor = max(cursor + 1, chunk_end - OVERLAP_LINES + 1)
    return chunks


def _make_chunk(
    document: SourceDocument,
    lines: list[str],
    start: int,
    end: int,
    *,
    symbol: str | None,
    chunk_kind: str,
) -> Chunk:
    text = "\n".join(lines[start : end + 1]).strip()
    exact_tokens = extract_exact_tokens("\n".join([symbol or "", text]))
    search_text = _search_text(symbol=symbol, text=text, exact_tokens=exact_tokens)
    chunk_id = sha1(
        f"{document.doc_id}:{symbol or ''}:{chunk_kind}:{start + 1}:{end + 1}:{document.content_hash}".encode("utf-8")
    ).hexdigest()[:32]
    return Chunk(
        chunk_id=chunk_id,
        doc_id=document.doc_id,
        source_id=document.source_id,
        canonical_uri=document.canonical_uri,
        display_path=document.display_path,
        file_path=document.file_path,
        symbol=symbol,
        chunk_kind=chunk_kind,
        start_line=start + 1,
        end_line=end + 1,
        commit=document.commit,
        content_hash=document.content_hash,
        text=text,
        search_text=search_text,
        exact_tokens=exact_tokens,
    )


def _search_text(*, symbol: str | None, text: str, exact_tokens: list[str]) -> str:
    words = WORD_RE.findall(text.lower())
    extra: list[str] = []
    if symbol:
        extra.extend(_expand_term(symbol.lower()))
    for token in exact_tokens:
        extra.extend(_expand_term(token))
    values = [*words, *extra]
    return " ".join(item for item in values if item)


def _expand_term(token: str) -> list[str]:
    raw = token.lower()
    parts = [item for item in SPLIT_RE.split(raw) if item]
    expanded = [raw]
    expanded.extend(parts)
    if "." in raw:
        expanded.append(raw.replace(".", "_"))
    if raw.startswith("__builtin_amdgcn_"):
        expanded.append(raw.removeprefix("__builtin_amdgcn_"))
    if raw.startswith("llvm.amdgcn."):
        expanded.append(raw.removeprefix("llvm.amdgcn."))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in expanded:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped
