# AMD Kernel RAG

Text-only hybrid retrieval for low-level AMD kernel engineering research.

The stack is built around:

- dense retrieval with Mixedbread Wholembed v3
- sparse retrieval with SQLite FTS5 BM25
- rank fusion with reciprocal-rank fusion
- symbol-aware chunking for LLVM docs, tablegen, tests, blogs, and repo-local probes
- exact-token boosts for `__builtin_amdgcn_*`, `llvm.amdgcn.*`, opcodes, and MFMA mnemonics

## Why this is integrated but still safe

The index/cache is intentionally separate from the handrolled optimizer state:

- retrieval data lives under `.agent-loop/retrieval/amd-kernel-rag/` by default
- `agent_loop` only reads a small retrieved context file when the index exists
- if the index is missing, the current workflow behaves exactly as before

That gives you the best of both worlds:

- no coupling between index rebuilds and candidate state
- no risk of corrupting the current search loop
- automatic grounded context for future mutation rounds once the index is built

## Quick Start

From `/Users/v/reference-kernels/problems/amd`:

```bash
python3 -m amd_kernel_rag.cli build
python3 -m amd_kernel_rag.cli summary
python3 -m amd_kernel_rag.cli query --query "__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4 operand order" --top-k 5
python3 -m amd_kernel_rag.cli answer --query "What local notes mention GPU memory faults in scaled MFMA experiments?"
python3 -m amd_kernel_rag.cli eval --benchmark amd_kernel_rag/benchmarks/kernel_queries.json --report-out amd_kernel_rag/reports/latest.md
```

## Dense Embeddings

Dense retrieval uses Mixedbread's direct embeddings API by default.

This repo does not currently use Mixedbread Stores for retrieval. It:

- chunks the local and web corpus itself
- writes a local SQLite/FTS index
- optionally asks Mixedbread only for dense vectors

That means the relevant Mixedbread quota is `Embedding Tokens`, not `Store Ingested Tokens` or `Store Search`.

Environment variables:

- `MIXEDBREAD_API_KEY` or `MXBAI_API_KEY`
- optional `AMD_KERNEL_RAG_EMBED_MODEL`
- optional `AMD_KERNEL_RAG_EMBED_BASE_URL`
- optional `AMD_KERNEL_RAG_EMBED_PROMPT`
- optional `AMD_KERNEL_RAG_EMBED_MAX_BATCH_CHARS`

Default model id:

- `mixedbread-ai/mxbai-embed-large-v1`

Default query prompt:

- `Represent this sentence for searching relevant passages:`

Default max batch chars:

- `32000`

Dense build profiles:

- `mxfp4-mm-free`:
  dense-index the curated `mxfp4-mm` subset only
- `full`:
  dense-index every chunk
- `none`:
  sparse-only, equivalent to skipping dense embeddings

Examples:

```bash
python3 -m amd_kernel_rag.cli build --dense-profile mxfp4-mm-free
python3 -m amd_kernel_rag.cli build --dense-profile full
python3 -m amd_kernel_rag.cli build --dense-profile none
python3 -m amd_kernel_rag.cli build --skip-dense
```

If you do not have a Mixedbread API key yet, you can still build the sparse index:

```bash
python3 -m amd_kernel_rag.cli build --skip-dense
```

## Free-Plan Reality Check

The sparse index works well on the full local corpus, but the full dense build can be much larger than a free embedding-token budget.

As of the current corpus:

- `156241` chunks
- about `273,609,021` text characters total in the chunk table

That is far beyond a `10M` embedding-token monthly budget in any realistic tokenizer. The default dense profile therefore uses a curated `mxfp4-mm-free` subset instead of trying to embed the whole corpus.

On the free plan, use one of these approaches:

- keep the full sparse index and use the default curated dense profile
- keep the full sparse index and skip dense
- raise the embedding budget before running a full dense rebuild

## Output Shape

Each retrieved chunk carries:

- source URL or absolute local path
- repo file path
- symbol name when detected
- exact line range
- commit hash when available
- raw snippet text for grounded citation

## Agent Loop Integration

No extra config is required.

Once the index exists at `.agent-loop/retrieval/amd-kernel-rag/`, `agent_loop` will automatically:

1. build a query from the current problem, recent failure history, and parent exact tokens
2. retrieve the top grounded chunks
3. write `rag_context.json` and `rag_context.txt` into the candidate directory
4. inject that grounded context into the mutator prompt

Override the index path with:

```bash
export AMD_KERNEL_RAG_INDEX_DIR=/absolute/path/to/index
```

## Files

- `amd_kernel_rag/cli.py`: build/query/eval CLI
- `amd_kernel_rag/sources.py`: corpus manifest and fetchers
- `amd_kernel_rag/chunking.py`: symbol-aware chunking and token normalization
- `amd_kernel_rag/index.py`: SQLite/FTS5 + dense vector build
- `amd_kernel_rag/retriever.py`: hybrid retrieval, exact-token boosts, grounded output
- `amd_kernel_rag/integration.py`: optional `agent_loop` bridge
- `amd_kernel_rag/eval.py`: benchmark metrics and report generation
- `amd_kernel_rag/ADDING_SOURCES.md`: how to extend the corpus
