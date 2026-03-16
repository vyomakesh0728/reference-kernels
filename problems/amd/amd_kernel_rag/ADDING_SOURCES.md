# Adding New Sources

The source manifest is JSON. You can generate the current default template with:

```bash
python3 -m amd_kernel_rag.cli write-default-manifest amd_kernel_rag/default_sources.json
```

Then edit the `sources` array and rebuild:

```bash
python3 -m amd_kernel_rag.cli build --manifest amd_kernel_rag/default_sources.json --refresh-sources
```

## Source Types

### `git`

Use this for GitHub or other git repos.

Fields:

- `id`: stable source id
- `kind`: `"git"`
- `repo`: clone URL
- `ref`: branch, tag, or ref name
- `paths`: files or directories to sparse-checkout

Example:

```json
{
  "id": "llvm-project-amdgpu",
  "kind": "git",
  "repo": "https://github.com/llvm/llvm-project.git",
  "ref": "main",
  "paths": [
    "clang/include/clang/Basic/BuiltinsAMDGPU.td",
    "llvm/include/llvm/IR/IntrinsicsAMDGPU.td"
  ]
}
```

### `web`

Use this for standalone HTML or text pages.

Fields:

- `id`
- `kind`: `"web"`
- `url`

Example:

```json
{
  "id": "rocm-cdna4-gemm-blog",
  "kind": "web",
  "url": "https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html"
}
```

### `local`

Use this for repo-local notes, probes, and kernels.

Fields:

- `id`
- `kind`: `"local"`
- `root`: absolute or manifest-relative root
- `patterns`: glob patterns to include
- `exclude`: glob patterns to skip

Example:

```json
{
  "id": "amd-local-docs",
  "kind": "local",
  "root": "/Users/v/reference-kernels/problems/amd",
  "patterns": [
    "session-summary.md",
    ".agent-loop/problems/*/manual/**/*.py"
  ],
  "exclude": [
    ".agent-loop/**/evaluation/**"
  ]
}
```

## Good Source-Addition Rules

- Prefer sparse git paths over cloning entire repos.
- Add local probes and summaries, but exclude bulky generated evaluation artifacts.
- Keep source ids stable so benchmark and report history stay comparable.
- Rebuild with `--refresh-sources` after changing git or web sources.
- If a new source introduces domain-specific symbols, add benchmark queries that reference them.

## Benchmark Maintenance

After adding meaningful sources:

1. add 1-3 new benchmark queries in `amd_kernel_rag/benchmarks/kernel_queries.json`
2. point each query at one or more relevant selectors with `path_contains`, `symbol_contains`, `text_contains`, or `source_contains`
3. rerun:

```bash
python3 -m amd_kernel_rag.cli eval --benchmark amd_kernel_rag/benchmarks/kernel_queries.json --report-out amd_kernel_rag/reports/latest.md
```

