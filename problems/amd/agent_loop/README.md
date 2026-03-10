# Agent Loop

This package adds a repo-local closed loop around `popcorn-cli`:

1. capture the current `submission.py` as a baseline candidate
2. submit it to Popcorn in `test`, `benchmark`, or `leaderboard` mode
3. parse the plain-text result into structured metrics
4. track candidates and scores in a local SQLite store
5. run mutation commands that generate new `submission.py` variants
6. promote winning candidates back into the repo

## Commands

From the repo root:

```bash
python3 -m agent_loop healthcheck
python3 -m agent_loop baseline --problem mxfp4_mm
python3 -m agent_loop loop --problem mxfp4_mm --iterations 10
python3 -m agent_loop swarm --rounds 5
python3 -m agent_loop status --problem mxfp4_mm
python3 -m agent_loop promote --problem mxfp4_mm --candidate <candidate-id>
```

## Config

The CLI reads `agent_loop.toml` by default. An example is provided at
[`agent_loop.example.toml`](/Users/v/reference-kernels/problems/amd/agent_loop.example.toml).

Important fields:

- `workspace.root`: local artifact/database directory
- `workspace.api_url`: optional override for `POPCORN_API_URL`
- `problems.<name>.submission_path`: repo file to mutate/promote
- `problems.<name>.leaderboard`: Popcorn leaderboard slug
- `problems.<name>.mutator_command`: external command that writes a candidate file

## Mutator Contract

`mutator_command` is a shell command template. The runner expands:

- `{parent}`: quoted path to the current best candidate source
- `{output}`: quoted path where the new candidate must be written
- `{context}`: quoted path to a JSON file with problem and parent metadata
- `{candidate_dir}`: quoted artifact directory for this candidate
- `{repo_root}`: quoted repo root
- `{submission}`: quoted repo submission path
- `{problem}`: problem key

The default setup now points at the live MI355X competition leaderboards:

- `mxfp4_mm -> amd-mxfp4-mm`
- `moe_mxfp4 -> amd-moe-mxfp4`
- `mixed_mla -> amd-mixed-mla`

The bundled mutator is a Triton-oriented generator:

```toml
mutator_command = "python3 -m agent_loop.triton_mutator --parent {parent} --output {output} --context {context}"
```

It writes fresh `submission.py` candidates with embedded metadata and a small per-problem
search space. The current seeds are:

- `mxfp4_mm`: AITER quantization plus a Triton matmul over dequantized MXFP4 inputs
- `moe_mxfp4`: cached MXFP4 dequantization, grouped expert execution, Triton SwiGLU fusion
- `mixed_mla`: Triton BF16 decode-attention baseline over the live absorbed-query contract

## Objective

The default objective is `geom_mean_ns`, computed from the returned
`benchmark.<i>.mean` values. Lower is better.

## Notes

- `popcorn-cli` returns plain text, not JSON. This package parses the result text and
  stores both raw and structured copies.
- Winners are promoted by copying the chosen candidate into the configured
  `submission_path`. The package does not create git commits on its own.
- `swarm` runs the configured problems round-robin so we can keep the three search loops moving
  together.
- The local repo task YAMLs are MI300-era. The loop is configured against the live MI355X
  leaderboard slugs and generates submissions against those server-side contracts.
