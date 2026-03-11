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
python3 -m agent_loop loop --problem mxfp4_mm --iterations 10 --family triton_explore
python3 -m agent_loop swarm --rounds 5 --family triton_explore --bootstrap-baseline
python3 -m agent_loop campaign --mode leaderboard --family triton_explore --bootstrap-baseline --rounds 50 --max-consecutive-non-improve 8
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

The default mutator is now an LLM-backed generator with deterministic Triton fallback:

```toml
mutator_command = "python3 -m agent_loop.llm_mutator --config agent_loop.toml --parent {parent} --output {output} --context {context}"
```

The `llm_mutator`:

- reads the parent candidate, recent critique/history, policy profile, and a seeded Triton template
- asks the configured LLM for a full new `submission.py`
- compile-checks the result locally
- falls back to the deterministic Triton generator if the API key is missing, the request fails, or the returned code is invalid

Configuration lives in the `[llm]` table:

```toml
[llm]
enabled = true
provider = "openai"
model = "gpt-5-mini"
api_url = "https://api.openai.com/v1/responses"
api_key_env_var = "OPENAI_API_KEY"
reasoning_effort = "medium"
max_output_tokens = 12000
fallback_to_triton = true
```

The seeded Triton generator still provides the search-space skeleton. The current setup mutates both kernel variants and policy profiles:

- policy profiles choose which family/shape regime to emphasize next
- critiques emit structured `policy_signal` values such as `contract_repair`,
  `throughput_shift`, and `latency_repair`
- candidate metadata records both the kernel variant and the chosen policy profile

- `mxfp4_mm`: contract-aware MXFP4 exploration around shuffled inputs and Triton matmul schedules
- `moe_mxfp4`: grouped expert execution with Triton SwiGLU/routing exploration
- `mixed_mla`: FP8-contract Triton decode-attention exploration over the live absorbed-query path

## Objective

The default objective is `geom_mean_ns`. In leaderboard mode, the parser now prefers the
`## Ranked Benchmark` section when it is present; otherwise it falls back to the plain
benchmark section. Lower is better.

## Notes

- `popcorn-cli` returns plain text, not JSON. This package parses the result text and
  stores both raw and structured copies.
- Winners are promoted by copying the chosen candidate into the configured
  `submission_path`. The package does not create git commits on its own.
- `swarm` runs the configured problems round-robin so we can keep the three search loops moving
  together.
- `campaign` is the overnight runner: it supports leaderboard-mode round-robin search and stops
  when each problem hits a configured non-improvement plateau.
- The local repo task YAMLs are MI300-era. The loop is configured against the live MI355X
  leaderboard slugs and generates submissions against those server-side contracts.
