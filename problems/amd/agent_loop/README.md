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
python3 -m agent_loop loop --problem mxfp4_mm --iterations 10 --family hip_explore
python3 -m agent_loop swarm --rounds 5 --bootstrap-baseline
python3 -m agent_loop campaign --mode leaderboard --bootstrap-baseline --rounds 50 --max-consecutive-non-improve 8
python3 -m agent_loop cleanup --stale-pending-hours 6
python3 -m agent_loop status --problem mxfp4_mm
python3 -m agent_loop promote --problem mxfp4_mm --candidate <candidate-id>
python3 -m agent_loop handroll-campaign --problem mxfp4_mm --rounds 5 --stages test,benchmark
python3 -m agent_loop harness-run --problem mxfp4_mm --source .agent-loop/problems/mxfp4_mm/manual/hip_reference_tiled_retry_20260312-074121/submission.py --label manual-hip-ref
python3 -m agent_loop harness-summary --problem mxfp4_mm
python3 -m agent_loop harness-resume --problem mxfp4_mm
python3 -m agent_loop mxfp4-closed-loop status --report
python3 -m agent_loop mxfp4-closed-loop preflight --variant v47 --source .agent-loop/manual/native_scaled_pure_thin_shscale_v47/submission.py --lane A --hypothesis "thin shuffled-scale direct-contract check"
python3 -m agent_loop mxfp4-closed-loop submit --variant v47 --source .agent-loop/manual/native_scaled_pure_thin_shscale_v47/submission.py --lane A --stage test
```

For `mxfp4_mm`, treat HIP `submission.py` files as the only KernelBot-facing artifacts.
If you need a semantic control, keep it as a separate local oracle file such as
`.agent-loop/problems/mxfp4_mm/manual/torch_reference_oracle_f32/oracle.py`;
do not treat a `torch.mm` control as a real submission candidate.

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

The default mutator is now an LLM-backed generator with seeded kernel families:

```toml
mutator_command = "python3 -m agent_loop.llm_mutator --config agent_loop.toml --parent {parent} --output {output} --context {context}"
```

The `llm_mutator`:

- reads the parent candidate, recent critique/history, policy profile, and a seeded kernel template
- optionally injects grounded AMD retrieval context from `amd_kernel_rag` when an index is available
- asks the configured generator for a full new `submission.py`
- compile-checks the result locally
- falls back to the deterministic seed renderer if the chosen provider is unavailable, the request fails, or the returned code is invalid

Optional retrieval hook:

- build the index from `/Users/v/reference-kernels/problems/amd` with `python3 -m amd_kernel_rag.cli build`
- the default discovery path is `.agent-loop/retrieval/amd-kernel-rag/`, with a fallback to `retrieval/amd-kernel-rag/`
- you can override discovery with `AMD_KERNEL_RAG_INDEX_DIR=/absolute/path/to/index`
- when present, each candidate directory gets `rag_context.json` and `rag_context.txt` before the mutator runs

Configuration lives in the `[llm]` table:

```toml
[llm]
enabled = true
provider = "auto"
model = "gpt-5-mini"
api_url = "https://api.openai.com/v1/responses"
api_key_env_var = "OPENAI_API_KEY"
anthropic_api_url = "https://api.anthropic.com/v1/messages"
anthropic_api_key_env_var = "ANTHROPIC_API_KEY"
anthropic_version = "2023-06-01"
openrouter_api_url = "https://openrouter.ai/api/v1/chat/completions"
openrouter_api_key_env_var = "OPENROUTER_API_KEY"
openrouter_http_referer = ""
openrouter_title = "reference-kernels-agent-loop"
reasoning_effort = "medium"
max_output_tokens = 12000
fallback_to_seed = true
codex_cli = "codex"
codex_model = ""
codex_sandbox = "read-only"
codex_use_plan = true
codex_parallel_agents = 3
```

Provider behavior:

- `provider = "auto"`: use OpenAI Responses when `OPENAI_API_KEY` is present, otherwise use local `codex exec`, otherwise fall back to the seeded kernel generator
- `provider = "openai"`: force the Responses API path
- `provider = "anthropic"`: use the Anthropic Messages API with `ANTHROPIC_API_KEY`
- `provider = "openrouter"`: use OpenRouter chat completions with `OPENROUTER_API_KEY`
- `provider = "codex_cli"`: force local Codex CLI generation

The Codex path is designed as the no-API-key option. It shells out to `codex exec`, asks for a concise plan, and explicitly allows helper-agent fanout up to the configured `codex_parallel_agents` count before returning a single final `submission.py`.
Leave `codex_model` empty to let the local Codex CLI use its account-default model.
If you set `codex_timeout_seconds`, the mutator will fail fast at that limit; if you omit it, `codex exec` runs without a wall-clock timeout.
For Codex, the mutator now uses an isolated candidate workspace in `workspace-write` mode: it seeds `submission.py` from the kept parent, asks Codex to edit that file in place, then validates the edited file locally before any remote submission.

The seeded generator provides the search-space skeleton. The current setup mutates both kernel variants and policy profiles:

- policy profiles choose which family/shape regime to emphasize next
- critiques emit structured `policy_signal` values such as `contract_repair`,
  `throughput_shift`, and `latency_repair`
- candidate metadata records both the kernel variant and the chosen policy profile

- `mxfp4_mm`: HIP-first `load_inline` exploration on `gfx950`, with correctness-first tiled bf16 seeds and scaled-MFMA/LDS seed variants
- `moe_mxfp4`: grouped expert execution with generated-kernel SwiGLU/routing exploration
- `mixed_mla`: FP8-contract generated decode-attention exploration over the live absorbed-query path

For `mxfp4_mm`, the current default is HIP-first:

- `default_family = "hip_explore"` routes MM experiments through `load_inline` on `gfx950`
- the seed path stays in one compilation pipeline: Python `submission.py` + HIP C++ only
- the first HIP seeds are correctness-first tiled bf16 kernels that the loop can evolve toward CDNA4 scaled-MFMA, LDS swizzle, and double buffering
- HIP candidates are seeded from the clean HIP template and rejected if they keep inherited Triton scaffold
- HIP candidates are also rejected if they contain purity-breaking tokens such as alternate `stream` usage, event/timing hacks, or inherited Triton scaffold
- HIP optimization is benchmark-fidelity constrained: the loop should prefer steady-state sustained throughput improvements and reject duty-cycle, stream, or timing-manipulation tricks that can distort measured microseconds
- this is intentionally closer to an AutoKernel-style microkernel program than a broad codegen search

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
- `handroll-campaign` is the phase-2 MM optimizer: it starts from the tracked working seed,
  applies one curated hand-rolled optimization move at a time, runs `test -> benchmark`, and
  keeps or reverts automatically based on the benchmark objective.
- `mxfp4-closed-loop` is the quota-aware coordinator for the pure/legal `mxfp4_mm` race:
  - safe immutable baseline is [v44](/Users/v/reference-kernels/problems/amd/.agent-loop/manual/native_scaled_pure_compiled_bscale_v44/submission.py)
  - experiment ledger lives at `.agent-loop/closed_loop/mxfp4_mm/experiment_ledger.jsonl`
  - preflight reports live under `.agent-loop/closed_loop/mxfp4_mm/preflight/`
  - the coordinator owns the remote budget and is the only path allowed to spend `test`, `benchmark`, or `leaderboard` quota
  - default mode is now remote-first: local preflight is static-only unless you explicitly request a Docker runtime
  - `benchmark` and `test` share the same hourly bucket operationally, with 2 slots reserved as contingency
  - new-candidate budget is capped at 2 remote `test` and 2 remote `benchmark` submissions per hour
  - `leaderboard` is capped at 1/hour and blocked before minute 45 unless you explicitly override the code
  - local preflight is required before any remote submission
  - leaderboard promotion is conservative: passing remote `test`, passing remote `benchmark`, at least 10% geomean improvement over the promoted safe baseline, and no single shape regressing by more than 5% unless the overall geomean improves by at least 15%
  - Docker preflight profiles live in `agent_loop/docker/`:
    - `Dockerfile.amd-parity-full`: ROCm 7.1 HIP SDK (`hiplibsdk`, `--no-dkms`) + torch 2.10.0+rocm7.1 + pinned `aiter` parity image
    - `Dockerfile.amd-compile-fast`: lighter ROCm/Torch compile image using the same minimal HIP SDK install for candidates that do not import `aiter`
  - local Docker preflight is forced to `linux/amd64` so ROCm package resolution matches the x86_64 competition environment even on Apple Silicon hosts
  - use `amd-parity-full` by default for current `mxfp4_mm` candidates because the promoted legal path still imports `aiter`
  - if Docker parity is slow or flaky, use the default `--runtime none` path and let the remote cluster be the runtime/JIT validator after static local checks pass
- `harness-run`, `harness-resume`, and `harness-summary` add a KernelBench-v3-inspired staged
  harness around one submission source:
  - every harness run gets its own artifact directory under `.agent-loop/harness_runs/<problem>/`
  - the submission source is copied into the run directory as `submission.py`
  - stages are explicit and ordered, typically `test -> benchmark -> leaderboard`
  - each stage records `result.txt`, `stdout.txt`, `stderr.txt`, and `parsed_metrics.json`
  - stage targets are named like `kernelbot-test`, `kernelbot-benchmark`, and `kernelbot-ranked`
  - `harness-resume` continues from the first non-`ok` stage, and `harness-summary` prints a
    compact JSON view of the run state
- `--bootstrap-baseline` is idempotent: it reuses an already evaluated baseline for that mode and
  only resubmits the repo anchor when the submission source has changed or no cached baseline eval
  exists yet.
- failed mutation candidates are compacted after evaluation: the loop keeps a small pruned summary
  with meta/critique/metrics for future prompt memory, then deletes the bulky candidate directory
  so dead ends do not accumulate indefinitely.
- `cleanup` backfills that same pruning logic for older failed candidates and can also drop stale
  unevaluated mutation directories after a safety threshold with `--stale-pending-hours`.
- `mxfp4_mm` can run in an AutoKernel-style single-trunk mode:
  - kept kernel lives at `.agent-loop/problems/mxfp4_mm/working/submission.py`
  - current trial lives at `.agent-loop/problems/mxfp4_mm/working/attempt/`
  - every ranked round appends one row to `.agent-loop/problems/mxfp4_mm/working/results.tsv`
  - detailed per-round records go to `.agent-loop/problems/mxfp4_mm/working/journal.jsonl`
  - `experiment.plan.json` captures the round hypothesis, policy, variant, and source inspirations before submission
  - `state.json` tracks the current kept candidate plus keep/revert counts
  - `candidate.diff` and `scope_check.json` capture the current diff and focused-edit budget check
  - `max_changed_lines` and `max_edit_hunks` in `agent_loop.toml` can reject broad rewrites as `scope_reject` before burning a ranked submission
  - `default_family = "hip_explore"` makes MM default to the HIP `load_inline` path unless you override `--family`
  - the prompt discipline intentionally borrows from AutoKernel, KernelAgent, ADRS/OpenEvolve, and Atom-of-Thoughts: one experiment, one evaluator, one keep/revert decision
- The local repo task YAMLs are MI300-era. The loop is configured against the live MI355X
  leaderboard slugs and generates submissions against those server-side contracts.
