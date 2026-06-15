# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Python CLI that measures tokens-per-second against an OpenAI-compatible inference endpoint (`/v1/completions` or `/v1/chat/completions`). Managed with `uv`; Python ≥3.11.

## Commands

```bash
uv sync --group dev                  # install deps + editable-install package
uv run benchmark                     # invoke the [project.scripts] entry (reads .env)
uv run python -m benchmark           # equivalent module form
uv run ruff check src tests          # lint (rule set locked in pyproject.toml)
uv run pytest                        # full test suite
uv run pytest -v
uv run pytest tests/test_questions.py::test_seeded_provider_is_reproducible
uv run pytest -k random
```

CLI flags override `.env`. `.env.example` documents every setting and targets a local vLLM server by default. `test1.sh` is a one-off curl probe for endpoint discovery — not part of the test suite.

## Architecture

The benchmark is a package at **`src/benchmark/`** organized by concern. `__init__.py` re-exports the public surface so tests and external scripts can keep using `from benchmark import X` without knowing which submodule X lives in.

| Module | Responsibility |
|---|---|
| `tokenizer.py` | `Tokenizer` protocol + `CharsTokenizer` / `TikTokenTokenizer`. `_build_tokenizer(spec)` (chars4 / auto / `tiktoken:<enc>`). `_estimate_tokens` (clamped output fallback). `_normalize_prompt` (pad/truncate via binary search). |
| `client.py` | `RequestResult`, `_is_chat_endpoint`, `_consume_sse_stream` (TTFT capture), and `benchmark_request` — the single HTTP call. Builds chat-vs-completions payload, threads `ignore_eos` / `min_tokens`, handles streaming vs non-streaming branches. |
| `prompts.py` | `_load_questions` + `_build_prompt_provider`. Provider is a `Callable[[], str]` — fixed (`--prompt`/`PROMPT`), random from JSON pool (`--questions_file`, seeded via `--seed`), or built-in default. |
| `records.py` | `RECORD_FIELDS` tuple = schema of the per-request log (`trial, request_index, prompt_chars, prompt_tokens, output_tokens, latency_s, ttft_s, scheduled_offset_s, queue_wait_s, ok, estimated, error`). `_record_writer(path)` context manager picks JSONL vs CSV by extension. |
| `stats.py` | `_percentile` (linear interpolation, numpy-default), `_poisson_arrival_offsets`, and human-readable printers (`_print_request_percentiles`, `_print_prompt_token_stats`). |
| `summary.py` | `RunSummary` dataclass — populated by every `run_benchmark` exit and pushed to a caller's list for the matrix-mode comparison table. |
| `matrix.py` | `_broadcast_configs` (zip + singleton broadcast) and `_print_comparison_table` (auto-adds TTFT / `achv_qps` columns based on what's populated). |
| `runner.py` | `run_benchmark` — dispatches between closed-loop (`trials × concurrent` barrier) and open-loop Poisson QPS (single executor pool, sleep-until-target dispatch, queue-wait tracking). Both paths accumulate the same stats lists and call `_finalize(rc)` which builds a `RunSummary` and (optionally) appends to `summary_out`. |
| `cli.py` | argparse layered over env vars via `_env_str/_int/_float`. Builds the prompt provider, applies `--prompt_tokens` normalization wrapper, broadcasts comma-separated `--endpoint`/`--model`/`--api_key` into N configs, loops `run_benchmark` per config (auto-suffixing `--output` filenames), and prints the comparison table when N > 1. |
| `__main__.py` | `python -m benchmark` entry — calls `cli.main()`. The `benchmark` console script in `pyproject.toml` is the same entry. |

Cross-module dependency graph is acyclic: `tokenizer` and `summary` are leaves; `client` / `stats` / `records` / `prompts` / `matrix` depend on at most one of those; `runner` pulls from all of them; `cli` sits at the top.

`src/benchmark/questions.json` is a JSON array of 100 prompts — the default random pool. `_load_questions` filters non-string/blank entries and rejects empty arrays or non-array roots.

Exit codes: `0` clean, `2` if any request failed, `1` if no successful trials at all. In matrix mode the process exits with `max(per-config codes)`. Failure and estimation counts surface as stderr warnings.

## Tests

`tests/` uses `pytest` with `pythonpath = ["src"]` configured in `pyproject.toml` so tests import `benchmark` directly.

**Patching gotcha after the module split:** mocks must target the binding site, not the public re-export. Use `monkeypatch.setattr("benchmark.runner.benchmark_request", fake)` (not `benchmark.benchmark_request`) and `patch("benchmark.client.requests.post", ...)` (not `benchmark.requests.post`). The hygiene test `test_no_unused_top_level_imports_in_benchmark` whitelists `from __future__ import annotations` since it's side-effect-only.

Test files by area: `test_questions.py` (prompt provider), `test_benchmark_request.py` (HTTP payload), `test_streaming.py` (SSE consumer + TTFT), `test_percentile.py` (math + integration), `test_output_log.py` (JSONL/CSV writer), `test_tokenizer.py` (tokenizer abstraction), `test_open_loop.py` (Poisson dispatcher), `test_prompt_length.py` (`_normalize_prompt`), `test_matrix.py` (broadcast + summary + comparison table), `test_ci_workflow.py` (YAML structure + injection guards), `test_hygiene.py` (no literal secrets, no unused imports).
