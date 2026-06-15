# token-sec-calc

Measure **tokens per second** against any OpenAI-compatible LLM inference endpoint (vLLM, SGLang, llama.cpp server, TGI, hosted gateways, OpenAI itself).

Designed to give you the numbers that actually matter for serving a model:

- aggregate throughput (tokens/sec across the whole run)
- per-request latency percentiles (p50 / p90 / p95 / p99)
- time-to-first-token (TTFT) under streaming
- dispatcher queue wait under sustained Poisson load
- side-by-side comparison across endpoints and models

Network is plain HTTPS — no SDK lock-in. Output stays in your terminal or lands in JSONL/CSV for downstream analysis.

---

## Features

- **Closed-loop benchmarking** — N concurrent requests × M trials, barrier between trials.
- **Open-loop Poisson QPS** — sustained arrival rate over a fixed window, with dispatcher queue-wait tracking.
- **Streaming + TTFT** — Server-Sent Events with per-request time-to-first-token.
- **Pinned output length** — `ignore_eos` + `min_tokens` so tokens/sec isn't skewed by EOS variance (vLLM/SGLang).
- **Real tokenizer** — optional `tiktoken` for accurate prompt-token and output-token counts; chars/4 fallback otherwise.
- **Prompt-length normalization** — pad or truncate every prompt to a fixed token budget to remove input-length noise.
- **Random prompt pool** — 100 built-in prompts; supply your own JSON array.
- **Per-request log** — JSONL or CSV, one row per non-warmup request.
- **Multi-endpoint compare** — pass comma-separated lists, get a side-by-side table.
- **Reproducible** — seed both the prompt picker and the Poisson schedule.

---

## Installation

Requires Python ≥ 3.11. The project uses [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/<owner>/token-sec-calc.git
cd token-sec-calc
uv sync                          # runtime deps + editable-installs the package
uv sync --group dev              # add pytest + pyyaml (for running the test suite)
uv sync --extra tiktoken         # optional: accurate token counting via tiktoken
```

After `uv sync`, the `benchmark` console script is on the venv's PATH:

```bash
uv run benchmark --help
uv run python -m benchmark --help   # equivalent
```

---

## Quick start

Configure once via `.env`:

```bash
cp .env.example .env
$EDITOR .env       # set ENDPOINT, API_KEY, MODEL
uv run benchmark
```

Or pass everything on the command line:

```bash
uv run benchmark \
  --endpoint http://localhost:8000/v1/completions \
  --api_key EMPTY \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --concurrent 4 --trials 5 --max_tokens 128
```

CLI flags always override the matching `.env` variable.

A typical closed-loop summary looks like:

```
Benchmarking http://localhost:8000/v1/completions
API: completions (mode=auto)
Model: meta-llama/Llama-3.1-8B-Instruct
Mode: closed-loop (4 concurrent × 5 trials)
Max tokens per request: 128 (ignore_eos + min_tokens pinned)
Sampling: temperature=0.0 top_p=1.0
Streaming: off
Tokenizer: tiktoken:cl100k_base
Prompt source: random pick per request from .../questions.json (100 questions)
------------------------------------------------------------
Trial  1:   512 tokens |  3.42s |  149.71 tok/s
Trial  2:   512 tokens |  3.18s |  161.01 tok/s
...
------------------------------------------------------------
Results over 5 trials (20 requests):
  Aggregate throughput:  158.42 tok/s   (total_tokens / total_wall_time)
  Mean of per-trial:     159.10 tok/s
  Min / Max per-trial:   149.71 / 165.30 tok/s
  Total tokens:          2560
  Total wall time:       16.16s
  Prompt input tokens:  min=    7  mean=  12.4  max=   24  p50=12  p99=23
  Per-request latency (s) over 20 successes:
    p50=0.821  p90=1.052  p95=1.118  p99=1.144
  Per-request latency / token (s):
    p50=0.0064  p90=0.0082  p95=0.0087  p99=0.0089
```

---

## Configuration

Every setting is reachable via CLI flag or environment variable. CLI wins.

### Required

| Flag | Env var | Description |
|---|---|---|
| `--endpoint` | `ENDPOINT` | Inference URL. Comma-separated values enable matrix mode. |
| `--api_key`  | `API_KEY`  | Bearer token (any non-empty string for local vLLM by default). May be comma-separated. |
| `--model`    | `MODEL`    | Model name. May be comma-separated. |

### Optional

| Flag | Env var | Default | Description |
|---|---|---|---|
| `--prompt` | `PROMPT` | — | Fixed prompt for every request. Unset → random pick from the questions file. |
| `--questions_file` | `QUESTIONS_FILE` | bundled 100 prompts | JSON array of prompts. Empty string disables. |
| `--seed` | `SEED` | -1 | Random seed for prompt pick and Poisson schedule. -1 = nondeterministic. |
| `--max_tokens` | `MAX_TOKENS` | 100 | Output ceiling per request. |
| `--ignore_eos` | `IGNORE_EOS` | true | Send `ignore_eos` + `min_tokens=max_tokens` (vLLM/SGLang) so every request emits exactly `max_tokens`. Set `false` for strict OpenAI gateways. |
| `--concurrent` | `CONCURRENT` | 1 | Closed-loop: concurrent requests per trial. |
| `--trials` | `TRIALS` | 5 | Closed-loop: number of trials. |
| `--warmup` | `WARMUP` | 1 | Warmup requests before measuring (not logged or reported). |
| `--temperature` | `TEMPERATURE` | 0.0 | 0 = greedy / reproducible. |
| `--top_p` | `TOP_P` | 1.0 | Nucleus sampling. |
| `--timeout` | `TIMEOUT` | 120 | Per-request timeout (seconds). |
| `--mode` | `MODE` | auto | `auto` / `chat` / `completions`. `auto` picks `chat` if URL contains `/chat/completions`. |
| `--stream` | `STREAM` | false | SSE streaming with TTFT capture. |
| `--tokenizer` | `TOKENIZER` | auto | `auto` (tiktoken `cl100k_base` if installed else `chars4`), `chars4`, or `tiktoken:<encoding>` (e.g. `tiktoken:o200k_base`). |
| `--prompt_tokens` | `PROMPT_TOKENS` | 0 (off) | Pad/truncate every prompt to this token count. |
| `--qps` | `QPS` | 0 (off) | Open-loop target requests/sec (Poisson). Requires `--duration`. |
| `--duration` | `DURATION` | 0 | Open-loop window in seconds. |
| `--max_concurrency` | `MAX_CONCURRENCY` | auto | Open-loop worker pool size. Default `max(32, ceil(qps*2))` capped at 256. |
| `--output` | `OUTPUT` | — | Write per-request log. Format chosen by extension: `.jsonl` or `.csv`. Warmup excluded. |

Run `uv run benchmark --help` for the canonical reference.

---

## Modes

### Closed-loop (default)

`N concurrent` requests fire in parallel, the slowest completion ends the trial, then the next trial begins. Throughput per trial = trial tokens / trial wall time. Aggregate = total tokens / total wall time across all trials.

```bash
uv run benchmark --concurrent 8 --trials 10 --max_tokens 256
```

Use when you want to characterize **steady-state batched serving** at a known concurrency level.

### Open-loop QPS

Set `--qps` and `--duration` to switch to a Poisson arrival schedule. Requests are pre-scheduled, dispatched at their target time, and processed by a shared worker pool. The summary adds `Achieved request rate` and `Dispatcher queue wait` rows.

```bash
uv run benchmark --qps 25 --duration 60 --max_tokens 128
```

Use when you want to characterize **what happens at a target request rate** — including head-of-line blocking and saturation. The queue-wait stats reveal when the worker pool can't keep up with the arrival rate.

### Streaming + TTFT

```bash
uv run benchmark --stream true --concurrent 4 --trials 5
```

Adds a `Time-to-first-token` percentile row to the summary and a `ttft_s` column to the per-request log. Combine with `--qps` for serving-style benchmarks.

### Multi-endpoint compare

Pass comma-separated lists to `--endpoint`, `--model`, and/or `--api_key`. Lists of length 1 are broadcast; non-singleton lists must share the same length.

```bash
uv run benchmark \
  --endpoint http://a:8000/v1/completions,http://b:8000/v1/completions \
  --model llama-8b,mistral-7b \
  --api_key EMPTY \
  --concurrent 4 --trials 3
```

Each config runs sequentially with the same flags; a summary table appears at the end:

```
=================================================================
Comparison
=================================================================
#  model      endpoint                          req  fail  tok/s   p50_lat  p95_lat  p99_lat  mean_pT
-  ---------  --------------------------------  ---  ----  ------  -------  -------  -------  -------
1  llama-8b   http://a:8000/v1/completions       12     0  158.42    0.821    1.118    1.144       12
2  mistral-7b http://b:8000/v1/completions       12     0  142.07    0.913    1.241    1.288       12
```

When `--output PATH.jsonl` is set in matrix mode, per-config logs are written to `PATH.0.jsonl`, `PATH.1.jsonl`, ...

---

## Output formats

`--output runs.jsonl` writes one JSON object per request:

```json
{"trial": 1, "request_index": 0, "prompt_chars": 47, "prompt_tokens": 12, "output_tokens": 128, "latency_s": 0.821, "ttft_s": null, "scheduled_offset_s": null, "queue_wait_s": null, "ok": true, "estimated": false, "error": ""}
```

`--output runs.csv` writes the same schema with a header row. Field semantics:

- `prompt_chars` / `prompt_tokens` — measured by the active tokenizer.
- `output_tokens` — server-reported `usage.completion_tokens` when present, otherwise estimated from text length (`estimated: true`).
- `latency_s` — wall time from request send to response complete.
- `ttft_s` — set only on streaming successes; `null` otherwise.
- `scheduled_offset_s` / `queue_wait_s` — set only in open-loop QPS mode.
- `error` — empty string on success.

Warmup requests are intentionally excluded from the log.

---

## Prompt control

By default, each request draws a random prompt from `src/benchmark/questions.json` (100 prompts spanning explanation, code, creative writing, science). Override:

- `--prompt "..."` — same prompt every request.
- `--questions_file path.json` — your own JSON array of strings.
- `--questions_file ""` — disable random pool, falls back to a built-in default prompt.
- `--seed 42` — reproducible random picks.

For apples-to-apples comparison across runs, **fix the input length** with `--prompt_tokens`:

```bash
uv run benchmark --prompt_tokens 256 --max_tokens 256
```

This pads or truncates every prompt to exactly 256 tokens (as counted by the active tokenizer) before sending, so output throughput isn't biased by prompt-length variation.

---

## Tokenizer

The tokenizer is used both for **prompt-token counting** (logged per request) and for the **output-token fallback** when the server omits `usage.completion_tokens`.

- `--tokenizer auto` (default) — tiktoken `cl100k_base` if installed; otherwise `chars/4` with a one-time stderr note.
- `--tokenizer chars4` — `len(text) // 4`. Fast, ASCII-biased, no dependency.
- `--tokenizer tiktoken:<encoding>` — explicit, e.g. `tiktoken:o200k_base` for GPT-4o-family encodings.

Install tiktoken with `uv sync --extra tiktoken`.

The server's reported `usage.completion_tokens` always wins over the local estimate — the local tokenizer only kicks in when the server omits it.

---

## Examples

**Pinned-length comparison of two backends:**

```bash
uv run benchmark \
  --endpoint http://vllm:8000/v1/completions,http://sglang:8000/v1/completions \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --api_key EMPTY \
  --prompt_tokens 512 --max_tokens 256 \
  --concurrent 8 --trials 5 \
  --output bake-off.jsonl
```

**TTFT under load:**

```bash
uv run benchmark --stream true --qps 20 --duration 60 \
  --max_tokens 256 --output ttft.jsonl
```

**Single-stream peak generation rate (decode tok/s):**

```bash
uv run benchmark --stream true --concurrent 1 --trials 10 \
  --max_tokens 512 --warmup 2
```

**OpenAI-compatible hosted gateway (strict — disable vLLM extension):**

```bash
uv run benchmark \
  --endpoint https://api.openai.com/v1/chat/completions \
  --api_key $OPENAI_API_KEY \
  --model gpt-4o-mini \
  --mode chat --ignore_eos false \
  --concurrent 2 --trials 5
```

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | All requests succeeded. |
| `1` | No successful trials (or no arrivals dispatched in QPS mode). |
| `2` | At least one request failed; throughput numbers are degraded. |
| `130` | Interrupted with Ctrl-C. |

In matrix mode the process exits with `max(per-config codes)`.

---

## Development

```bash
uv sync --group dev
uv run ruff check src tests                            # lint
uv run pytest                                          # full suite (~130 tests)
uv run pytest -v
uv run pytest tests/test_streaming.py                  # single file
uv run pytest tests/test_percentile.py::test_percentile_endpoints_match_min_and_max
uv run pytest -k stream                                # by keyword
```

Lint config lives in `[tool.ruff]` / `[tool.ruff.lint]` in `pyproject.toml`. CI runs `ruff check` before `pytest` on every push and PR.

### Project layout

```
src/benchmark/
  __init__.py          # public re-exports
  __main__.py          # `python -m benchmark` entry
  cli.py               # argparse + env layering + matrix dispatch
  client.py            # benchmark_request, SSE consumer, RequestResult
  matrix.py            # broadcast + comparison table
  prompts.py           # prompt provider + questions loader
  records.py           # RECORD_FIELDS + JSONL/CSV writer
  runner.py            # run_benchmark (closed-loop + open-loop dispatch)
  stats.py             # percentiles, Poisson schedule, percentile printers
  summary.py           # RunSummary dataclass
  tokenizer.py         # Tokenizer protocol + chars/4 / tiktoken impls
  questions.json       # default 100-prompt pool
tests/                 # pytest suite, fully mocked (no network)
.github/workflows/     # CI: pytest matrix on Python 3.11 + 3.12
```

`tests/test_hygiene.py` enforces: no literal API-key-shaped tokens in tracked files, no unused top-level imports in package modules.

CI runs on push and pull request via GitHub Actions.

---

## Caveats

- **`ignore_eos`/`min_tokens` are vLLM/SGLang extensions.** Hosted gateways (OpenAI, Together, etc.) may reject the request or silently ignore the field. Use `--ignore_eos false` against strict gateways.
- **Token counts.** Without `tiktoken` installed, the chars/4 fallback is ASCII-biased and underestimates non-English / code-heavy prompts.
- **Open-loop QPS.** The dispatcher sleeps until each scheduled target then submits. If the worker pool is saturated, requests queue inside the executor — surfaced as `queue_wait_s` per request and the `Dispatcher queue wait` summary row.
- **No retries.** A 5xx or network error counts as a failure; throughput numbers are degraded accordingly. Investigate the underlying cause rather than masking with retries inside the benchmark.
