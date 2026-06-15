"""argparse front-end, env-variable layering, and multi-config dispatch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from .matrix import _broadcast_configs, _print_comparison_table
from .prompts import (
    DEFAULT_PROMPT,
    DEFAULT_QUESTIONS_FILE,
    _build_prompt_provider,
)
from .runner import run_benchmark
from .summary import RunSummary
from .tokenizer import _build_tokenizer, _normalize_prompt


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return raw


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Measure tokens per second against an LLM inference endpoint. "
        "All flags may be supplied via a .env file (see .env.example). "
        "CLI flags override environment values."
    )
    parser.add_argument("--endpoint", default=_env_str("ENDPOINT"),
                        help="Inference endpoint URL. Comma-separated values "
                             "enable matrix mode (env: ENDPOINT)")
    parser.add_argument("--api_key", default=_env_str("API_KEY"),
                        help="API key for authentication. May be comma-separated "
                             "(env: API_KEY)")
    parser.add_argument("--model", default=_env_str("MODEL"),
                        help="Model name to benchmark. May be comma-separated "
                             "(env: MODEL)")
    parser.add_argument("--prompt", default=_env_str("PROMPT"),
                        help="Fixed prompt for every request. If omitted and a questions "
                             "file is available, a random prompt is picked per request "
                             "(env: PROMPT)")
    parser.add_argument("--questions_file",
                        default=_env_str("QUESTIONS_FILE", DEFAULT_QUESTIONS_FILE),
                        help="Path to JSON array of prompts. Used when --prompt/PROMPT is "
                             "unset. Pass empty string to disable (env: QUESTIONS_FILE)")
    parser.add_argument("--seed", type=int, default=_env_int("SEED", -1),
                        help="Random seed for question selection. -1 = nondeterministic "
                             "(env: SEED)")
    parser.add_argument("--max_tokens", type=int, default=_env_int("MAX_TOKENS", 100),
                        help="Maximum tokens to generate per request (env: MAX_TOKENS)")
    parser.add_argument("--concurrent", type=int, default=_env_int("CONCURRENT", 1),
                        help="Number of concurrent requests (env: CONCURRENT)")
    parser.add_argument("--trials", type=int, default=_env_int("TRIALS", 5),
                        help="Number of benchmark trials (env: TRIALS)")
    parser.add_argument("--warmup", type=int, default=_env_int("WARMUP", 1),
                        help="Number of warmup requests (env: WARMUP)")
    parser.add_argument("--temperature", type=float, default=_env_float("TEMPERATURE", 0.0),
                        help="Sampling temperature. Default 0 = greedy decoding for "
                             "reproducible token counts (env: TEMPERATURE)")
    parser.add_argument("--top_p", type=float, default=_env_float("TOP_P", 1.0),
                        help="Nucleus sampling top_p (env: TOP_P)")
    parser.add_argument("--timeout", type=float, default=_env_float("TIMEOUT", 120.0),
                        help="Per-request timeout seconds (env: TIMEOUT)")
    parser.add_argument("--mode", choices=("auto", "chat", "completions"),
                        default=_env_str("MODE", "auto"),
                        help="API mode: 'completions' uses {prompt}, 'chat' uses {messages}, "
                             "'auto' picks chat if URL contains /chat/completions (env: MODE)")
    parser.add_argument("--prompt_tokens", type=int,
                        default=_env_int("PROMPT_TOKENS", 0),
                        help="If > 0, pad/truncate every prompt to this many tokens "
                             "(as counted by the active tokenizer). Removes input-length "
                             "noise when comparing runs (env: PROMPT_TOKENS)")
    parser.add_argument("--qps", type=float, default=_env_float("QPS", 0.0),
                        help="Open-loop target requests/sec. When > 0, the run uses a "
                             "Poisson arrival schedule for --duration seconds instead "
                             "of the closed-loop trials × concurrent model (env: QPS)")
    parser.add_argument("--duration", type=float, default=_env_float("DURATION", 0.0),
                        help="Open-loop window in seconds. Required with --qps "
                             "(env: DURATION)")
    parser.add_argument("--max_concurrency", type=int,
                        default=_env_int("MAX_CONCURRENCY", 0),
                        help="Open-loop worker pool size. Defaults to max(32, "
                             "ceil(qps*2)) capped at 256. Ignored in closed-loop mode "
                             "(env: MAX_CONCURRENCY)")
    parser.add_argument("--tokenizer", default=_env_str("TOKENIZER", "auto"),
                        help="Tokenizer used for prompt-token logging and for the "
                             "output-token fallback when the server omits `usage`. "
                             "Values: 'auto' (tiktoken cl100k_base if installed, else "
                             "chars/4), 'chars4', or 'tiktoken:<encoding>' e.g. "
                             "'tiktoken:o200k_base' (env: TOKENIZER)")
    parser.add_argument("--stream", default=_env_str("STREAM", "false"),
                        choices=("true", "false"),
                        help="If true, use SSE streaming and measure time-to-first-token "
                             "(TTFT). Default false. Most useful for interactive serving "
                             "benchmarks (env: STREAM)")
    parser.add_argument("--output", default=_env_str("OUTPUT"),
                        help="Write a per-request record to PATH. Format selected by "
                             "extension: .jsonl (one JSON per line) or .csv. Warmup "
                             "requests are excluded (env: OUTPUT)")
    parser.add_argument("--ignore_eos", default=_env_str("IGNORE_EOS", "true"),
                        choices=("true", "false"),
                        help="If true (default), send 'ignore_eos' + 'min_tokens=max_tokens' "
                             "so every request emits exactly --max_tokens. vLLM/SGLang "
                             "extension; harmless on servers that ignore unknown fields. "
                             "Set false for OpenAI/strict gateways (env: IGNORE_EOS)")

    # Reference DEFAULT_PROMPT to keep the symbol exported from this module.
    _ = DEFAULT_PROMPT

    args = parser.parse_args()

    missing = [n for n in ("endpoint", "api_key", "model") if not getattr(args, n)]
    if missing:
        parser.error(
            f"missing required setting(s): {', '.join(missing)}. "
            "Provide via CLI flag or .env file."
        )

    for name, val in (("concurrent", args.concurrent), ("trials", args.trials),
                      ("max_tokens", args.max_tokens)):
        if val < 1:
            parser.error(f"--{name} must be >= 1 (got {val})")
    if args.warmup < 0:
        parser.error(f"--warmup must be >= 0 (got {args.warmup})")
    if args.qps > 0 and args.duration <= 0:
        parser.error("--qps requires --duration > 0")
    if args.qps < 0:
        parser.error(f"--qps must be >= 0 (got {args.qps})")
    if args.max_concurrency < 0:
        parser.error(f"--max_concurrency must be >= 0 (got {args.max_concurrency})")
    if args.prompt_tokens < 0:
        parser.error(f"--prompt_tokens must be >= 0 (got {args.prompt_tokens})")

    prompt_provider, prompt_label = _build_prompt_provider(
        explicit_prompt=args.prompt,
        questions_file=args.questions_file,
        seed=args.seed,
    )

    try:
        tokenizer = _build_tokenizer(args.tokenizer)
    except (ValueError, ImportError) as e:
        parser.error(f"--tokenizer: {e}")

    if args.prompt_tokens > 0:
        inner_provider = prompt_provider
        target_tokens = args.prompt_tokens

        def _normalized_provider() -> str:
            return _normalize_prompt(inner_provider(), target_tokens, tokenizer)

        prompt_provider = _normalized_provider
        prompt_label += f" (normalized to {target_tokens} tokens via {tokenizer.label})"

    try:
        configs = _broadcast_configs(
            [s.strip() for s in args.endpoint.split(",") if s.strip()],
            [s.strip() for s in args.model.split(",") if s.strip()],
            [s.strip() for s in args.api_key.split(",") if s.strip()],
        )
    except ValueError as e:
        parser.error(str(e))

    summaries: List[RunSummary] = []
    exit_codes: List[int] = []
    n = len(configs)

    try:
        for i, (endpoint, model, api_key) in enumerate(configs):
            run_output = args.output
            if run_output and n > 1:
                p = Path(run_output)
                run_output = str(p.with_name(f"{p.stem}.{i}{p.suffix}"))
            if n > 1:
                print(f"\n========== Config {i + 1}/{n}: {model} @ {endpoint} ==========")
            rc = run_benchmark(
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                prompt_provider=prompt_provider,
                prompt_label=prompt_label,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                concurrent_requests=args.concurrent,
                trials=args.trials,
                warmup=args.warmup,
                timeout=args.timeout,
                mode=args.mode,
                ignore_eos=(args.ignore_eos == "true"),
                output_path=run_output,
                stream=(args.stream == "true"),
                tokenizer=tokenizer,
                qps=(args.qps if args.qps > 0 else None),
                duration=(args.duration if args.qps > 0 else None),
                max_concurrency=(args.max_concurrency or None),
                schedule_seed=args.seed,
                summary_out=summaries,
            )
            exit_codes.append(rc)

        if n > 1:
            _print_comparison_table(summaries)
        sys.exit(max(exit_codes) if exit_codes else 0)
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user.", file=sys.stderr)
        sys.exit(130)
