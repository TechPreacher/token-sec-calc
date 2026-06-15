"""Public surface for the benchmark package.

Tests and external scripts can keep using `from benchmark import X` for any
of the symbols below. Module internals live in submodules organized by
concern: tokenizer / client / runner / stats / records / prompts / matrix /
summary / cli.
"""

from __future__ import annotations

from .cli import _env_float, _env_int, _env_str, main
from .client import (
    RequestResult,
    _consume_sse_stream,
    _is_chat_endpoint,
    benchmark_request,
)
from .matrix import _broadcast_configs, _print_comparison_table
from .prompts import (
    DEFAULT_PROMPT,
    DEFAULT_QUESTIONS_FILE,
    _build_prompt_provider,
    _load_questions,
)
from .records import RECORD_FIELDS, _record_writer
from .runner import run_benchmark
from .stats import (
    _percentile,
    _poisson_arrival_offsets,
    _print_prompt_token_stats,
    _print_request_percentiles,
)
from .summary import RunSummary
from .tokenizer import (
    CharsTokenizer,
    TikTokenTokenizer,
    Tokenizer,
    _build_tokenizer,
    _estimate_tokens,
    _normalize_prompt,
)

__all__ = [
    "CharsTokenizer",
    "DEFAULT_PROMPT",
    "DEFAULT_QUESTIONS_FILE",
    "RECORD_FIELDS",
    "RequestResult",
    "RunSummary",
    "TikTokenTokenizer",
    "Tokenizer",
    "_broadcast_configs",
    "_build_prompt_provider",
    "_build_tokenizer",
    "_consume_sse_stream",
    "_env_float",
    "_env_int",
    "_env_str",
    "_estimate_tokens",
    "_is_chat_endpoint",
    "_load_questions",
    "_normalize_prompt",
    "_percentile",
    "_poisson_arrival_offsets",
    "_print_comparison_table",
    "_print_prompt_token_stats",
    "_print_request_percentiles",
    "_record_writer",
    "benchmark_request",
    "main",
    "run_benchmark",
]
