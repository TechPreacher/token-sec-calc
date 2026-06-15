"""Aggregate stats for one benchmark run, suitable for cross-config compare."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunSummary:
    endpoint: str
    model: str
    exit_code: int
    total_requests: int = 0
    failed_requests: int = 0
    total_tokens: int = 0
    total_duration_s: float = 0.0
    aggregate_tok_s: float = 0.0
    p50_latency_s: Optional[float] = None
    p95_latency_s: Optional[float] = None
    p99_latency_s: Optional[float] = None
    mean_ttft_s: Optional[float] = None
    mean_prompt_tokens: Optional[float] = None
    achieved_qps: Optional[float] = None
