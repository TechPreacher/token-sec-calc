"""Pure stat helpers and human-readable printers shared between the two run
modes (closed-loop trial × concurrent and open-loop Poisson QPS)."""

from __future__ import annotations

import random
from typing import List, Optional


def _percentile(values: List[float], p: float) -> float:
    """Linear-interpolation percentile. Matches numpy's default method.

    `p` is 0..100. Empty input returns 0.0. Single value returns itself.
    """
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _poisson_arrival_offsets(qps: float, duration: float, seed: int) -> List[float]:
    """Generate Poisson-process arrival offsets (seconds from t=0) for an
    open-loop benchmark with target rate `qps` and window `duration`.

    Inter-arrival times are exponential with mean 1/qps. Seed semantics match
    the prompt provider: `seed >= 0` is reproducible, `seed < 0` is
    nondeterministic. Returns the (possibly empty) list of offsets strictly
    less than `duration`, in ascending order.
    """
    if qps <= 0:
        raise ValueError(f"qps must be > 0 (got {qps})")
    if duration <= 0:
        raise ValueError(f"duration must be > 0 (got {duration})")
    rng = random.Random(seed) if seed >= 0 else random.Random()
    offsets: List[float] = []
    t = 0.0
    while True:
        t += rng.expovariate(qps)
        if t >= duration:
            return offsets
        offsets.append(t)


def _print_request_percentiles(
    req_latencies: List[float],
    req_norm_latencies: List[float],
    req_ttfts: List[float],
    queue_waits: Optional[List[float]] = None,
) -> None:
    if not req_latencies:
        return
    p_levels = (50, 90, 95, 99)
    lat_pcts = [_percentile(req_latencies, p) for p in p_levels]
    print(f"  Per-request latency (s) over {len(req_latencies)} successes:")
    print("    " + "  ".join(f"p{p}={v:.3f}" for p, v in zip(p_levels, lat_pcts)))
    if req_norm_latencies:
        norm_pcts = [_percentile(req_norm_latencies, p) for p in p_levels]
        print("  Per-request latency / token (s):")
        print("    " + "  ".join(f"p{p}={v:.4f}" for p, v in zip(p_levels, norm_pcts)))
    if req_ttfts:
        ttft_pcts = [_percentile(req_ttfts, p) for p in p_levels]
        mean_ttft = sum(req_ttfts) / len(req_ttfts)
        print(f"  Time-to-first-token (s) over {len(req_ttfts)} streamed successes:")
        print("    " + "  ".join(f"p{p}={v:.3f}" for p, v in zip(p_levels, ttft_pcts))
              + f"  mean={mean_ttft:.3f}")
    if queue_waits:
        qw_pcts = [_percentile(queue_waits, p) for p in p_levels]
        mean_qw = sum(queue_waits) / len(queue_waits)
        print(f"  Dispatcher queue wait (s) over {len(queue_waits)} requests:")
        print("    " + "  ".join(f"p{p}={v:.4f}" for p, v in zip(p_levels, qw_pcts))
              + f"  mean={mean_qw:.4f}")


def _print_prompt_token_stats(prompt_tokens_seen: List[int]) -> None:
    if not prompt_tokens_seen:
        return
    fs = [float(v) for v in prompt_tokens_seen]
    mean = sum(fs) / len(fs)
    p50 = _percentile(fs, 50)
    p99 = _percentile(fs, 99)
    print(
        f"  Prompt input tokens:  min={min(prompt_tokens_seen):>5d}  "
        f"mean={mean:6.1f}  max={max(prompt_tokens_seen):>5d}  "
        f"p50={p50:.0f}  p99={p99:.0f}"
    )
