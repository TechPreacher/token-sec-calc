"""Top-level benchmark driver. Routes between closed-loop trials and
open-loop Poisson QPS modes, accumulates per-request stats, and emits a
`RunSummary` via the optional `summary_out` list."""

from __future__ import annotations

import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List, Optional

from .client import RequestResult, _is_chat_endpoint, benchmark_request
from .records import _record_writer
from .stats import (
    _percentile,
    _poisson_arrival_offsets,
    _print_prompt_token_stats,
    _print_request_percentiles,
)
from .summary import RunSummary
from .tokenizer import CharsTokenizer, Tokenizer


def run_benchmark(
    endpoint: str,
    api_key: str,
    model: str,
    prompt_provider: Callable[[], str],
    prompt_label: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    concurrent_requests: int,
    trials: int,
    warmup: int,
    timeout: float,
    mode: str,
    ignore_eos: bool,
    output_path: Optional[str] = None,
    stream: bool = False,
    tokenizer: Optional[Tokenizer] = None,
    qps: Optional[float] = None,
    duration: Optional[float] = None,
    max_concurrency: Optional[int] = None,
    schedule_seed: int = -1,
    summary_out: Optional[List[RunSummary]] = None,
) -> int:
    if tokenizer is None:
        tokenizer = CharsTokenizer()
    open_loop = qps is not None and qps > 0
    if open_loop and (duration is None or duration <= 0):
        raise ValueError("--qps requires --duration > 0")

    api_kind = "chat/completions" if _is_chat_endpoint(endpoint, mode) else "completions"
    print(f"\nBenchmarking {endpoint}")
    print(f"API: {api_kind} (mode={mode})")
    print(f"Model: {model}")
    workers = 0
    if open_loop:
        assert qps is not None and duration is not None
        workers = max_concurrency or max(32, min(256, int(math.ceil(qps * 2))))
        print(f"Mode: open-loop Poisson (qps={qps}, duration={duration}s, "
              f"max_concurrency={workers})")
    else:
        print(f"Mode: closed-loop ({concurrent_requests} concurrent × {trials} trials)")
    if warmup > 0:
        print(f"Warmup: {warmup} request(s)")
    print(f"Max tokens per request: {max_tokens}"
          + (" (ignore_eos + min_tokens pinned)" if ignore_eos else ""))
    print(f"Sampling: temperature={temperature} top_p={top_p}")
    print(f"Streaming: {'on' if stream else 'off'}")
    print(f"Tokenizer: {tokenizer.label}")
    print(f"Prompt source: {prompt_label}")
    if output_path:
        print(f"Per-request log: {output_path}")
    print("-" * 60)

    def one() -> tuple[str, RequestResult]:
        p = prompt_provider()
        return p, benchmark_request(
            endpoint, api_key, model, p, max_tokens,
            temperature, top_p, timeout, mode, ignore_eos, stream, tokenizer,
        )

    if warmup > 0:
        print(f"Running {warmup} warmup request(s)...")
        for _ in range(warmup):
            _, r = one()
            if r.ok:
                tps = r.tokens / r.latency if r.latency > 0 else 0.0
                tag = " (est)" if r.estimated else ""
                print(f"  Warmup: {r.tokens} tokens in {r.latency:.2f}s ({tps:.2f} tok/s){tag}")
            else:
                print(f"  Warmup FAILED in {r.latency:.2f}s: {r.error}", file=sys.stderr)
        print()

    per_trial_tps: List[float] = []
    req_latencies: List[float] = []
    req_norm_latencies: List[float] = []
    req_ttfts: List[float] = []
    queue_waits: List[float] = []
    prompt_tokens_seen: List[int] = []
    total_tokens_all = 0
    total_duration_all = 0.0
    total_requests = 0
    failed_requests = 0
    estimated_requests = 0

    def _finalize(rc: int) -> int:
        aggregate = (total_tokens_all / total_duration_all) if total_duration_all > 0 else 0.0
        summary = RunSummary(
            endpoint=endpoint,
            model=model,
            exit_code=rc,
            total_requests=total_requests,
            failed_requests=failed_requests,
            total_tokens=total_tokens_all,
            total_duration_s=total_duration_all,
            aggregate_tok_s=aggregate,
            p50_latency_s=_percentile(req_latencies, 50) if req_latencies else None,
            p95_latency_s=_percentile(req_latencies, 95) if req_latencies else None,
            p99_latency_s=_percentile(req_latencies, 99) if req_latencies else None,
            mean_ttft_s=(sum(req_ttfts) / len(req_ttfts)) if req_ttfts else None,
            mean_prompt_tokens=(
                sum(prompt_tokens_seen) / len(prompt_tokens_seen)
                if prompt_tokens_seen else None
            ),
            achieved_qps=(
                total_requests / total_duration_all
                if (open_loop and total_duration_all > 0) else None
            ),
        )
        if summary_out is not None:
            summary_out.append(summary)
        return rc

    if open_loop:
        assert qps is not None and duration is not None
        offsets = _poisson_arrival_offsets(qps, duration, schedule_seed)
        if not offsets:
            print("No arrivals generated (qps × duration too small).", file=sys.stderr)
            return _finalize(1)

        def one_at(target_perf: float) -> tuple[float, str, RequestResult]:
            started = time.perf_counter()
            qw = max(0.0, started - target_perf)
            p, r = one()
            return qw, p, r

        with _record_writer(output_path) as write_record:
            run_start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                pending = []
                for i, off in enumerate(offsets):
                    target = run_start + off
                    delay = target - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    fut = executor.submit(one_at, target)
                    pending.append((i, off, fut))

                for i, off, fut in pending:
                    qw, prompt_text, r = fut.result()
                    total_requests += 1
                    if r.ok:
                        total_tokens_all += r.tokens
                        req_latencies.append(r.latency)
                        if r.tokens > 0:
                            req_norm_latencies.append(r.latency / r.tokens)
                        if r.ttft is not None:
                            req_ttfts.append(r.ttft)
                        if r.estimated:
                            estimated_requests += 1
                    else:
                        failed_requests += 1
                    queue_waits.append(qw)
                    p_tok = tokenizer.count(prompt_text)
                    prompt_tokens_seen.append(p_tok)

                    if write_record is not None:
                        write_record({
                            "trial": 1,
                            "request_index": i,
                            "prompt_chars": len(prompt_text),
                            "prompt_tokens": p_tok,
                            "output_tokens": r.tokens,
                            "latency_s": round(r.latency, 6),
                            "ttft_s": round(r.ttft, 6) if r.ttft is not None else None,
                            "scheduled_offset_s": round(off, 6),
                            "queue_wait_s": round(qw, 6),
                            "ok": r.ok,
                            "estimated": r.estimated,
                            "error": r.error or "",
                        })

            total_duration_all = time.perf_counter() - run_start

        print("-" * 60)
        if total_requests == 0:
            print("No requests dispatched.", file=sys.stderr)
            return _finalize(1)

        aggregate_tps = total_tokens_all / total_duration_all if total_duration_all > 0 else 0.0
        achieved_qps = total_requests / total_duration_all if total_duration_all > 0 else 0.0
        print(f"Open-loop results ({total_requests} requests, target qps={qps}):")
        print(f"  Aggregate throughput: {aggregate_tps:7.2f} tok/s   (total_tokens / wall_time)")
        print(f"  Achieved request rate:{achieved_qps:7.2f} req/s")
        print(f"  Total tokens:         {total_tokens_all}")
        print(f"  Total wall time:      {total_duration_all:.2f}s")
        _print_prompt_token_stats(prompt_tokens_seen)
        _print_request_percentiles(req_latencies, req_norm_latencies, req_ttfts, queue_waits)
    else:
        with _record_writer(output_path) as write_record:
            for trial in range(trials):
                trial_start = time.perf_counter()
                trial_tokens = 0
                trial_failures = 0

                with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
                    futures = {executor.submit(one): i for i in range(concurrent_requests)}
                    for future in as_completed(futures):
                        idx = futures[future]
                        prompt_text, r = future.result()
                        total_requests += 1
                        if r.ok:
                            trial_tokens += r.tokens
                            req_latencies.append(r.latency)
                            if r.tokens > 0:
                                req_norm_latencies.append(r.latency / r.tokens)
                            if r.ttft is not None:
                                req_ttfts.append(r.ttft)
                            if r.estimated:
                                estimated_requests += 1
                        else:
                            trial_failures += 1
                            failed_requests += 1

                        p_tok = tokenizer.count(prompt_text)
                        prompt_tokens_seen.append(p_tok)
                        if write_record is not None:
                            write_record({
                                "trial": trial + 1,
                                "request_index": idx,
                                "prompt_chars": len(prompt_text),
                                "prompt_tokens": p_tok,
                                "output_tokens": r.tokens,
                                "latency_s": round(r.latency, 6),
                                "ttft_s": round(r.ttft, 6) if r.ttft is not None else None,
                                "scheduled_offset_s": None,
                                "queue_wait_s": None,
                                "ok": r.ok,
                                "estimated": r.estimated,
                                "error": r.error or "",
                            })

                trial_duration = time.perf_counter() - trial_start
                total_tokens_all += trial_tokens
                total_duration_all += trial_duration

                if trial_duration > 0:
                    tps = trial_tokens / trial_duration
                    per_trial_tps.append(tps)
                    fail_note = f" | {trial_failures} FAILED" if trial_failures else ""
                    print(
                        f"Trial {trial + 1:2d}: {trial_tokens:5d} tokens | "
                        f"{trial_duration:5.2f}s | {tps:7.2f} tok/s{fail_note}"
                    )
                else:
                    print(f"Trial {trial + 1:2d}: zero duration (error?)")

        print("-" * 60)

        if not per_trial_tps:
            print("No successful trials.", file=sys.stderr)
            return _finalize(1)

        avg_tps = sum(per_trial_tps) / len(per_trial_tps)
        min_tps = min(per_trial_tps)
        max_tps = max(per_trial_tps)
        aggregate_tps = total_tokens_all / total_duration_all if total_duration_all > 0 else 0.0

        print(f"Results over {trials} trials ({total_requests} requests):")
        print(f"  Aggregate throughput: {aggregate_tps:7.2f} tok/s   (total_tokens / total_wall_time)")
        print(f"  Mean of per-trial:    {avg_tps:7.2f} tok/s")
        print(f"  Min / Max per-trial:  {min_tps:7.2f} / {max_tps:7.2f} tok/s")
        print(f"  Total tokens:         {total_tokens_all}")
        print(f"  Total wall time:      {total_duration_all:.2f}s")
        _print_prompt_token_stats(prompt_tokens_seen)
        _print_request_percentiles(req_latencies, req_norm_latencies, req_ttfts)

    if failed_requests:
        print(
            f"  WARNING: {failed_requests}/{total_requests} requests failed — "
            "throughput numbers are degraded by failures.",
            file=sys.stderr,
        )
    if estimated_requests:
        print(
            f"  NOTE: {estimated_requests}/{total_requests} responses lacked usage stats; "
            "tokens estimated from text length (~4 chars/token).",
            file=sys.stderr,
        )

    return _finalize(0 if failed_requests == 0 else 2)
