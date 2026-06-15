"""Helpers for the multi-endpoint compare mode: arg broadcasting + the
side-by-side summary table."""

from __future__ import annotations

from typing import List

from .summary import RunSummary


def _broadcast_configs(
    endpoints: List[str], models: List[str], api_keys: List[str],
) -> List[tuple[str, str, str]]:
    """Zip endpoint/model/api_key lists into (endpoint, model, api_key) triples.

    A list of length 1 is broadcast to match the longest list. All non-singleton
    lists must share the same length; otherwise ValueError is raised.
    """
    lengths = {len(endpoints), len(models), len(api_keys)} - {1}
    if len(lengths) > 1:
        raise ValueError(
            "mismatched list lengths for --endpoint/--model/--api_key: "
            f"endpoints={len(endpoints)}, models={len(models)}, "
            f"api_keys={len(api_keys)} (non-singleton lists must match)"
        )
    n = max(len(endpoints), len(models), len(api_keys))

    def expand(seq: List[str]) -> List[str]:
        return seq * n if len(seq) == 1 else seq

    return list(zip(expand(endpoints), expand(models), expand(api_keys)))


def _print_comparison_table(summaries: List[RunSummary]) -> None:
    """Render a side-by-side comparison table across runs."""
    if not summaries:
        return
    has_ttft = any(s.mean_ttft_s is not None for s in summaries)
    has_qps = any(s.achieved_qps is not None for s in summaries)

    headers = ["#", "model", "endpoint", "req", "fail",
               "tok/s", "p50_lat", "p95_lat", "p99_lat", "mean_pT"]
    if has_ttft:
        headers.append("mean_TTFT")
    if has_qps:
        headers.append("achv_qps")

    rows: List[List[str]] = []
    for i, s in enumerate(summaries):
        row = [
            str(i + 1),
            s.model,
            s.endpoint,
            str(s.total_requests),
            str(s.failed_requests),
            f"{s.aggregate_tok_s:.2f}",
            "-" if s.p50_latency_s is None else f"{s.p50_latency_s:.3f}",
            "-" if s.p95_latency_s is None else f"{s.p95_latency_s:.3f}",
            "-" if s.p99_latency_s is None else f"{s.p99_latency_s:.3f}",
            "-" if s.mean_prompt_tokens is None else f"{s.mean_prompt_tokens:.0f}",
        ]
        if has_ttft:
            row.append("-" if s.mean_ttft_s is None else f"{s.mean_ttft_s:.3f}")
        if has_qps:
            row.append("-" if s.achieved_qps is None else f"{s.achieved_qps:.2f}")
        rows.append(row)

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    sep = "  ".join("-" * w for w in widths)

    def fmt(cells: List[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print("\n" + "=" * len(sep))
    print("Comparison")
    print("=" * len(sep))
    print(fmt(headers))
    print(sep)
    for r in rows:
        print(fmt(r))
