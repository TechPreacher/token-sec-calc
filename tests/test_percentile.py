import math

import pytest

from benchmark import RequestResult, _percentile, run_benchmark

# ---------- _percentile pure math ----------


def test_percentile_empty_returns_zero():
    assert _percentile([], 50) == 0.0
    assert _percentile([], 99) == 0.0


def test_percentile_single_value():
    assert _percentile([7.5], 0) == 7.5
    assert _percentile([7.5], 50) == 7.5
    assert _percentile([7.5], 100) == 7.5


def test_percentile_endpoints_match_min_and_max():
    vs = [3.0, 1.0, 4.0, 1.5, 9.0, 2.0, 6.0]
    assert _percentile(vs, 0) == min(vs)
    assert _percentile(vs, 100) == max(vs)


def test_percentile_linear_interpolation_matches_numpy_default():
    """Hand-computed: values 1..100, indices 0..99.
    p50 → k = 99*0.5 = 49.5 → 50 + 0.5*(51-50) = 50.5
    p90 → k = 99*0.9 = 89.1 → 90 + 0.1*(91-90) = 90.1
    p95 → k = 99*0.95 = 94.05 → 95 + 0.05*(96-95) = 95.05
    p99 → k = 99*0.99 = 98.01 → 99 + 0.01*(100-99) = 99.01
    """
    vs = [float(i) for i in range(1, 101)]
    assert math.isclose(_percentile(vs, 50), 50.5, abs_tol=1e-9)
    assert math.isclose(_percentile(vs, 90), 90.1, abs_tol=1e-9)
    assert math.isclose(_percentile(vs, 95), 95.05, abs_tol=1e-9)
    assert math.isclose(_percentile(vs, 99), 99.01, abs_tol=1e-9)


def test_percentile_does_not_mutate_input():
    vs = [5.0, 1.0, 3.0, 2.0, 4.0]
    before = list(vs)
    _percentile(vs, 50)
    assert vs == before


def test_percentile_handles_duplicates():
    vs = [1.0] * 10
    assert _percentile(vs, 50) == 1.0
    assert _percentile(vs, 99) == 1.0


def test_percentile_two_values():
    # p50 of [10, 20] → halfway → 15
    assert _percentile([10.0, 20.0], 50) == 15.0


# ---------- run_benchmark prints percentile rows ----------


def _result(latency: float, tokens: int = 10) -> RequestResult:
    return RequestResult(tokens=tokens, latency=latency, ok=True, estimated=False)


@pytest.fixture
def fixed_latencies(monkeypatch):
    """Make every benchmark_request return a predictable RequestResult."""
    latencies = iter([
        0.10, 0.20, 0.30, 0.40, 0.50,
        0.60, 0.70, 0.80, 0.90, 1.00,
    ])

    def fake_request(*args, **kwargs):
        return _result(next(latencies))

    monkeypatch.setattr("benchmark.runner.benchmark_request", fake_request)


def test_run_benchmark_prints_per_request_percentiles(fixed_latencies, capsys):
    rc = run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: "p",
        prompt_label="fixed",
        max_tokens=10,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=2,
        trials=5,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Per-request latency (s)" in out
    assert "p50=" in out
    assert "p90=" in out
    assert "p95=" in out
    assert "p99=" in out
    assert "Per-request latency / token (s)" in out


def test_run_benchmark_omits_percentiles_when_no_successes(monkeypatch, capsys):
    def all_fail(*args, **kwargs):
        return RequestResult(tokens=0, latency=0.01, ok=False, estimated=False, error="boom")

    monkeypatch.setattr("benchmark.runner.benchmark_request", all_fail)

    rc = run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: "p",
        prompt_label="fixed",
        max_tokens=10,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=1,
        trials=2,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
    )
    out = capsys.readouterr().out
    # No successful requests → no percentile section.
    assert "Per-request latency (s)" not in out
    # Exit code is 2 (failures) when at least one trial ran a request.
    assert rc == 2


def test_run_benchmark_skips_normalized_percentiles_when_all_tokens_zero(monkeypatch, capsys):
    def zero_token_ok(*args, **kwargs):
        # Successful response but server reported 0 tokens (and no text to estimate from).
        return RequestResult(tokens=0, latency=0.05, ok=True, estimated=False)

    monkeypatch.setattr("benchmark.runner.benchmark_request", zero_token_ok)

    run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: "p",
        prompt_label="fixed",
        max_tokens=10,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=1,
        trials=3,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
    )
    out = capsys.readouterr().out
    # Raw-latency percentile row still appears (tokens=0 but request succeeded).
    assert "Per-request latency (s)" in out
    # Normalized row should NOT appear because no req_norm_latencies were collected.
    assert "Per-request latency / token (s)" not in out
