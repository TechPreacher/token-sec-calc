import json
from pathlib import Path

import pytest

from benchmark import (
    RECORD_FIELDS,
    RequestResult,
    _poisson_arrival_offsets,
    run_benchmark,
)

# ---------- _poisson_arrival_offsets pure math ----------


def test_offsets_validate_args():
    with pytest.raises(ValueError, match="qps"):
        _poisson_arrival_offsets(0.0, 1.0, seed=0)
    with pytest.raises(ValueError, match="qps"):
        _poisson_arrival_offsets(-5.0, 1.0, seed=0)
    with pytest.raises(ValueError, match="duration"):
        _poisson_arrival_offsets(10.0, 0.0, seed=0)
    with pytest.raises(ValueError, match="duration"):
        _poisson_arrival_offsets(10.0, -1.0, seed=0)


def test_offsets_are_monotonic_and_within_duration():
    offs = _poisson_arrival_offsets(qps=100.0, duration=1.0, seed=42)
    assert offs == sorted(offs)
    assert all(0.0 < t < 1.0 for t in offs)


def test_offsets_are_reproducible_with_same_seed():
    a = _poisson_arrival_offsets(qps=50.0, duration=2.0, seed=7)
    b = _poisson_arrival_offsets(qps=50.0, duration=2.0, seed=7)
    assert a == b


def test_offsets_diverge_with_different_seeds():
    a = _poisson_arrival_offsets(qps=50.0, duration=2.0, seed=7)
    b = _poisson_arrival_offsets(qps=50.0, duration=2.0, seed=999)
    assert a != b


def test_offsets_count_concentrates_near_qps_times_duration():
    """Sanity: averaging across seeds, count ≈ qps × duration.
    Single-seed runs can vary; allow generous bounds."""
    counts = [
        len(_poisson_arrival_offsets(qps=200.0, duration=1.0, seed=s))
        for s in range(20)
    ]
    avg = sum(counts) / len(counts)
    # Expected mean 200; loose bounds to keep test stable.
    assert 150 <= avg <= 250


# ---------- run_benchmark open-loop integration ----------


def _ok_result_factory(latency: float = 0.0, tokens: int = 10):
    """Return a fake benchmark_request that emits deterministic results."""

    def fake(*args, **kwargs):
        return RequestResult(tokens=tokens, latency=latency, ok=True, estimated=False)

    return fake


def test_open_loop_writes_per_request_records_with_schedule_metadata(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("benchmark.runner.benchmark_request", _ok_result_factory(latency=0.0, tokens=5))

    out = tmp_path / "qps.jsonl"
    rc = run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: "ping",
        prompt_label="fixed",
        max_tokens=5,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=1,
        trials=1,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        output_path=str(out),
        qps=500.0,
        duration=0.05,
        schedule_seed=42,
    )
    assert rc == 0

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) > 0
    for row in rows:
        assert set(row.keys()) == set(RECORD_FIELDS)
        assert row["scheduled_offset_s"] is not None
        assert row["queue_wait_s"] is not None
        assert row["trial"] == 1
        assert row["ok"] is True
    offsets = [r["scheduled_offset_s"] for r in rows]
    assert offsets == sorted(offsets)
    indices = [r["request_index"] for r in rows]
    assert indices == list(range(len(rows)))


def test_closed_loop_records_have_null_schedule_fields(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("benchmark.runner.benchmark_request", _ok_result_factory(latency=0.0, tokens=5))

    out = tmp_path / "closed.jsonl"
    run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: "p",
        prompt_label="fixed",
        max_tokens=5,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=2,
        trials=2,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        output_path=str(out),
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    for row in rows:
        assert row["scheduled_offset_s"] is None
        assert row["queue_wait_s"] is None


def test_open_loop_prints_open_loop_header_and_queue_wait_section(monkeypatch, capsys):
    monkeypatch.setattr("benchmark.runner.benchmark_request", _ok_result_factory(latency=0.0, tokens=5))

    run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: "p",
        prompt_label="fixed",
        max_tokens=5,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=1,
        trials=1,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        qps=500.0,
        duration=0.05,
        schedule_seed=42,
    )
    out = capsys.readouterr().out
    assert "Mode: open-loop Poisson" in out
    assert "Open-loop results" in out
    assert "Achieved request rate" in out
    assert "Dispatcher queue wait" in out


def test_closed_loop_header_still_present_when_qps_absent(monkeypatch, capsys):
    monkeypatch.setattr("benchmark.runner.benchmark_request", _ok_result_factory(latency=0.0, tokens=5))

    run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: "p",
        prompt_label="fixed",
        max_tokens=5,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=1,
        trials=1,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
    )
    out = capsys.readouterr().out
    assert "Mode: closed-loop" in out
    assert "Mode: open-loop" not in out


def test_qps_without_duration_raises_in_run_benchmark():
    with pytest.raises(ValueError, match="duration"):
        run_benchmark(
            endpoint="http://x/v1/completions",
            api_key="k",
            model="m",
            prompt_provider=lambda: "p",
            prompt_label="fixed",
            max_tokens=5,
            temperature=0.0,
            top_p=1.0,
            concurrent_requests=1,
            trials=1,
            warmup=0,
            timeout=30.0,
            mode="completions",
            ignore_eos=False,
            qps=10.0,
            duration=None,
        )


def test_open_loop_max_concurrency_override(monkeypatch, capsys):
    monkeypatch.setattr("benchmark.runner.benchmark_request", _ok_result_factory(latency=0.0, tokens=1))

    run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: "p",
        prompt_label="fixed",
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=1,
        trials=1,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        qps=100.0,
        duration=0.02,
        max_concurrency=7,
        schedule_seed=42,
    )
    out = capsys.readouterr().out
    assert "max_concurrency=7" in out


def test_open_loop_returns_1_when_schedule_empty(monkeypatch, capsys):
    monkeypatch.setattr("benchmark.runner.benchmark_request", _ok_result_factory(latency=0.0, tokens=1))

    # qps × duration ≈ 0.01 expected arrivals; with seed that produces 0 it errors out.
    # Use a tiny duration that almost guarantees zero arrivals at this qps/seed.
    rc = run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: "p",
        prompt_label="fixed",
        max_tokens=1,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=1,
        trials=1,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        qps=0.001,
        duration=0.0001,  # 0.001 × 0.0001 = 1e-7 expected
        schedule_seed=0,
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "No arrivals" in err
