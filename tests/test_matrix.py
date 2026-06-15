
import pytest

from benchmark import (
    RequestResult,
    RunSummary,
    _broadcast_configs,
    _print_comparison_table,
    run_benchmark,
)

# ---------- _broadcast_configs ----------


def test_broadcast_single_triple():
    cfgs = _broadcast_configs(["e"], ["m"], ["k"])
    assert cfgs == [("e", "m", "k")]


def test_broadcast_singleton_endpoint_with_multi_models():
    cfgs = _broadcast_configs(["e"], ["m1", "m2", "m3"], ["k"])
    assert cfgs == [("e", "m1", "k"), ("e", "m2", "k"), ("e", "m3", "k")]


def test_broadcast_matching_multi_lists():
    cfgs = _broadcast_configs(
        ["e1", "e2"], ["m1", "m2"], ["k1", "k2"],
    )
    assert cfgs == [("e1", "m1", "k1"), ("e2", "m2", "k2")]


def test_broadcast_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="mismatched"):
        _broadcast_configs(["e1", "e2"], ["m1", "m2", "m3"], ["k"])
    with pytest.raises(ValueError, match="mismatched"):
        _broadcast_configs(["e"], ["m1", "m2"], ["k1", "k2", "k3"])


# ---------- summary_out population ----------


def _ok_result(*args, **kwargs):
    return RequestResult(tokens=10, latency=0.1, ok=True, estimated=False)


def test_summary_out_appends_runsummary_in_closed_loop(monkeypatch):
    monkeypatch.setattr("benchmark.runner.benchmark_request", _ok_result)
    summaries = []
    rc = run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="model-A",
        prompt_provider=lambda: "p",
        prompt_label="fixed",
        max_tokens=10,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=2,
        trials=3,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        summary_out=summaries,
    )
    assert len(summaries) == 1
    s = summaries[0]
    assert isinstance(s, RunSummary)
    assert s.endpoint == "http://x/v1/completions"
    assert s.model == "model-A"
    assert s.exit_code == rc == 0
    assert s.total_requests == 6
    assert s.failed_requests == 0
    assert s.total_tokens == 60
    assert s.p50_latency_s is not None
    assert s.achieved_qps is None  # closed-loop


def test_summary_out_includes_achieved_qps_and_ttft_when_open_loop(monkeypatch):
    def streaming_fake(*args, **kwargs):
        return RequestResult(
            tokens=10, latency=0.1, ok=True, estimated=False, ttft=0.02,
        )

    monkeypatch.setattr("benchmark.runner.benchmark_request", streaming_fake)

    summaries = []
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
        trials=1,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        qps=500.0,
        duration=0.05,
        schedule_seed=42,
        stream=True,
        summary_out=summaries,
    )
    assert len(summaries) == 1
    s = summaries[0]
    assert s.achieved_qps is not None and s.achieved_qps > 0
    assert s.mean_ttft_s == pytest.approx(0.02, abs=1e-9)


def test_summary_populated_even_when_no_successes(monkeypatch):
    def fail(*args, **kwargs):
        return RequestResult(
            tokens=0, latency=0.01, ok=False, estimated=False, error="boom",
        )

    monkeypatch.setattr("benchmark.runner.benchmark_request", fail)

    summaries = []
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
        summary_out=summaries,
    )
    # No successes ⇒ exit code 2 (failures), summary still emitted with zeros.
    assert len(summaries) == 1
    s = summaries[0]
    assert s.exit_code == rc
    assert s.failed_requests == 2
    assert s.p50_latency_s is None


# ---------- _print_comparison_table ----------


def _make_summary(**overrides) -> RunSummary:
    base = dict(
        endpoint="http://e",
        model="m",
        exit_code=0,
        total_requests=10,
        failed_requests=0,
        total_tokens=100,
        total_duration_s=1.0,
        aggregate_tok_s=100.0,
        p50_latency_s=0.1,
        p95_latency_s=0.2,
        p99_latency_s=0.3,
        mean_ttft_s=None,
        mean_prompt_tokens=42.0,
        achieved_qps=None,
    )
    base.update(overrides)
    return RunSummary(**base)


def test_comparison_table_includes_all_runs_and_columns(capsys):
    summaries = [
        _make_summary(model="modelA", endpoint="http://a", aggregate_tok_s=120.5),
        _make_summary(model="modelB", endpoint="http://b", aggregate_tok_s=98.7),
    ]
    _print_comparison_table(summaries)
    out = capsys.readouterr().out
    assert "Comparison" in out
    assert "modelA" in out
    assert "modelB" in out
    assert "120.50" in out
    assert "98.70" in out
    # No TTFT or achv_qps columns when no summary has them.
    assert "mean_TTFT" not in out
    assert "achv_qps" not in out


def test_comparison_table_adds_ttft_column_when_any_run_streamed(capsys):
    summaries = [
        _make_summary(model="A", mean_ttft_s=0.05),
        _make_summary(model="B", mean_ttft_s=None),  # mixed: B not streamed
    ]
    _print_comparison_table(summaries)
    out = capsys.readouterr().out
    assert "mean_TTFT" in out
    assert "0.050" in out
    # The non-streamed row prints "-" for the TTFT cell.
    assert "-" in out.split("B")[1]


def test_comparison_table_adds_qps_column_when_any_run_open_loop(capsys):
    summaries = [
        _make_summary(model="A", achieved_qps=42.0),
        _make_summary(model="B", achieved_qps=None),
    ]
    _print_comparison_table(summaries)
    out = capsys.readouterr().out
    assert "achv_qps" in out
    assert "42.00" in out


def test_comparison_table_handles_empty_summaries(capsys):
    _print_comparison_table([])
    out = capsys.readouterr().out
    assert out == ""  # no-op


# ---------- end-to-end: matrix via two run_benchmark calls (no CLI plumbing) ----------


def test_two_back_to_back_runs_each_append_their_own_summary(monkeypatch):
    monkeypatch.setattr("benchmark.runner.benchmark_request", _ok_result)
    summaries: list[RunSummary] = []

    for model in ("a", "b", "c"):
        run_benchmark(
            endpoint=f"http://x/v1/{model}",
            api_key="k",
            model=model,
            prompt_provider=lambda: "p",
            prompt_label="fixed",
            max_tokens=10,
            temperature=0.0,
            top_p=1.0,
            concurrent_requests=1,
            trials=1,
            warmup=0,
            timeout=30.0,
            mode="completions",
            ignore_eos=False,
            summary_out=summaries,
        )

    assert [s.model for s in summaries] == ["a", "b", "c"]
    assert all(s.exit_code == 0 for s in summaries)
