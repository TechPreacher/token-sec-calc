import csv
import json
from pathlib import Path

import pytest

from benchmark import (
    RECORD_FIELDS,
    RequestResult,
    _record_writer,
    run_benchmark,
)

# ---------- _record_writer unit tests ----------


def test_writer_none_for_empty_path():
    with _record_writer(None) as w:
        assert w is None
    with _record_writer("") as w:
        assert w is None


def test_writer_jsonl_emits_one_object_per_line(tmp_path: Path):
    out = tmp_path / "log.jsonl"
    with _record_writer(str(out)) as w:
        assert w is not None
        w({"trial": 1, "request_index": 0, "ok": True})
        w({"trial": 1, "request_index": 1, "ok": False})

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    rows = [json.loads(line) for line in lines]
    assert rows[0]["trial"] == 1
    assert rows[1]["ok"] is False


def test_writer_csv_writes_header_then_rows(tmp_path: Path):
    out = tmp_path / "log.csv"
    with _record_writer(str(out)) as w:
        assert w is not None
        w({
            "trial": 1, "request_index": 0, "prompt_chars": 12,
            "output_tokens": 50, "latency_s": 0.42, "ok": True,
            "estimated": False, "error": "",
        })

    with open(out, encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        assert rdr.fieldnames == list(RECORD_FIELDS)
        rows = list(rdr)
    assert len(rows) == 1
    assert rows[0]["trial"] == "1"
    assert rows[0]["output_tokens"] == "50"
    assert rows[0]["ok"] == "True"


def test_writer_unknown_extension_raises(tmp_path: Path):
    bad = tmp_path / "log.txt"
    with pytest.raises(ValueError, match="unsupported extension"):
        with _record_writer(str(bad)):
            pass


# ---------- integration via run_benchmark ----------


def _make_fake_request(latencies, tokens):
    """Yield deterministic RequestResults in order."""
    it_lat = iter(latencies)
    it_tok = iter(tokens)

    def fake(*args, **kwargs):
        return RequestResult(tokens=next(it_tok), latency=next(it_lat), ok=True, estimated=False)

    return fake


def test_run_benchmark_writes_jsonl_with_all_request_records(monkeypatch, tmp_path: Path):
    n_requests = 6  # 2 concurrent × 3 trials
    monkeypatch.setattr(
        "benchmark.runner.benchmark_request",
        _make_fake_request([0.1] * n_requests, [25] * n_requests),
    )

    out = tmp_path / "runs.jsonl"
    run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: "the-prompt",
        prompt_label="fixed",
        max_tokens=25,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=2,
        trials=3,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        output_path=str(out),
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == n_requests
    for row in rows:
        assert set(row.keys()) == set(RECORD_FIELDS)
        assert row["prompt_chars"] == len("the-prompt")
        assert row["output_tokens"] == 25
        assert row["latency_s"] == 0.1
        assert row["ok"] is True
        assert row["error"] == ""

    trials_seen = sorted({r["trial"] for r in rows})
    assert trials_seen == [1, 2, 3]
    indices_per_trial = {
        t: sorted(r["request_index"] for r in rows if r["trial"] == t) for t in trials_seen
    }
    assert indices_per_trial == {1: [0, 1], 2: [0, 1], 3: [0, 1]}


def test_run_benchmark_writes_csv(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "benchmark.runner.benchmark_request",
        _make_fake_request([0.05, 0.06], [10, 12]),
    )

    out = tmp_path / "runs.csv"
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
        trials=2,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        output_path=str(out),
    )

    with open(out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {r["output_tokens"] for r in rows} == {"10", "12"}


def test_run_benchmark_logs_failures_with_error_string(monkeypatch, tmp_path: Path):
    def failing(*args, **kwargs):
        return RequestResult(tokens=0, latency=0.02, ok=False, estimated=False, error="HTTP 500")

    monkeypatch.setattr("benchmark.runner.benchmark_request", failing)

    out = tmp_path / "runs.jsonl"
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
        trials=2,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        output_path=str(out),
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(r["ok"] is False and r["error"] == "HTTP 500" for r in rows)


def test_run_benchmark_excludes_warmup_from_log(monkeypatch, tmp_path: Path):
    counter = {"calls": 0}

    def counted(*args, **kwargs):
        counter["calls"] += 1
        return RequestResult(tokens=5, latency=0.01, ok=True, estimated=False)

    monkeypatch.setattr("benchmark.runner.benchmark_request", counted)

    out = tmp_path / "runs.jsonl"
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
        trials=2,
        warmup=3,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        output_path=str(out),
    )

    rows = out.read_text(encoding="utf-8").splitlines()
    # 3 warmup + 2 trial requests called, but only the 2 trial requests logged.
    assert counter["calls"] == 5
    assert len(rows) == 2


def test_no_output_path_writes_no_file(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "benchmark.runner.benchmark_request",
        _make_fake_request([0.01], [1]),
    )

    before = set(tmp_path.iterdir())
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
        output_path=None,
    )
    after = set(tmp_path.iterdir())
    assert before == after
