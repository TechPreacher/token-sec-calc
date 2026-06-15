import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from benchmark import (
    RECORD_FIELDS,
    _consume_sse_stream,
    benchmark_request,
    run_benchmark,
)

# ---------- fake streaming response ----------


class FakeStreamingResponse:
    def __init__(self, lines, ok: bool = True, status: int = 200,
                 reason: str = "OK", text: str = ""):
        self.ok = ok
        self.status_code = status
        self.reason = reason
        self.text = text
        self._lines = lines
        self.closed = False

    def iter_lines(self):
        for line in self._lines:
            if isinstance(line, str):
                yield line.encode("utf-8")
            else:
                yield line

    def json(self):
        raise AssertionError("json() should not be called on a streaming response")

    def close(self):
        self.closed = True


def _chat_chunk(content: str) -> str:
    return "data: " + json.dumps(
        {"choices": [{"delta": {"content": content}, "index": 0}]}
    )


def _completions_chunk(text: str) -> str:
    return "data: " + json.dumps({"choices": [{"text": text, "index": 0}]})


def _usage_chunk(n: int) -> str:
    return "data: " + json.dumps({"choices": [], "usage": {"completion_tokens": n}})


# ---------- _consume_sse_stream ----------


def test_consume_chat_stream_captures_ttft_and_tokens():
    lines = [
        _chat_chunk("Hello"),
        "",
        _chat_chunk(" world"),
        _usage_chunk(2),
        "data: [DONE]",
        _chat_chunk("AFTER-DONE"),  # must be ignored after [DONE]
    ]
    resp = FakeStreamingResponse(lines)
    import time as _time
    start = _time.perf_counter()
    tokens, ttft, estimated = _consume_sse_stream(resp, chat=True, start=start)
    assert tokens == 2
    assert estimated is False
    assert ttft is not None and ttft > 0


def test_consume_completions_stream_uses_text_field():
    lines = [
        _completions_chunk("foo"),
        _completions_chunk("bar"),
        _usage_chunk(7),
        "data: [DONE]",
    ]
    resp = FakeStreamingResponse(lines)
    import time as _time
    tokens, ttft, estimated = _consume_sse_stream(resp, chat=False, start=_time.perf_counter())
    assert tokens == 7
    assert estimated is False
    assert ttft is not None


def test_consume_stream_estimates_when_usage_missing():
    # Total text "abcdefgh" (8 chars) → estimate 8 // 4 = 2 tokens
    lines = [_completions_chunk("abcd"), _completions_chunk("efgh"), "data: [DONE]"]
    resp = FakeStreamingResponse(lines)
    import time as _time
    tokens, ttft, estimated = _consume_sse_stream(resp, chat=False, start=_time.perf_counter())
    assert tokens == 2
    assert estimated is True
    assert ttft is not None


def test_consume_stream_returns_none_ttft_when_no_content():
    # Only usage and DONE, no content chunks → ttft stays None.
    lines = [_usage_chunk(0), "data: [DONE]"]
    resp = FakeStreamingResponse(lines)
    import time as _time
    tokens, ttft, estimated = _consume_sse_stream(resp, chat=False, start=_time.perf_counter())
    assert tokens == 0
    assert ttft is None
    assert estimated is False


def test_consume_stream_skips_malformed_json_lines():
    lines = [
        "data: not-json-at-all",
        _completions_chunk("ok"),
        _usage_chunk(1),
        "data: [DONE]",
    ]
    resp = FakeStreamingResponse(lines)
    import time as _time
    tokens, ttft, _ = _consume_sse_stream(resp, chat=False, start=_time.perf_counter())
    assert tokens == 1
    assert ttft is not None


def test_consume_stream_ignores_non_data_lines():
    lines = [
        ": ping",
        "event: heartbeat",
        _completions_chunk("x"),
        _usage_chunk(1),
        "data: [DONE]",
    ]
    resp = FakeStreamingResponse(lines)
    import time as _time
    tokens, _, _ = _consume_sse_stream(resp, chat=False, start=_time.perf_counter())
    assert tokens == 1


# ---------- benchmark_request streaming integration ----------


def test_streaming_request_sets_stream_payload_and_passes_stream_arg():
    lines = [_completions_chunk("hi"), _usage_chunk(3), "data: [DONE]"]
    with patch(
        "benchmark.client.requests.post", return_value=FakeStreamingResponse(lines),
    ) as post:
        r = benchmark_request(
            "https://x/v1/completions", "k", "m", "p", max_tokens=10,
            mode="completions", stream=True,
        )

    assert r.ok
    assert r.tokens == 3
    assert r.ttft is not None and r.ttft > 0
    assert r.latency >= r.ttft

    sent = post.call_args.kwargs["json"]
    assert sent["stream"] is True
    assert sent["stream_options"] == {"include_usage": True}
    # urllib-level streaming flag enabled on requests.post
    assert post.call_args.kwargs["stream"] is True


def test_non_streaming_request_does_not_include_stream_options():
    """Back-compat: non-streaming payload must not carry the streaming-only knob."""
    ok = MagicMock()
    ok.ok = True
    ok.json.return_value = {
        "choices": [{"text": "x"}], "usage": {"completion_tokens": 1},
    }
    with patch("benchmark.client.requests.post", return_value=ok) as post:
        r = benchmark_request(
            "https://x/v1/completions", "k", "m", "p", max_tokens=10,
            mode="completions", stream=False,
        )
    assert r.ok
    assert r.ttft is None  # only set on streaming path
    sent = post.call_args.kwargs["json"]
    assert sent["stream"] is False
    assert "stream_options" not in sent
    assert post.call_args.kwargs.get("stream") is False


def test_streaming_http_error_marks_failure_and_does_not_parse_stream():
    err = MagicMock()
    err.ok = False
    err.status_code = 503
    err.reason = "Service Unavailable"
    err.text = "overloaded"
    err.iter_lines.side_effect = AssertionError("must not iter on failed response")
    with patch("benchmark.client.requests.post", return_value=err):
        r = benchmark_request(
            "https://x/v1/completions", "k", "m", "p", max_tokens=10,
            mode="completions", stream=True,
        )
    assert r.ok is False
    assert r.tokens == 0
    assert r.ttft is None
    assert r.error is not None and "503" in r.error


def test_streaming_response_is_closed_after_consumption():
    lines = [_completions_chunk("x"), _usage_chunk(1), "data: [DONE]"]
    resp = FakeStreamingResponse(lines)
    with patch("benchmark.client.requests.post", return_value=resp):
        benchmark_request(
            "https://x/v1/completions", "k", "m", "p", max_tokens=10,
            mode="completions", stream=True,
        )
    assert resp.closed is True


# ---------- run_benchmark + percentile + record schema ----------


def test_record_fields_include_ttft_s():
    assert "ttft_s" in RECORD_FIELDS


def test_run_benchmark_records_ttft_in_jsonl_when_streaming(monkeypatch, tmp_path: Path):
    def fake_streaming(*args, **kwargs):
        from benchmark import RequestResult
        return RequestResult(tokens=10, latency=0.5, ok=True, estimated=False, ttft=0.123)

    monkeypatch.setattr("benchmark.runner.benchmark_request", fake_streaming)

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
        trials=3,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        output_path=str(out),
        stream=True,
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    for row in rows:
        assert row["ttft_s"] == 0.123


def test_run_benchmark_records_none_ttft_when_not_streaming(monkeypatch, tmp_path: Path):
    def fake_nonstream(*args, **kwargs):
        from benchmark import RequestResult
        return RequestResult(tokens=10, latency=0.5, ok=True, estimated=False, ttft=None)

    monkeypatch.setattr("benchmark.runner.benchmark_request", fake_nonstream)

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
        stream=False,
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        assert row["ttft_s"] is None


def test_run_benchmark_prints_ttft_percentiles_when_streaming(monkeypatch, capsys):
    ttfts = iter([0.05, 0.10, 0.15, 0.20])

    def fake(*args, **kwargs):
        from benchmark import RequestResult
        return RequestResult(tokens=10, latency=0.5, ok=True, estimated=False, ttft=next(ttfts))

    monkeypatch.setattr("benchmark.runner.benchmark_request", fake)

    run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: "p",
        prompt_label="fixed",
        max_tokens=10,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=2,
        trials=2,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        stream=True,
    )
    out = capsys.readouterr().out
    assert "Time-to-first-token" in out
    assert "p50=" in out
    assert "p99=" in out
    assert "mean=" in out


def test_run_benchmark_omits_ttft_section_when_not_streaming(monkeypatch, capsys):
    def fake(*args, **kwargs):
        from benchmark import RequestResult
        return RequestResult(tokens=10, latency=0.5, ok=True, estimated=False, ttft=None)

    monkeypatch.setattr("benchmark.runner.benchmark_request", fake)

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
        stream=False,
    )
    out = capsys.readouterr().out
    assert "Time-to-first-token" not in out
