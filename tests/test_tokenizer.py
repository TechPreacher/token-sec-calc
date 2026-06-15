import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmark import (
    RECORD_FIELDS,
    CharsTokenizer,
    RequestResult,
    TikTokenTokenizer,
    Tokenizer,
    _build_tokenizer,
    _consume_sse_stream,
    _estimate_tokens,
    benchmark_request,
    run_benchmark,
)

# ---------- chars/4 tokenizer ----------


def test_chars_tokenizer_counts_floor_div_4():
    t = CharsTokenizer()
    assert t.count("") == 0
    assert t.count("abc") == 0  # 3 // 4
    assert t.count("abcd") == 1
    assert t.count("a" * 100) == 25
    assert t.label == "chars/4"


# ---------- _estimate_tokens fallback clamp ----------


def test_estimate_tokens_returns_zero_for_empty_text():
    assert _estimate_tokens("", CharsTokenizer()) == 0


def test_estimate_tokens_clamps_nonempty_to_at_least_one():
    # CharsTokenizer would return 0 for 3 chars; clamp keeps it real.
    assert _estimate_tokens("abc", CharsTokenizer()) == 1


def test_estimate_tokens_passes_through_when_above_one():
    assert _estimate_tokens("a" * 40, CharsTokenizer()) == 10


def test_estimate_tokens_uses_provided_tokenizer():
    class Two(Tokenizer):
        label = "always-2"

        def count(self, text: str) -> int:
            return 2 if text else 0

    assert _estimate_tokens("anything", Two()) == 2
    assert _estimate_tokens("", Two()) == 0


# ---------- _build_tokenizer ----------


def test_build_tokenizer_chars4():
    t = _build_tokenizer("chars4")
    assert isinstance(t, CharsTokenizer)
    assert t.label == "chars/4"


def test_build_tokenizer_rejects_unknown_spec():
    with pytest.raises(ValueError, match="unknown tokenizer spec"):
        _build_tokenizer("bogus")


def test_build_tokenizer_auto_falls_back_when_tiktoken_missing(capsys):
    def fake_init(self, name):
        raise ImportError("tiktoken not installed")

    with patch.object(TikTokenTokenizer, "__init__", fake_init):
        t = _build_tokenizer("auto")
    assert isinstance(t, CharsTokenizer)
    err = capsys.readouterr().err
    assert "tiktoken not installed" in err
    assert "chars/4" in err


def test_build_tokenizer_auto_uses_tiktoken_when_available():
    class Fake(TikTokenTokenizer):
        def __init__(self, name):  # bypass real tiktoken import
            self.label = f"tiktoken:{name}"

            class _E:
                def encode(self_inner, text):
                    return list(text)

            self._enc = _E()

    with patch("benchmark.tokenizer.TikTokenTokenizer", Fake):
        t = _build_tokenizer("auto")
    assert t.label == "tiktoken:cl100k_base"
    assert t.count("abcdef") == 6


def test_build_tokenizer_explicit_tiktoken_propagates_encoding_name():
    captured = {}

    class Fake(TikTokenTokenizer):
        def __init__(self, name):
            captured["name"] = name
            self.label = f"tiktoken:{name}"

            class _E:
                def encode(self_inner, text):
                    return [0] * len(text.split())

            self._enc = _E()

    with patch("benchmark.tokenizer.TikTokenTokenizer", Fake):
        t = _build_tokenizer("tiktoken:o200k_base")
    assert captured["name"] == "o200k_base"
    assert t.label == "tiktoken:o200k_base"


# ---------- estimation path uses the configured tokenizer ----------


class WordTokenizer(Tokenizer):
    """Trivial deterministic tokenizer: 1 token per whitespace-separated word."""

    label = "words"

    def count(self, text: str) -> int:
        return len(text.split())


def test_non_streaming_estimation_uses_supplied_tokenizer():
    body = {"choices": [{"text": "alpha beta gamma delta"}]}  # no usage → estimated
    ok = MagicMock()
    ok.ok = True
    ok.json.return_value = body

    with patch("benchmark.client.requests.post", return_value=ok):
        r = benchmark_request(
            "https://x/v1/completions", "k", "m", "p", max_tokens=10,
            mode="completions", tokenizer=WordTokenizer(),
        )
    assert r.ok
    assert r.estimated is True
    assert r.tokens == 4  # 4 words via WordTokenizer (chars/4 would give 5)


def test_streaming_estimation_uses_supplied_tokenizer():
    lines = [
        "data: " + json.dumps({"choices": [{"text": "alpha beta"}]}),
        "data: " + json.dumps({"choices": [{"text": " gamma"}]}),
        # No usage chunk → forced to estimate.
        "data: [DONE]",
    ]

    class FakeResp:
        ok = True
        status_code = 200
        reason = "OK"
        text = ""

        def iter_lines(self):
            for line in lines:
                yield line.encode("utf-8")

        def close(self):
            pass

    with patch("benchmark.client.requests.post", return_value=FakeResp()):
        r = benchmark_request(
            "https://x/v1/completions", "k", "m", "p", max_tokens=10,
            mode="completions", stream=True, tokenizer=WordTokenizer(),
        )
    assert r.ok
    assert r.estimated is True
    assert r.tokens == 3  # "alpha beta gamma" → 3 words


def test_server_usage_still_overrides_tokenizer_estimate():
    body = {
        "choices": [{"text": "alpha beta gamma delta"}],
        "usage": {"completion_tokens": 99},
    }
    ok = MagicMock()
    ok.ok = True
    ok.json.return_value = body

    with patch("benchmark.client.requests.post", return_value=ok):
        r = benchmark_request(
            "https://x/v1/completions", "k", "m", "p", max_tokens=10,
            mode="completions", tokenizer=WordTokenizer(),
        )
    assert r.ok
    assert r.estimated is False
    assert r.tokens == 99  # server wins over local tokenizer


def test_consume_sse_stream_default_tokenizer_is_chars4():
    # Back-compat: existing tests call _consume_sse_stream() without tokenizer.
    lines = [
        "data: " + json.dumps({"choices": [{"text": "abcdefgh"}]}),
        "data: [DONE]",
    ]

    class Resp:
        def iter_lines(self):
            for line in lines:
                yield line.encode("utf-8")

    import time as _time
    tokens, _ttft, estimated = _consume_sse_stream(Resp(), chat=False, start=_time.perf_counter())
    assert tokens == 2  # chars/4 = 8//4
    assert estimated is True


# ---------- record schema + prompt_tokens logging ----------


def test_record_fields_include_prompt_tokens():
    assert "prompt_tokens" in RECORD_FIELDS
    # ordering: prompt_chars must precede prompt_tokens
    fields = list(RECORD_FIELDS)
    assert fields.index("prompt_chars") < fields.index("prompt_tokens")


def test_run_benchmark_logs_prompt_tokens_using_chosen_tokenizer(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        "benchmark.runner.benchmark_request",
        lambda *a, **kw: RequestResult(tokens=10, latency=0.1, ok=True, estimated=False),
    )

    out = tmp_path / "runs.jsonl"
    run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: "alpha beta gamma",  # 3 words / 16 chars
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
        tokenizer=WordTokenizer(),
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    for row in rows:
        assert row["prompt_chars"] == len("alpha beta gamma")
        assert row["prompt_tokens"] == 3  # WordTokenizer count


def test_run_benchmark_prints_tokenizer_label(monkeypatch, capsys):
    monkeypatch.setattr(
        "benchmark.runner.benchmark_request",
        lambda *a, **kw: RequestResult(tokens=1, latency=0.01, ok=True, estimated=False),
    )

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
        tokenizer=WordTokenizer(),
    )
    out = capsys.readouterr().out
    assert "Tokenizer: words" in out
