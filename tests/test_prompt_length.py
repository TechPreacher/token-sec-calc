import json
from pathlib import Path

from benchmark import (
    CharsTokenizer,
    RequestResult,
    Tokenizer,
    _normalize_prompt,
    run_benchmark,
)


class WordTokenizer(Tokenizer):
    """1 token per whitespace-separated word."""

    label = "words"

    def count(self, text: str) -> int:
        return len(text.split())


# ---------- _normalize_prompt ----------


def test_normalize_zero_or_negative_target_returns_text_unchanged():
    t = CharsTokenizer()
    assert _normalize_prompt("hello world", 0, t) == "hello world"
    assert _normalize_prompt("hello world", -5, t) == "hello world"


def test_normalize_pads_short_text_to_at_least_target():
    t = CharsTokenizer()  # count = len(text) // 4
    short = "hi"  # 0 tokens
    out = _normalize_prompt(short, target_tokens=10, tokenizer=t)
    assert t.count(out) >= 10
    # Padded result starts with the original text.
    assert out.startswith(short)


def test_normalize_truncates_long_text_to_target_or_close():
    t = CharsTokenizer()
    long = "a" * 1000  # count = 250
    out = _normalize_prompt(long, target_tokens=25, tokenizer=t)
    assert t.count(out) >= 25
    # Binary search lands at the first char-length that reaches the target,
    # so the resulting count is target_tokens (or target_tokens + 1 for
    # tokenizers whose boundaries don't align with chars).
    assert t.count(out) <= 26
    assert len(out) <= len(long)


def test_normalize_with_word_tokenizer_hits_target_for_pad_and_truncate():
    t = WordTokenizer()
    # Pad
    short = "alpha beta"  # 2 words
    out = _normalize_prompt(short, target_tokens=10, tokenizer=t)
    assert t.count(out) == 10
    # Truncate
    long = " ".join(["word"] * 50)  # 50 words
    out = _normalize_prompt(long, target_tokens=12, tokenizer=t)
    assert t.count(out) == 12


def test_normalize_empty_text_pads_from_filler():
    t = CharsTokenizer()
    out = _normalize_prompt("", target_tokens=5, tokenizer=t)
    assert t.count(out) >= 5
    assert len(out) > 0


# ---------- integration: provider wrap pins prompt tokens per request ----------


def _ok_result(*args, **kwargs):
    return RequestResult(tokens=5, latency=0.01, ok=True, estimated=False)


def test_normalized_provider_pins_prompt_tokens_per_request(monkeypatch, tmp_path: Path):
    """End-to-end: when a provider is wrapped with _normalize_prompt before being
    passed to run_benchmark, every per-request record shows the same
    prompt_tokens value (matching the target)."""
    monkeypatch.setattr("benchmark.runner.benchmark_request", _ok_result)
    t = WordTokenizer()

    raw_prompts = iter([
        "alpha",
        "alpha beta gamma",
        "one two three four five six seven eight nine ten eleven twelve",
    ])

    target = 5

    def wrapped():
        return _normalize_prompt(next(raw_prompts), target, t)

    out = tmp_path / "runs.jsonl"
    run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=wrapped,
        prompt_label="fixed",
        max_tokens=5,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=1,
        trials=3,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        output_path=str(out),
        tokenizer=t,
    )

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    for row in rows:
        assert row["prompt_tokens"] == target


# ---------- summary prints input-token stats ----------


def test_summary_shows_prompt_input_token_stats(monkeypatch, capsys):
    monkeypatch.setattr("benchmark.runner.benchmark_request", _ok_result)
    t = WordTokenizer()
    prompts = iter(["a", "a b", "a b c", "a b c d"])

    run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: next(prompts),
        prompt_label="varied",
        max_tokens=5,
        temperature=0.0,
        top_p=1.0,
        concurrent_requests=1,
        trials=4,
        warmup=0,
        timeout=30.0,
        mode="completions",
        ignore_eos=False,
        tokenizer=t,
    )
    out = capsys.readouterr().out
    assert "Prompt input tokens" in out
    # min=1 (single word), max=4 (four words)
    assert "min=    1" in out
    assert "max=    4" in out


def test_summary_input_token_stats_in_open_loop_mode_too(monkeypatch, capsys):
    monkeypatch.setattr("benchmark.runner.benchmark_request", _ok_result)
    t = WordTokenizer()

    run_benchmark(
        endpoint="http://x/v1/completions",
        api_key="k",
        model="m",
        prompt_provider=lambda: "alpha beta gamma",  # always 3 tokens
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
        tokenizer=t,
    )
    out = capsys.readouterr().out
    assert "Prompt input tokens" in out
    assert "min=    3" in out
    assert "max=    3" in out
