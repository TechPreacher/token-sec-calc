from unittest.mock import MagicMock, patch

import pytest

from benchmark import _is_chat_endpoint, benchmark_request

# ---------- _is_chat_endpoint ----------


@pytest.mark.parametrize(
    "endpoint,mode,expected",
    [
        ("https://x/v1/chat/completions", "auto", True),
        ("https://x/v1/completions", "auto", False),
        ("https://x/v1/completions", "chat", True),
        ("https://x/v1/chat/completions", "completions", False),
        ("https://x/v1/anything", "chat", True),
        ("https://x/v1/anything", "completions", False),
    ],
)
def test_is_chat_endpoint(endpoint, mode, expected):
    assert _is_chat_endpoint(endpoint, mode) is expected


# ---------- benchmark_request payload shape ----------


def _ok_response(json_body):
    resp = MagicMock()
    resp.ok = True
    resp.json.return_value = json_body
    return resp


def test_chat_payload_uses_messages_and_returns_usage_tokens():
    body = {
        "choices": [{"message": {"content": "hi there"}}],
        "usage": {"completion_tokens": 42},
    }
    with patch("benchmark.client.requests.post", return_value=_ok_response(body)) as post:
        r = benchmark_request(
            "https://x/v1/chat/completions", "k", "m", "prompt-text", 50, mode="chat",
        )
    assert r.ok
    assert r.tokens == 42
    assert r.estimated is False
    sent = post.call_args.kwargs["json"]
    assert sent["messages"] == [{"role": "user", "content": "prompt-text"}]
    assert "prompt" not in sent
    assert sent["model"] == "m"
    assert sent["max_tokens"] == 50
    assert sent["stream"] is False


def test_completions_payload_uses_prompt_field():
    body = {"choices": [{"text": "hello"}], "usage": {"completion_tokens": 7}}
    with patch("benchmark.client.requests.post", return_value=_ok_response(body)) as post:
        r = benchmark_request(
            "https://x/v1/completions", "k", "m", "the-prompt", 25, mode="completions",
        )
    assert r.ok
    assert r.tokens == 7
    sent = post.call_args.kwargs["json"]
    assert sent["prompt"] == "the-prompt"
    assert "messages" not in sent


def test_auth_header_set():
    body = {"choices": [{"text": "x"}], "usage": {"completion_tokens": 1}}
    with patch("benchmark.client.requests.post", return_value=_ok_response(body)) as post:
        benchmark_request("https://x/v1/completions", "secret-key", "m", "p", 10)
    headers = post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer secret-key"
    assert headers["Content-Type"] == "application/json"


# ---------- token extraction ----------


def test_falls_back_to_text_length_estimation_when_usage_missing_chat():
    text = "a" * 40  # 40/4 = 10 tokens estimate
    body = {"choices": [{"message": {"content": text}}]}
    with patch("benchmark.client.requests.post", return_value=_ok_response(body)):
        r = benchmark_request("https://x/v1/chat/completions", "k", "m", "p", 10, mode="chat")
    assert r.ok
    assert r.estimated is True
    assert r.tokens == 10


def test_falls_back_to_text_length_estimation_when_usage_missing_completions():
    text = "abcdefgh"  # 8/4 = 2 tokens estimate
    body = {"choices": [{"text": text}]}
    with patch("benchmark.client.requests.post", return_value=_ok_response(body)):
        r = benchmark_request("https://x/v1/completions", "k", "m", "p", 10, mode="completions")
    assert r.ok
    assert r.estimated is True
    assert r.tokens == 2


def test_estimation_never_returns_zero_tokens_for_nonempty_text():
    body = {"choices": [{"text": "ab"}]}  # 2/4 = 0, clamped to 1
    with patch("benchmark.client.requests.post", return_value=_ok_response(body)):
        r = benchmark_request("https://x/v1/completions", "k", "m", "p", 10, mode="completions")
    assert r.tokens == 1
    assert r.estimated is True


# ---------- error paths ----------


def test_http_error_marks_failure_and_captures_body():
    err_resp = MagicMock()
    err_resp.ok = False
    err_resp.status_code = 500
    err_resp.reason = "Internal Server Error"
    err_resp.text = "boom\nmore detail"
    with patch("benchmark.client.requests.post", return_value=err_resp):
        r = benchmark_request("https://x/v1/completions", "k", "m", "p", 10)
    assert r.ok is False
    assert r.tokens == 0
    assert r.error is not None
    assert "HTTP 500" in r.error
    assert "boom" in r.error
    # newlines stripped per implementation
    assert "\n" not in r.error


def test_network_exception_caught_and_reported():
    with patch("benchmark.client.requests.post", side_effect=ConnectionError("no route")):
        r = benchmark_request("https://x/v1/completions", "k", "m", "p", 10)
    assert r.ok is False
    assert r.tokens == 0
    assert r.error == "no route"
    assert r.latency >= 0


def test_latency_is_measured_for_successful_request():
    body = {"choices": [{"text": "x"}], "usage": {"completion_tokens": 1}}
    with patch("benchmark.client.requests.post", return_value=_ok_response(body)):
        r = benchmark_request("https://x/v1/completions", "k", "m", "p", 10)
    assert r.ok
    assert r.latency > 0


# ---------- greedy default + ignore_eos behavior ----------


def test_default_sampling_is_greedy():
    """Defaults must be greedy (temp=0, top_p=1) so output is reproducible."""
    import inspect

    from benchmark import benchmark_request as br

    sig = inspect.signature(br)
    assert sig.parameters["temperature"].default == 0.0
    assert sig.parameters["top_p"].default == 1.0
    assert sig.parameters["ignore_eos"].default is True


def test_ignore_eos_default_sends_min_tokens_and_flag_chat():
    body = {"choices": [{"message": {"content": "x"}}], "usage": {"completion_tokens": 1}}
    with patch("benchmark.client.requests.post", return_value=_ok_response(body)) as post:
        benchmark_request(
            "https://x/v1/chat/completions", "k", "m", "p", max_tokens=64, mode="chat",
        )
    sent = post.call_args.kwargs["json"]
    assert sent["ignore_eos"] is True
    assert sent["min_tokens"] == 64
    assert sent["max_tokens"] == 64


def test_ignore_eos_default_sends_min_tokens_and_flag_completions():
    body = {"choices": [{"text": "x"}], "usage": {"completion_tokens": 1}}
    with patch("benchmark.client.requests.post", return_value=_ok_response(body)) as post:
        benchmark_request(
            "https://x/v1/completions", "k", "m", "p", max_tokens=32, mode="completions",
        )
    sent = post.call_args.kwargs["json"]
    assert sent["ignore_eos"] is True
    assert sent["min_tokens"] == 32


def test_ignore_eos_false_omits_extension_fields():
    body = {"choices": [{"text": "x"}], "usage": {"completion_tokens": 1}}
    with patch("benchmark.client.requests.post", return_value=_ok_response(body)) as post:
        benchmark_request(
            "https://x/v1/completions", "k", "m", "p", max_tokens=32,
            mode="completions", ignore_eos=False,
        )
    sent = post.call_args.kwargs["json"]
    assert "ignore_eos" not in sent
    assert "min_tokens" not in sent
    # max_tokens still present
    assert sent["max_tokens"] == 32


def test_explicit_temperature_overrides_default():
    body = {"choices": [{"text": "x"}], "usage": {"completion_tokens": 1}}
    with patch("benchmark.client.requests.post", return_value=_ok_response(body)) as post:
        benchmark_request(
            "https://x/v1/completions", "k", "m", "p", 10,
            temperature=0.9, top_p=0.5,
        )
    sent = post.call_args.kwargs["json"]
    assert sent["temperature"] == 0.9
    assert sent["top_p"] == 0.5
