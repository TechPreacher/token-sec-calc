"""Single HTTP request to an OpenAI-compatible inference endpoint.

Handles both non-streaming and SSE streaming responses, including TTFT capture.
The non-streaming path is the simple case; the SSE consumer is factored so it
can be tested in isolation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Optional

import requests

from .tokenizer import CharsTokenizer, Tokenizer, _estimate_tokens


@dataclass
class RequestResult:
    tokens: int
    latency: float
    ok: bool
    estimated: bool
    error: Optional[str] = None
    ttft: Optional[float] = None  # time-to-first-token (s), set only on streaming success


def _is_chat_endpoint(endpoint: str, mode: str) -> bool:
    if mode == "chat":
        return True
    if mode == "completions":
        return False
    return "/chat/completions" in endpoint


def _consume_sse_stream(
    response, chat: bool, start: float, tokenizer: Optional[Tokenizer] = None,
) -> tuple[int, Optional[float], bool]:
    """Read SSE chunks from a streaming response.

    Returns `(output_tokens, ttft_seconds, estimated)`. `ttft` is None if no
    content chunk arrived (e.g. server returned only usage). `estimated` is True
    when the server omitted `usage.completion_tokens` and the count was derived
    from accumulated text length.
    """
    if tokenizer is None:
        tokenizer = CharsTokenizer()
    ttft: Optional[float] = None
    text_parts: list[str] = []
    completion_tokens = 0

    for raw in response.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        piece = ""
        choices = chunk.get("choices") or []
        if choices:
            first = choices[0]
            if chat:
                delta = first.get("delta") or {}
                piece = delta.get("content") or ""
            else:
                piece = first.get("text") or ""
        if piece:
            if ttft is None:
                ttft = time.perf_counter() - start
            text_parts.append(piece)

        usage = chunk.get("usage")
        if usage and usage.get("completion_tokens") is not None:
            completion_tokens = int(usage["completion_tokens"])

    estimated = False
    if completion_tokens == 0:
        text = "".join(text_parts)
        if text:
            completion_tokens = _estimate_tokens(text, tokenizer)
            estimated = True

    return completion_tokens, ttft, estimated


def benchmark_request(
    endpoint: str,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    top_p: float = 1.0,
    timeout: float = 120.0,
    mode: str = "auto",
    ignore_eos: bool = True,
    stream: bool = False,
    tokenizer: Optional[Tokenizer] = None,
) -> RequestResult:
    if tokenizer is None:
        tokenizer = CharsTokenizer()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    chat = _is_chat_endpoint(endpoint, mode)
    if chat:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }
    else:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": stream,
        }

    if stream:
        payload["stream_options"] = {"include_usage": True}

    if ignore_eos:
        payload["ignore_eos"] = True
        payload["min_tokens"] = max_tokens

    start = time.perf_counter()
    try:
        response = requests.post(
            endpoint, headers=headers, json=payload, timeout=timeout, stream=stream,
        )
        if not response.ok:
            body = (response.text or "")[:500].replace("\n", " ")
            latency = time.perf_counter() - start
            return RequestResult(
                0, latency, ok=False, estimated=False,
                error=f"HTTP {response.status_code} {response.reason}: {body}",
            )

        if stream:
            try:
                completion_tokens, ttft, estimated = _consume_sse_stream(
                    response, chat, start, tokenizer,
                )
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
            latency = time.perf_counter() - start
            return RequestResult(
                completion_tokens, latency, ok=True, estimated=estimated, ttft=ttft,
            )

        latency = time.perf_counter() - start

        data = response.json()
        usage = data.get("usage") or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        estimated = False

        if completion_tokens == 0:
            choices = data.get("choices") or []
            if choices:
                first = choices[0]
                if chat:
                    msg = first.get("message") or {}
                    text = msg.get("content") or ""
                else:
                    text = first.get("text") or ""
                completion_tokens = _estimate_tokens(text, tokenizer)
                estimated = completion_tokens > 0

        return RequestResult(completion_tokens, latency, ok=True, estimated=estimated)

    except Exception as e:
        latency = time.perf_counter() - start
        return RequestResult(0, latency, ok=False, estimated=False, error=str(e))
