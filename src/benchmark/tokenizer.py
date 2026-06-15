"""Tokenizer abstraction + helpers for output-token estimation and prompt
normalization. Used both during request handling (for the `usage`-missing
fallback) and at the trial level (for prompt-token logging and `--prompt_tokens`
padding/truncation)."""

from __future__ import annotations

import sys


class Tokenizer:
    """Tokenizer protocol. Implementations expose `count(text) -> int` and
    a human-readable `label` describing which encoding is in use.

    Returning 0 for empty input is allowed — callers that need a clamp-to-1
    for nonempty estimation must do so themselves (see `_estimate_tokens`).
    """
    label: str = "tokenizer"

    def count(self, text: str) -> int:  # pragma: no cover - interface only
        raise NotImplementedError


class CharsTokenizer(Tokenizer):
    label = "chars/4"

    def count(self, text: str) -> int:
        return len(text) // 4


class TikTokenTokenizer(Tokenizer):
    def __init__(self, name: str) -> None:
        try:
            import tiktoken  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "tiktoken not installed. Install with: pip install tiktoken"
            ) from e
        self._enc = tiktoken.get_encoding(name)
        self.label = f"tiktoken:{name}"

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))


def _build_tokenizer(spec: str) -> Tokenizer:
    """Build a tokenizer from a spec string.

    - `chars4` — fast ASCII-biased approximation (`len(text) // 4`).
    - `auto`   — try tiktoken `cl100k_base`; if tiktoken not installed,
                 emit a one-time stderr note and fall back to `chars4`.
    - `tiktoken:<encoding>` — explicit encoding, e.g. `tiktoken:cl100k_base`,
                              `tiktoken:o200k_base`.
    """
    if spec == "chars4":
        return CharsTokenizer()
    if spec == "auto":
        try:
            return TikTokenTokenizer("cl100k_base")
        except ImportError:
            print(
                "NOTE: tiktoken not installed (`pip install tiktoken`); "
                "falling back to chars/4 estimate for token counts.",
                file=sys.stderr,
            )
            return CharsTokenizer()
    if spec.startswith("tiktoken:"):
        return TikTokenTokenizer(spec.split(":", 1)[1])
    raise ValueError(f"unknown tokenizer spec: {spec!r}")


_NORMALIZE_FILLER = (
    " the quick brown fox jumps over the lazy dog "
    "the rain in spain falls mainly on the plain"
)


def _normalize_prompt(text: str, target_tokens: int, tokenizer: Tokenizer) -> str:
    """Pad or truncate `text` so `tokenizer.count(result)` is as close to
    `target_tokens` as possible without going under.

    Uses binary search on character length, so it works for any tokenizer that
    exposes only `count`. With non-character tokenizers the result lands at
    `target_tokens` or `target_tokens + 1` (token boundaries don't align with
    char boundaries). Empty `text` is treated as the empty string.
    """
    if target_tokens <= 0:
        return text

    if tokenizer.count(text) < target_tokens:
        padded = text
        while tokenizer.count(padded) < target_tokens:
            padded += _NORMALIZE_FILLER
        text = padded

    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if tokenizer.count(text[:mid]) >= target_tokens:
            hi = mid
        else:
            lo = mid + 1
    return text[:lo]


def _estimate_tokens(text: str, tokenizer: Tokenizer) -> int:
    """Output-token fallback when the server omits `usage.completion_tokens`.

    Returns 0 for empty text. Clamps non-empty text to at least 1 so a real
    response never appears as zero work (matches existing behavior).
    """
    if not text:
        return 0
    return max(1, tokenizer.count(text))
