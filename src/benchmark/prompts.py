"""Prompt provider: either a fixed string or a random pick from a JSON pool."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Callable, List, Optional

DEFAULT_PROMPT = "Explain the theory of relativity in simple terms."
DEFAULT_QUESTIONS_FILE = str(Path(__file__).parent / "questions.json")


def _load_questions(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected a non-empty JSON array of strings")
    questions = [str(q) for q in data if isinstance(q, str) and q.strip()]
    if not questions:
        raise ValueError(f"{path}: contains no usable string prompts")
    return questions


def _build_prompt_provider(
    explicit_prompt: Optional[str],
    questions_file: Optional[str],
    seed: int,
) -> tuple[Callable[[], str], str]:
    if explicit_prompt:
        prompt = explicit_prompt
        return (lambda: prompt), f"fixed prompt ({len(prompt)} chars)"

    if questions_file:
        try:
            questions = _load_questions(questions_file)
        except FileNotFoundError:
            print(
                f"NOTE: questions file '{questions_file}' not found; "
                f"falling back to default prompt.",
                file=sys.stderr,
            )
        else:
            rng = random.Random(seed) if seed >= 0 else random.Random()
            return (
                lambda: rng.choice(questions),
                f"random pick per request from {questions_file} "
                f"({len(questions)} questions"
                + (f", seed={seed}" if seed >= 0 else "")
                + ")",
            )

    return (lambda: DEFAULT_PROMPT), "built-in default prompt"
