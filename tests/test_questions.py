import json
from pathlib import Path

import pytest

from benchmark import (
    DEFAULT_PROMPT,
    DEFAULT_QUESTIONS_FILE,
    _build_prompt_provider,
    _load_questions,
)

# ---------- bundled questions.json ----------


def test_bundled_questions_file_has_100_unique_strings():
    qs = _load_questions(DEFAULT_QUESTIONS_FILE)
    assert len(qs) == 100
    assert len(set(qs)) == 100
    assert all(isinstance(q, str) and q.strip() for q in qs)


# ---------- _load_questions ----------


def test_load_questions_happy_path(tmp_path: Path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
    assert _load_questions(str(p)) == ["a", "b", "c"]


def test_load_questions_filters_non_string_and_blank(tmp_path: Path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps(["ok", "", 42, None, "  ", "good"]), encoding="utf-8")
    assert _load_questions(str(p)) == ["ok", "good"]


def test_load_questions_rejects_non_array(tmp_path: Path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty JSON array"):
        _load_questions(str(p))


def test_load_questions_rejects_empty_array(tmp_path: Path):
    p = tmp_path / "q.json"
    p.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty JSON array"):
        _load_questions(str(p))


def test_load_questions_rejects_array_of_only_invalid_items(tmp_path: Path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps([1, 2, None, ""]), encoding="utf-8")
    with pytest.raises(ValueError, match="no usable string prompts"):
        _load_questions(str(p))


def test_load_questions_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _load_questions(str(tmp_path / "missing.json"))


# ---------- _build_prompt_provider ----------


def test_explicit_prompt_overrides_questions_file(tmp_path: Path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps(["x", "y", "z"]), encoding="utf-8")
    provider, label = _build_prompt_provider(
        explicit_prompt="hardcoded",
        questions_file=str(p),
        seed=0,
    )
    assert [provider() for _ in range(5)] == ["hardcoded"] * 5
    assert "fixed prompt" in label


def test_random_pick_uses_only_questions_from_file(tmp_path: Path):
    p = tmp_path / "q.json"
    questions = ["q1", "q2", "q3", "q4"]
    p.write_text(json.dumps(questions), encoding="utf-8")
    provider, label = _build_prompt_provider(
        explicit_prompt=None, questions_file=str(p), seed=-1,
    )
    picks = {provider() for _ in range(200)}
    assert picks.issubset(set(questions))
    # 200 draws across 4 options → all should appear (probability of miss ~ 0)
    assert picks == set(questions)
    assert "random pick per request" in label
    assert "4 questions" in label


def test_seeded_provider_is_reproducible(tmp_path: Path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps([f"q{i}" for i in range(20)]), encoding="utf-8")

    p1, _ = _build_prompt_provider(None, str(p), seed=123)
    p2, _ = _build_prompt_provider(None, str(p), seed=123)
    seq1 = [p1() for _ in range(50)]
    seq2 = [p2() for _ in range(50)]
    assert seq1 == seq2

    p3, _ = _build_prompt_provider(None, str(p), seed=999)
    seq3 = [p3() for _ in range(50)]
    assert seq3 != seq1


def test_seed_label_only_when_seeded(tmp_path: Path):
    p = tmp_path / "q.json"
    p.write_text(json.dumps(["a", "b"]), encoding="utf-8")
    _, label_seeded = _build_prompt_provider(None, str(p), seed=7)
    _, label_random = _build_prompt_provider(None, str(p), seed=-1)
    assert "seed=7" in label_seeded
    assert "seed=" not in label_random


def test_missing_questions_file_falls_back_to_default(tmp_path: Path, capsys):
    missing = str(tmp_path / "nope.json")
    provider, label = _build_prompt_provider(
        explicit_prompt=None, questions_file=missing, seed=-1,
    )
    assert provider() == DEFAULT_PROMPT
    assert label == "built-in default prompt"
    err = capsys.readouterr().err
    assert "not found" in err
    assert missing in err


def test_disabled_questions_file_uses_default():
    provider, label = _build_prompt_provider(
        explicit_prompt=None, questions_file="", seed=-1,
    )
    assert provider() == DEFAULT_PROMPT
    assert label == "built-in default prompt"


def test_none_questions_file_uses_default():
    provider, label = _build_prompt_provider(
        explicit_prompt=None, questions_file=None, seed=-1,
    )
    assert provider() == DEFAULT_PROMPT
    assert label == "built-in default prompt"


def test_explicit_prompt_used_even_without_questions_file():
    provider, label = _build_prompt_provider(
        explicit_prompt="only-this", questions_file=None, seed=-1,
    )
    assert provider() == "only-this"
    assert "fixed prompt" in label
