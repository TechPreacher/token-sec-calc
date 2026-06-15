"""Sanity tests for the GitHub Actions workflow.

We don't run the workflow here — these tests just ensure the YAML is valid and
that the structure won't silently rot (missing pytest step, wrong action ref,
etc.). They also guard against introducing the workflow-injection footguns
flagged by GitHub's security guidance.
"""

import re
from pathlib import Path

import pytest
import yaml

WORKFLOW_PATH = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"


# YAML parses "on:" as Python bool True; preserve the string key.
@pytest.fixture(scope="module")
def workflow():
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    return raw, yaml.safe_load(raw)


def test_workflow_file_exists():
    assert WORKFLOW_PATH.is_file(), f"missing CI workflow at {WORKFLOW_PATH}"


def test_workflow_yaml_parses(workflow):
    _, doc = workflow
    assert isinstance(doc, dict)
    assert doc.get("name")


def test_workflow_triggers_on_push_and_pr(workflow):
    _, doc = workflow
    # PyYAML parses bare `on:` to Python True; allow either key.
    triggers = doc.get("on") or doc.get(True)
    assert triggers, "workflow has no 'on' triggers"
    assert "push" in triggers
    assert "pull_request" in triggers


def test_workflow_has_test_job_with_pytest_step(workflow):
    _, doc = workflow
    jobs = doc.get("jobs") or {}
    assert "test" in jobs, f"expected a 'test' job; found {list(jobs)}"
    steps = jobs["test"].get("steps") or []
    assert steps, "test job has no steps"
    pytest_steps = [s for s in steps if "pytest" in (s.get("run") or "")]
    assert pytest_steps, "no step runs pytest"


def test_workflow_runs_ruff_lint(workflow):
    _, doc = workflow
    steps = doc["jobs"]["test"]["steps"]
    ruff_steps = [s for s in steps if "ruff" in (s.get("run") or "")]
    assert ruff_steps, "no step runs ruff"


def test_workflow_uses_uv_and_caches(workflow):
    _, doc = workflow
    steps = doc["jobs"]["test"]["steps"]
    uv_step = next((s for s in steps if "setup-uv" in (s.get("uses") or "")), None)
    assert uv_step is not None, "missing astral-sh/setup-uv action"
    assert uv_step.get("with", {}).get("enable-cache") is True


def test_workflow_matrix_includes_python_311_and_312(workflow):
    _, doc = workflow
    matrix = doc["jobs"]["test"]["strategy"]["matrix"]
    versions = matrix["python-version"]
    assert "3.11" in versions
    assert "3.12" in versions


def test_workflow_pins_action_versions(workflow):
    """Every `uses:` reference must pin a tag/sha — never bare `@main`."""
    _, doc = workflow
    for step in doc["jobs"]["test"]["steps"]:
        uses = step.get("uses")
        if not uses:
            continue
        assert "@" in uses, f"unpinned action: {uses}"
        ref = uses.split("@", 1)[1]
        assert ref not in ("main", "master", "latest"), (
            f"action pinned to floating ref: {uses}"
        )


def test_workflow_does_not_reference_untrusted_event_inputs(workflow):
    """Block known injection-prone github.event.* fields in `run:` blocks."""
    raw, _ = workflow
    injection_sources = (
        "github.event.issue.title",
        "github.event.issue.body",
        "github.event.pull_request.title",
        "github.event.pull_request.body",
        "github.event.comment.body",
        "github.event.review.body",
        "github.event.review_comment.body",
        "github.event.head_commit.message",
        "github.event.head_commit.author.email",
        "github.event.head_commit.author.name",
        "github.head_ref",
    )
    for src in injection_sources:
        assert src not in raw, (
            f"workflow interpolates untrusted input '{src}' — "
            "move to an env: var and quote in shell instead"
        )


def test_workflow_only_uses_safe_interpolations(workflow):
    """Any `${{ ... }}` expression in a run/ref must come from a controlled
    source (matrix.*, env.*, secrets.*, github.repository, etc.). Reject
    anything beginning with `github.event.` outright."""
    raw, _ = workflow
    expressions = re.findall(r"\$\{\{\s*([^}]+?)\s*\}\}", raw)
    bad = [e for e in expressions if e.startswith("github.event.")]
    assert not bad, f"untrusted event interpolation(s) found: {bad}"
