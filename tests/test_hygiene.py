import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "src" / "benchmark"

# Matches OpenAI-style secret token literals (sk- followed by >=20 base62/hex chars).
SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9]{20,}")

# Files where a literal secret would be a leak. Excludes .env (gitignored) and
# bytecode/cache directories.
TRACKED_GLOBS = ("*.py", "*.sh", "*.toml", "*.md", "*.json", "*.yml", "*.yaml")
IGNORED_DIRS = {".venv", ".pytest_cache", "__pycache__", ".git", "node_modules"}


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # `from __future__ import ...` is imported for side effects only;
            # the names aren't meant to be referenced.
            if node.module == "__future__":
                continue
            for a in node.names:
                if a.name == "*":
                    continue
                names.add(a.asname or a.name)
    return names


def _referenced_names(tree: ast.AST) -> set[str]:
    refs: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            refs.add(node.id)
        elif isinstance(node, ast.Attribute):
            cur: ast.AST = node
            while isinstance(cur, ast.Attribute):
                cur = cur.value
            if isinstance(cur, ast.Name):
                refs.add(cur.id)
    return refs


def test_no_unused_top_level_imports_in_benchmark():
    """Sweep every .py file in the benchmark package for dead imports."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(PKG.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # __init__.py re-exports symbols — its "imports" are intentionally
        # unreferenced inside the file itself.
        if path.name == "__init__.py":
            continue
        imported = _imported_names(tree)
        referenced = _referenced_names(tree)
        unused = sorted(imported - referenced)
        if unused:
            offenders[str(path.relative_to(REPO))] = unused
    assert not offenders, f"unused imports in package modules: {offenders}"


def _iter_tracked_files():
    for path in REPO.rglob("*"):
        if not path.is_file():
            continue
        if any(part in IGNORED_DIRS for part in path.parts):
            continue
        if path.name.startswith(".env"):
            continue  # .env / .env.example may contain placeholders
        if not any(path.match(g) for g in TRACKED_GLOBS) and path.suffix != ".sh":
            continue
        yield path


def test_no_literal_api_keys_in_tracked_files():
    offenders = []
    for path in _iter_tracked_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in SECRET_PATTERN.finditer(text):
            offenders.append(f"{path.relative_to(REPO)}: {match.group(0)[:10]}…")
    assert not offenders, (
        "literal API-key-shaped tokens found in tracked files — "
        "move them to env vars:\n  " + "\n  ".join(offenders)
    )


def test_test1_sh_requires_env_var():
    text = (REPO / "test1.sh").read_text(encoding="utf-8")
    assert "PULSAR_KEY" in text
    assert "?set PULSAR_KEY" in text or "?need PULSAR_KEY" in text or ":?" in text
    assert not SECRET_PATTERN.search(text)


def test_benchmark_module_imports_cleanly():
    # NB: do NOT importlib.reload(benchmark) here — that swaps class identities
    # and breaks subsequent isinstance() checks in tokenizer tests.
    import benchmark

    assert hasattr(benchmark, "benchmark_request")
    assert hasattr(benchmark, "run_benchmark")
    assert hasattr(benchmark, "_build_prompt_provider")
