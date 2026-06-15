"""Per-request log file schema and JSONL/CSV writer factory."""

from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Optional

RECORD_FIELDS = (
    "trial",
    "request_index",
    "prompt_chars",
    "prompt_tokens",
    "output_tokens",
    "latency_s",
    "ttft_s",
    "scheduled_offset_s",  # only set in open-loop QPS mode
    "queue_wait_s",        # only set in open-loop QPS mode
    "ok",
    "estimated",
    "error",
)


@contextmanager
def _record_writer(
    path: Optional[str],
) -> Generator[Optional[Callable[[Dict[str, Any]], None]], None, None]:
    """Yield a row-writer callable for `path`, or None if `path` is empty.

    Format is selected by file extension: `.jsonl` or `.csv`. Unknown
    extensions raise ValueError.
    """
    if not path:
        yield None
        return

    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        with open(path, "w", encoding="utf-8") as f:
            def write_jsonl(row: Dict[str, Any]) -> None:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            yield write_jsonl
    elif suffix == ".csv":
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(RECORD_FIELDS))
            writer.writeheader()
            yield writer.writerow  # type: ignore[misc]
    else:
        raise ValueError(
            f"--output: unsupported extension '{suffix}' for {path!r}; "
            "use .jsonl or .csv"
        )
