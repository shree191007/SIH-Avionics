"""
SIH26054 Replan to Learn: Atomic multi-process artifact persistence (M3, F11).
write-to-temp-then-rename plus a filelock, so a concurrent reader never sees
a partially-written artifact and two writers never interleave.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock


def lock_path_for(target: Path) -> Path:
    return target.with_name(f"{target.name}.lock")


def atomic_write_bytes(target: Path, data: bytes) -> Path:
    """Atomically write `data` to `target`, guarded by a sidecar file lock."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path_for(target))):
        fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, target)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)
    return target


def atomic_write_json(target: Path, obj: Any) -> Path:
    return atomic_write_bytes(target, json.dumps(obj, indent=2, sort_keys=True).encode("utf-8"))


def atomic_write_via(target: Path, writer: Callable[[Path], None]) -> Path:
    """
    Atomically produce `target` using a `writer(tmp_path)` callback -- for
    formats (e.g. Parquet via pyarrow) that need to write to a real file path
    rather than an in-memory bytes buffer. `writer` must write its full
    output to the temp path it's given; the temp file is then atomically
    renamed into place under the same file lock as atomic_write_bytes.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(lock_path_for(target))):
        fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            writer(tmp_path)
            os.replace(tmp_name, target)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    return target
