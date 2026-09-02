from __future__ import annotations

import contextlib
import os
from pathlib import Path


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a pid-suffixed temp file and os.replace.

    A reader sees either the old file or the complete new one, never a torn
    write, and concurrent writers (sharded runs share cache paths) overwrite
    each other with whole files. The temp name carries the pid so two
    processes never share one temp file.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(data)
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))
