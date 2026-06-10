"""IO helpers: parquet read/write, JSON manifests, file checksums.

Heavy deps (pyarrow) are imported lazily so importing this module never fails when
only validation/imputation on in-memory frames is needed (e.g. unit tests).
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (matches Binance .CHECKSUM files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    return path


def read_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")


def write_manifest(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str, sort_keys=True)
    return path


def read_manifest(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git_commit_hash(cwd: str | Path | None = None, short: bool = True) -> str | None:
    """Return the current git commit hash, or None if unavailable.

    Read-only: never modifies the repo. Used to stamp reports for reproducibility.
    """
    import subprocess

    args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.run(
            args, cwd=str(cwd) if cwd else None,
            capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None
