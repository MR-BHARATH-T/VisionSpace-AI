# utils/file_utils.py
"""
VisionSpace AI — Production File Utilities
==========================================
Provides robust file I/O with:
  - Atomic JSON read / write
  - SHA-256 / MD5 checksum generation and verification
  - Lightweight version-controlled JSON store (append-only log + HEAD pointer)
  - Directory helpers (ensure, clean, size)
  - Output path builder aligned with config.OUTPUT_DIR structure
  - Kaggle-safe path resolution (works when /kaggle/working is cwd)
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import config
from utils.constants import JSON_INDENT, CHECKSUM_ALGORITHM, OUTPUT_SUBDIRS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path Helpers
# ---------------------------------------------------------------------------

def resolve_output_path(subdir: str, filename: str) -> Path:
    """Return an absolute path inside config.OUTPUT_DIR / subdir / filename.

    Creates all intermediate directories automatically.  Works on both local
    Windows dev environments and Kaggle (/kaggle/working) without any change.

    Args:
        subdir:   One of the OUTPUT_SUBDIRS (e.g. "reports", "budgets").
        filename: Target filename including extension.

    Returns:
        Fully resolved :class:`pathlib.Path`.

    Example::

        path = resolve_output_path("reports", "session_001.json")
        # → <cwd>/outputs/reports/session_001.json
    """
    base = Path(config.OUTPUT_DIR).resolve()
    target = base / subdir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def ensure_output_dirs() -> None:
    """Create all standard output sub-directories defined in ``OUTPUT_SUBDIRS``.

    Safe to call multiple times; silently skips existing directories.
    """
    base = Path(config.OUTPUT_DIR).resolve()
    for sub in OUTPUT_SUBDIRS:
        (base / sub).mkdir(parents=True, exist_ok=True)
    logger.debug("Output directory tree verified at: %s", base)


def directory_size_mb(path: str | Path) -> float:
    """Return the total size of a directory tree in megabytes.

    Args:
        path: Root directory to measure.

    Returns:
        Total size in MB, or 0.0 if path does not exist.
    """
    root = Path(path)
    if not root.exists():
        return 0.0
    total = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
    return round(total / (1024 ** 2), 3)


def clean_directory(path: str | Path, keep_structure: bool = True) -> int:
    """Remove all files inside *path* without deleting the directory itself.

    Args:
        path:           Directory to clean.
        keep_structure: If True, sub-directories are kept but emptied.
                        If False, all contents including sub-dirs are removed.

    Returns:
        Number of files deleted.
    """
    root = Path(path)
    if not root.exists():
        return 0

    deleted = 0
    for item in root.rglob("*"):
        if item.is_file():
            item.unlink()
            deleted += 1
        elif item.is_dir() and not keep_structure:
            shutil.rmtree(item, ignore_errors=True)

    logger.info("Cleaned %d files from %s", deleted, root)
    return deleted


# ---------------------------------------------------------------------------
# Atomic JSON I/O
# ---------------------------------------------------------------------------

def save_json(data: Any, file_path: str | Path, *, indent: int = JSON_INDENT) -> None:
    """Atomically write *data* to *file_path* as JSON.

    Uses a temporary file + rename for crash-safe writes on all platforms.
    Parent directories are created automatically.

    Args:
        data:      Any JSON-serialisable Python object.
        file_path: Destination file path.
        indent:    JSON indentation level (default 4).

    Raises:
        TypeError: If *data* contains non-serialisable types.
    """
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=target.parent, suffix=".tmp", prefix=".write_"
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        # Atomic rename — on POSIX this is O_RENAME; on Windows it overwrites
        os.replace(tmp_path, target)
    except Exception:
        # Clean up orphan temp file on error
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    logger.debug("Saved JSON → %s (%d bytes)", target, target.stat().st_size)


def load_json(file_path: str | Path) -> Any:
    """Load and return the contents of a JSON file.

    Args:
        file_path: Source file path.

    Returns:
        Deserialised Python object.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    target = Path(file_path)
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def load_json_safe(file_path: str | Path, default: Any = None) -> Any:
    """Like :func:`load_json` but returns *default* on any error.

    Useful in initialisation paths where a missing/corrupt file is acceptable.
    """
    try:
        return load_json(file_path)
    except Exception as exc:
        logger.warning("load_json_safe: could not load %s — %s", file_path, exc)
        return default


def append_json_lines(records: list[dict], file_path: str | Path) -> None:
    """Append *records* to a JSON-Lines file (one JSON object per line).

    Creates the file if it does not exist.  Useful for streaming log outputs.

    Args:
        records:   List of dicts to append.
        file_path: Target ``.jsonl`` file.
    """
    target = Path(file_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Checksum Utilities
# ---------------------------------------------------------------------------

def compute_checksum(
    file_path: str | Path,
    algorithm: str = CHECKSUM_ALGORITHM,
    chunk_size: int = 1 << 20,   # 1 MB
) -> str:
    """Compute and return the hex-digest checksum of a file.

    Args:
        file_path:  Path to the file.
        algorithm:  Hash algorithm name understood by :mod:`hashlib`
                    (e.g. ``"sha256"``, ``"md5"``).
        chunk_size: Read chunk size in bytes (default 1 MB).

    Returns:
        Lowercase hex-digest string.

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError:        If *algorithm* is not supported.
    """
    try:
        h = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: '{algorithm}'") from exc

    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)

    return h.hexdigest()


def verify_checksum(
    file_path: str | Path,
    expected: str,
    algorithm: str = CHECKSUM_ALGORITHM,
) -> bool:
    """Return True if the file's checksum matches *expected*.

    Args:
        file_path: Path to the file to verify.
        expected:  Expected hex-digest string (case-insensitive).
        algorithm: Hash algorithm used when producing *expected*.

    Returns:
        ``True`` if checksums match, ``False`` otherwise.
    """
    actual = compute_checksum(file_path, algorithm=algorithm)
    match = actual.lower() == expected.lower()
    if not match:
        logger.warning(
            "Checksum mismatch for %s: expected=%s actual=%s",
            file_path, expected[:12] + "…", actual[:12] + "…",
        )
    return match


def save_json_with_checksum(
    data: Any,
    file_path: str | Path,
) -> str:
    """Save *data* as JSON and return its SHA-256 checksum.

    Useful for verifying integrity of exported scene snapshots.

    Returns:
        Hex-digest SHA-256 string of the written file.
    """
    save_json(data, file_path)
    return compute_checksum(file_path)


# ---------------------------------------------------------------------------
# Version-Controlled JSON Store
# ---------------------------------------------------------------------------

class VersionedJSONStore:
    """Lightweight version-controlled store for a single JSON document.

    Layout on disk::

        <root>/
          HEAD               — plain-text: version number of current HEAD
          v0001.json         — first committed version
          v0002.json         — second committed version
          ...

    This is intentionally simple (no branching, no diffs) — it is designed
    for session artefacts and agent outputs, not source code.

    Example::

        store = VersionedJSONStore("outputs/scenes/room_001")
        store.commit({"objects": [...], "style": "modern"})
        store.commit({"objects": [...], "style": "scandinavian"})  # v2
        data = store.load()          # HEAD → v2
        data = store.load(version=1) # explicit v1
        history = store.history()    # [{version, timestamp, size_bytes}, ...]
    """

    _VERSION_FMT = "v{:04d}.json"
    _HEAD_FILE = "HEAD"

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _head(self) -> int:
        head_file = self._root / self._HEAD_FILE
        if not head_file.exists():
            return 0
        return int(head_file.read_text(encoding="utf-8").strip())

    def _set_head(self, version: int) -> None:
        (self._root / self._HEAD_FILE).write_text(str(version), encoding="utf-8")

    def _version_path(self, version: int) -> Path:
        return self._root / self._VERSION_FMT.format(version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def commit(self, data: Any) -> int:
        """Write *data* as the next version and advance HEAD.

        Args:
            data: JSON-serialisable object.

        Returns:
            New version number (1-indexed).
        """
        next_version = self._head() + 1
        envelope = {
            "_version": next_version,
            "_committed_at": datetime.now(tz=timezone.utc).isoformat(),
            "data": data,
        }
        save_json(envelope, self._version_path(next_version))
        self._set_head(next_version)
        logger.info("VersionedJSONStore committed v%04d → %s", next_version, self._root)
        return next_version

    def load(self, version: int | None = None) -> Any:
        """Load a version's data payload.

        Args:
            version: Specific version number, or None for HEAD.

        Returns:
            The ``"data"`` field of the stored envelope.

        Raises:
            FileNotFoundError: If requested version does not exist.
            ValueError:        If the store is empty (no commits yet).
        """
        v = version if version is not None else self._head()
        if v == 0:
            raise ValueError(f"Store at '{self._root}' has no commits yet.")
        path = self._version_path(v)
        if not path.exists():
            raise FileNotFoundError(f"Version v{v:04d} not found in {self._root}")
        return load_json(path)["data"]

    def history(self) -> list[dict]:
        """Return a list of version metadata dicts, newest first.

        Each dict contains: ``version``, ``committed_at``, ``size_bytes``,
        ``checksum``.
        """
        records = []
        head = self._head()
        for v in range(head, 0, -1):
            path = self._version_path(v)
            if not path.exists():
                continue
            envelope = load_json(path)
            records.append({
                "version":      v,
                "committed_at": envelope.get("_committed_at", "unknown"),
                "size_bytes":   path.stat().st_size,
                "checksum":     compute_checksum(path),
            })
        return records

    def rollback(self, version: int) -> Any:
        """Re-set HEAD to *version* (does not delete newer versions).

        Returns:
            The data at the rolled-back version.
        """
        head = self._head()
        if version < 1 or version > head:
            raise ValueError(
                f"Invalid rollback target v{version}. "
                f"Store has versions 1–{head}."
            )
        self._set_head(version)
        logger.warning(
            "VersionedJSONStore rolled back from v%04d → v%04d in %s",
            head, version, self._root,
        )
        return self.load(version)

    def latest_version(self) -> int:
        """Return the current HEAD version number (0 if no commits)."""
        return self._head()


# ---------------------------------------------------------------------------
# Convenience Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Path helpers
    "resolve_output_path",
    "ensure_output_dirs",
    "directory_size_mb",
    "clean_directory",
    # JSON I/O
    "save_json",
    "load_json",
    "load_json_safe",
    "append_json_lines",
    # Checksums
    "compute_checksum",
    "verify_checksum",
    "save_json_with_checksum",
    # Versioned store
    "VersionedJSONStore",
]
