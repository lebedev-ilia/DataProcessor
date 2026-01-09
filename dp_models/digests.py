from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


def sha256_file(path: str, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass(frozen=True)
class DirDigestOptions:
    """
    Options for directory digesting.

    NOTE: This is best-effort and can be expensive for large model repos.
    """

    ignore_names: Tuple[str, ...] = (
        ".git",
        "__pycache__",
        ".DS_Store",
    )
    ignore_suffixes: Tuple[str, ...] = (
        ".tmp",
        ".lock",
    )
    max_files: int = 50_000


def _should_ignore(name: str, *, opts: DirDigestOptions) -> bool:
    if name in opts.ignore_names:
        return True
    for suf in opts.ignore_suffixes:
        if name.endswith(suf):
            return True
    return False


def _iter_files(root_dir: str, *, opts: DirDigestOptions) -> List[Tuple[str, str]]:
    """
    Returns sorted list of (abs_path, rel_path) for all files.
    """
    out: List[Tuple[str, str]] = []
    for base, dirs, files in os.walk(root_dir):
        # prune ignored dirs in-place
        dirs[:] = [d for d in dirs if not _should_ignore(d, opts=opts)]
        for fn in files:
            if _should_ignore(fn, opts=opts):
                continue
            abs_path = os.path.join(base, fn)
            rel_path = os.path.relpath(abs_path, root_dir)
            out.append((abs_path, rel_path))
            if len(out) > int(opts.max_files):
                raise RuntimeError(f"Directory digest exceeds max_files={opts.max_files}: {root_dir}")
    out.sort(key=lambda t: t[1])
    return out


def sha256_dir(root_dir: str, *, opts: Optional[DirDigestOptions] = None) -> str:
    """
    Deterministic directory digest:
    sha256 of concatenation of per-file records (path + size + sha256(file)).
    """
    opts = opts or DirDigestOptions()
    files = _iter_files(root_dir, opts=opts)
    h = hashlib.sha256()
    for abs_path, rel_path in files:
        st = os.stat(abs_path)
        h.update(rel_path.encode("utf-8"))
        h.update(b"\0")
        h.update(str(int(st.st_size)).encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(abs_path).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


