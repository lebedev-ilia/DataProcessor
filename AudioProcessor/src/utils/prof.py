"""
Tiny profiling helpers for inline timing prints.

Usage:
    from src.utils.prof import timeit
    with timeit("mel: torchaudio.melspectrogram"):
        mel = transform(x)
"""

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def timeit(label: str) -> Iterator[None]:
    t0 = perf_counter()
    try:
        yield
    finally:
        dt = perf_counter() - t0
        try:
            print(f"[TIMER] {label}: {dt:.3f}s")
        except Exception:
            # Fallback in environments where print is overridden
            pass


