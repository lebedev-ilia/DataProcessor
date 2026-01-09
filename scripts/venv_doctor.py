from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class VenvCheck:
    name: str
    python_path: str
    required_imports: List[str]
    install_hint: Optional[str] = None


def _run(py: str, code: str) -> subprocess.CompletedProcess:
    return subprocess.run([py, "-c", code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _exists_file(path: str) -> bool:
    return os.path.isfile(path)


def _ok(msg: str) -> None:
    print(f"[OK] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _err(msg: str) -> None:
    print(f"[ERROR] {msg}")


def _check_binaries() -> None:
    for bin_name in ("ffmpeg", "ffprobe"):
        p = shutil.which(bin_name)
        if p:
            _ok(f"{bin_name} найден: {p}")
        else:
            _warn(f"{bin_name} не найден в PATH (Segmenter audio extraction может упасть)")


def _check_venv(v: VenvCheck) -> bool:
    ok = True
    if not _exists_file(v.python_path):
        _warn(f"{v.name}: python не найден: {v.python_path}")
        if v.install_hint:
            _warn(f"{v.name}: hint: {v.install_hint}")
        return False

    _ok(f"{v.name}: python найден: {v.python_path}")

    r = _run(v.python_path, "import sys; print(sys.version)")
    if r.returncode != 0:
        _err(f"{v.name}: не смог запустить python: {r.stderr.strip()}")
        return False

    for mod in v.required_imports:
        rr = _run(v.python_path, f"import {mod}")
        if rr.returncode != 0:
            ok = False
            _warn(f"{v.name}: import {mod} FAILED: {rr.stderr.strip().splitlines()[-1] if rr.stderr else 'unknown error'}")

    if not ok and v.install_hint:
        _warn(f"{v.name}: чтобы починить: {v.install_hint}")
    return ok


def main(argv: Optional[List[str]] = None) -> int:
    _ = argv
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    checks: List[VenvCheck] = [
        VenvCheck(
            name="DataProcessor (.data_venv) — orchestrator + Segmenter (smoke)",
            python_path=os.path.join(repo_root, ".data_venv", "bin", "python"),
            required_imports=["yaml", "numpy", "cv2"],
            install_hint=f"{os.path.join(repo_root, '.data_venv', 'bin', 'pip')} install -r {os.path.join(repo_root, 'requirements', 'dataprocessor_smoke.txt')}",
        ),
        VenvCheck(
            name="VisualProcessor (.vp_venv) — orchestrator (VisualProcessor/main.py) + modules (default)",
            python_path=os.path.join(repo_root, "VisualProcessor", ".vp_venv", "bin", "python"),
            required_imports=["yaml", "numpy", "cv2"],
            install_hint=None,
        ),
        VenvCheck(
            name="VisualProcessor core_face_landmarks (.core_face_landmarks_venv)",
            python_path=os.path.join(
                repo_root,
                "VisualProcessor",
                "core",
                "model_process",
                "core_face_landmarks",
                ".core_face_landmarks_venv",
                "bin",
                "python",
            ),
            required_imports=["mediapipe", "numpy", "cv2"],
            install_hint=None,
        ),
    ]

    print("=== venv doctor ===")
    print(f"repo_root: {repo_root}")
    _check_binaries()

    all_ok = True
    for c in checks:
        print()
        ok = _check_venv(c)
        all_ok = all_ok and ok

    print()
    if all_ok:
        _ok("Все проверки пройдены")
        return 0
    _warn("Есть проблемы с окружениями (см. WARN выше)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


