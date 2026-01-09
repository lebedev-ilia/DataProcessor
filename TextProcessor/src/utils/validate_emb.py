# quick_check_three_embeddings.py
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

from src.core.path_utils import default_artifacts_dir

FILE_PATTERNS = [
    "title_embedding_*.npy",
    "description_embedding_*.npy",
    "transcript_whisper_embedding_*.npy",
    "transcript_youtube_auto_embedding_*.npy",
]


def _latest(pattern: str) -> str | None:
    art = default_artifacts_dir()
    files = list(art.glob(pattern))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(files[0])

def load_and_check(path):
    print("\n=== Файл:", path)
    if not os.path.exists(path):
        print("  Файл не найден!")
        return None
    arr = np.load(path)
    print("  shape:", arr.shape, " dtype:", arr.dtype)
    # Приводим к форме (n,d)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
        print("  -> reshaped to", arr.shape)
    n, d = arr.shape
    # базовые проверки
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    zero_count = int((arr == 0).sum())
    zero_frac = zero_count / (n * d)
    l2 = np.linalg.norm(arr, axis=1)
    print(f"  has_nan: {has_nan}, has_inf: {has_inf}")
    print(f"  zero_count: {zero_count} ({zero_frac*100:.4f}%)")
    print(f"  L2 norms (per row): min {l2.min():.6g}, mean {l2.mean():.6g}, max {l2.max():.6g}")
    # первые элементы первой строки
    print("  first row - first 20 values:", np.round(arr[0, :20], 6))
    # basic stats per-dim (если много данных — это просто оценка)
    print("  per-dim std mean (est):", float(arr.std(axis=0).mean()))
    # если только одна строка - rank = 1 (обычно), но выдадим top singular value
    try:
        s = np.linalg.svd(arr, compute_uv=False)
        print("  singular values (top 3):", [float(x) for x in s[:3]])
        print("  matrix_rank:", int(np.linalg.matrix_rank(arr)))
    except Exception as e:
        print("  SVD error:", e)
    return arr

def pairwise_cosines(list_of_arrays, names=None):
    mats = []
    for a in list_of_arrays:
        if a is None:
            mats.append(None)
        else:
            # сводим каждую к вектору (если несколько строк — усредняем, но у тебя 1 строка)
            if a.shape[0] == 1:
                mats.append(a[0].reshape(1, -1))
            else:
                mats.append(a.mean(axis=0).reshape(1, -1))
    # составим матрицу из тех, что не None
    valid_idx = [i for i, m in enumerate(mats) if m is not None]
    if not valid_idx:
        print("Нет валидных массивов для сравнения.")
        return
    stacked = np.vstack([mats[i] for i in valid_idx])
    cos = cosine_similarity(stacked)
    print("\n=== Косинусная матрица (между файлами, по порядку VALID):")
    for i_row, i in enumerate(valid_idx):
        row_name = names[i] if names else str(i)
        print(f"  [{i_row}] {os.path.basename(FILES[i])} ->", end=" ")
        vals = ["{:.6f}".format(x) for x in cos[i_row]]
        print(" ".join(vals))
    print("\nПримечание: значения ~1 означают почти идентичные вектора; <<1 — различия.")

def main():
    loaded = []
    files = []
    for pat in FILE_PATTERNS:
        p = _latest(pat)
        files.append(p)
    for fp in files:
        if fp is None:
            loaded.append(None)
            continue
        a = load_and_check(fp)
        loaded.append(a)
    pairwise_cosines(loaded, names=[(os.path.basename(f) if f else "missing") for f in files])

if __name__ == "__main__":
    main()
