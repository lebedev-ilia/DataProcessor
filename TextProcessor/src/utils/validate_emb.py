# quick_check_three_embeddings.py
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

FILES = [
    "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts/title_embedding_bddb4dc824b703bf4e14a4435cf710b32f483cd93d87192e79ba1260b5d0efb9.npy",
    "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts/description_embedding_d980b647fd1794bd8ffeea49cf9caba70dba325a1f25062b71e9080226f22afb.npy",
    "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts/transcript_whisper_embedding_bc5d3fa9c0d68a3237c9347a9f838f4a5894417adfc65dcfe1d8f4b541b7204a.npy",
    "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts/transcript_youtube_auto_embedding_9774d229f4ca95fdc60ff0f76499c8f22ddcf4c4a3b413ff1d8bb734e3e64893.npy"
]

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
    for fp in FILES:
        a = load_and_check(fp)
        loaded.append(a)
    pairwise_cosines(loaded, names=[os.path.basename(f) for f in FILES])

if __name__ == "__main__":
    main()
