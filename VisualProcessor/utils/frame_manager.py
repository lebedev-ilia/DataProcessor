from pathlib import Path
import numpy as np
import json


class FrameManager:

    def __init__(self, frames_dir: str, chunk_size: int = 32, cache_size: int = 2):
        self.frames_dir = Path(frames_dir)
        meta_path = self.frames_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.frames_dir}")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)
        self.total_frames = int(self.meta["total_frames"])
        # NOTE: legacy Segmenter писал `chunk_size`, новые пайплайны могут писать `batch_size`.
        self.batch_size = int(self.meta.get("batch_size") or self.meta.get("chunk_size") or chunk_size)
        self.batches = self.meta["batches"]
        # Добавлено: читаем размеры из self.meta
        self.height = self.meta["height"]
        self.width = self.meta["width"]
        self.channels = self.meta["channels"]
        self.fps = self.meta.get("fps", 30)
        self.color_space = self.meta.get("color_space")
        # create mapping of batch_index -> batch_info
        self.batch_by_idx = {b["batch_index"]: b for b in self.batches}
        self.cache = {}  # pid -> memmap ndarray
        self.cache_order = []  # LRU list
        self.cache_size = int(cache_size)

    def _load_batch_pid(self, pid: int):
        if pid in self.cache:
            # move to MRU
            try:
                self.cache_order.remove(pid)
            except ValueError:
                pass
            self.cache_order.append(pid)
            return self.cache[pid]
        batch = self.batch_by_idx.get(pid)
        if batch is None:
            raise IndexError(f"batch {pid} not found")
        p = self.frames_dir / batch["path"]
        # Добавлено: вычисляем реальный num_frames для батча (для частичных)
        num_frames = batch["end_frame"] - batch["start_frame"] + 1
        # Поддерживаем два формата хранения:
        # - `.npy` (legacy): читаем через np.load(..., mmap_mode="r")
        # - raw uint8 tensor без заголовка: np.memmap с известной shape
        if p.suffix.lower() == ".npy":
            arr = np.load(str(p), mmap_mode="r")
        else:
            arr = np.memmap(
                str(p),
                dtype=np.uint8,
                mode="r",
                shape=(num_frames, self.height, self.width, self.channels),
            )
        self.cache[pid] = arr
        self.cache_order.append(pid)
        # evict
        while len(self.cache_order) > self.cache_size:
            ev = self.cache_order.pop(0)
            self.cache.pop(ev, None)
        return arr

    def get(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.total_frames:
            raise IndexError("Frame index out of bounds")
        pid = idx // self.batch_size
        local = idx - pid * self.batch_size
        arr = self._load_batch_pid(pid)
        if local >= arr.shape[0]:
            # should not happen but guard
            raise IndexError(f"Local index {local} >= chunk size {arr.shape[0]} (global {idx})")
        return np.asarray(arr[local])

    # Добавлено: метод для очистки кэша (вызовем после видео)
    def close(self):
        self.cache.clear()
        self.cache_order.clear()
