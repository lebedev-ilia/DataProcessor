# Core CLIP Embeddings Provider

## Описание

Core провайдер для вычисления CLIP-эмбеддингов на кадрах видео. Запускается один раз на видео в фазе core-провайдеров, сохраняет универсальные визуальные эмбеддинги для использования всеми модулями.

## Расположение

- **Провайдер**: `VisualProcessor/core/model_process/core_clip/main.py`
- **Выходные данные**: `result_store/core_clip/embeddings.npz`

## Формат выходных данных

### Файл: `embeddings.npz`

NPZ-архив с ключами:

- `frame_embeddings` (numpy.ndarray, shape `[N, D]`):
  - Per-frame CLIP эмбеддинги, где `N` — количество кадров, `D` — размерность эмбеддинга (обычно 512 для ViT-B/32)
  - Эмбеддинги нормализованы (L2-norm = 1.0)
  - Порядок соответствует последовательности кадров (frame 0, frame 1, ..., frame N-1)

- `model_name` (str):
  - Название модели CLIP (например, "ViT-B/32", "ViT-L/14")

- `created_at` (str):
  - ISO-формат времени создания: `YYYY-MM-DDTHH:MM:SS.microseconds`

- `total_frames` (int):
  - Общее количество кадров в видео

## Параметры запуска

```bash
python core/model_process/core_clip/main.py \
    --frames-dir <path_to_frames> \
    --rs-path <result_store_path> \
    --model-name "ViT-B/32" \
    --batch-size 32
```

### Параметры

- `--frames-dir` (required): Путь к директории с кадрами (от Segmenter)
- `--rs-path` (required): Путь к result_store
- `--model-name` (default: "ViT-B/32"): Название модели CLIP
- `--batch-size` (default: 32): Размер батча для обработки

## Использование в модулях

Модули, которые используют core_clip:

- `video_pacing` — для semantic energy curves
- `story_structure` — для topic clustering и semantic segmentation
- `high_level_semantic` — для high-level визуальных фич
- `cut_detection` — для определения срезов на основе семантической схожести
- `shot_quality` — для оценки семантического качества кадров

### Пример чтения данных

```python
import numpy as np
import os

def load_core_clip_embeddings(rs_path: str):
    """Загружает CLIP эмбеддинги из core_clip провайдера."""
    npz_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
    if not os.path.isfile(npz_path):
        return None
    
    data = np.load(npz_path)
    return {
        "embeddings": data["frame_embeddings"],  # [N, D]
        "model_name": str(data.get("model_name", "ViT-B/32")),
        "total_frames": int(data.get("total_frames", 0)),
    }
```

## Версионирование

- **Версия 1.0**: Базовый формат с per-frame embeddings

## Зависимости

- `torch`
- `clip` (OpenAI CLIP)
- `PIL` (Pillow)
- `numpy`

