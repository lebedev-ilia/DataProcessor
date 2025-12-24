# Core Optical Flow Provider

## Описание

Core провайдер для вычисления оптического потока между последовательными кадрами видео. Использует RAFT (Recurrent All-Pairs Field Transforms) или Farneback для оценки движения. Запускается один раз на видео в фазе core-провайдеров.

## Расположение

- **Провайдер**: `VisualProcessor/modules/optical_flow/` (пока работает как модуль, в будущем будет перенесён в `core/model_process/core_optical_flow/`)
- **Выходные данные**: `result_store/optical_flow/statistical_analysis.json` (и другие файлы)

## Формат выходных данных

### Файл: `statistical_analysis.json`

Основной файл со статистикой оптического потока:

```json
{
  "version": "1.0",
  "created_at": "2025-01-15T10:30:00.123456",
  "statistics": {
    "frame_statistics": [
      {
        "frame_index": 0,
        "magnitude_mean": 0.123,
        "magnitude_std": 0.045,
        "magnitude_max": 0.567,
        "magnitude_mean_px_sec": 12.3,
        "magnitude_mean_px_sec_norm": 0.45,
        "direction_mean": 1.23,
        "direction_std": 0.12,
        ...
      },
      ...
    ],
    "video_level": {
      "mean_magnitude": 0.234,
      "std_magnitude": 0.056,
      ...
    }
  }
}
```

### Ключевые поля в `frame_statistics`:

- `frame_index` (int): Индекс кадра
- `magnitude_mean` (float): Средняя величина оптического потока (в пикселях)
- `magnitude_mean_px_sec` (float): Средняя величина в пикселях в секунду
- `magnitude_mean_px_sec_norm` (float): Нормализованная величина (0..1)
- `direction_mean` (float): Средний угол направления движения (в радианах)
- Дополнительные поля зависят от конфигурации (camera motion, clusters, etc.)

## Параметры запуска

```bash
python modules/optical_flow/main.py \
    --frames-dir <path_to_frames> \
    --rs-path <result_store_path> \
    --model "small" \
    --max-dim 256 \
    --run-stats \
    --enable-advanced-features
```

### Основные параметры

- `--frames-dir` (required): Путь к директории с кадрами
- `--rs-path` (required): Путь к result_store
- `--model` (default: "small"): Модель RAFT ("small" или "large")
- `--max-dim` (default: 256): Максимальный размер стороны для обработки
- `--run-stats` (flag): Запустить статистический анализ
- `--enable-advanced-features` (flag): Включить расширенные фичи (camera motion, clusters, etc.)

## Использование в модулях

Модули, которые используют core_optical_flow:

- `video_pacing` — для motion energy curves и pacing analysis
- `story_structure` — для определения кульминаций и структуры повествования
- `cut_detection` — для обнаружения срезов на основе движения
- `text_scoring` — для alignment текста с движением
- `optical_flow` (сам модуль) — для детального анализа движения

### Пример чтения данных

```python
import json
import os

def load_core_optical_flow(rs_path: str):
    """Загружает статистику оптического потока из core_optical_flow."""
    stats_path = os.path.join(rs_path, "optical_flow", "statistical_analysis.json")
    if not os.path.isfile(stats_path):
        return None
    
    with open(stats_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    frame_stats = (data.get("statistics") or {}).get("frame_statistics") or []
    
    # Извлекаем кривую движения
    motion_curve = []
    for fs in frame_stats:
        v = (
            fs.get("magnitude_mean_px_sec_norm")
            if "magnitude_mean_px_sec_norm" in fs
            else fs.get("magnitude_mean_px_sec", fs.get("magnitude_mean", 0.0))
        )
        motion_curve.append(float(v))
    
    return {
        "motion_curve": motion_curve,
        "frame_statistics": frame_stats,
        "video_level": data.get("statistics", {}).get("video_level", {}),
    }
```

## Версионирование

- **Версия 1.0**: Базовый формат со статистикой по кадрам

## Зависимости

- `torch`
- `raft` (RAFT optical flow model)
- `numpy`
- `scipy` (для статистики)

