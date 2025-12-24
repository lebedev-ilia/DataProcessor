# Core Face Landmarks Provider

## Описание

Core провайдер для извлечения landmarks лица, позы и рук с помощью Mediapipe. Запускается один раз на видео в фазе core-провайдеров, сохраняет универсальные геометрические данные для использования всеми модулями.

## Расположение

- **Провайдер**: `VisualProcessor/core/model_process/core_face_landmarks/main.py`
- **Выходные данные**: `result_store/core_face_landmarks/landmarks.json`

## Формат выходных данных

### Файл: `landmarks.json`

JSON-файл со структурой:

```json
{
  "version": "1.0",
  "created_at": "2025-01-15T10:30:00.123456",
  "total_frames": 1000,
  "use_pose": true,
  "use_hands": true,
  "use_face_mesh": true,
  "frames": [
    {
      "frame_index": 0,
      "pose_landmarks": [
        {"x": 0.5, "y": 0.3, "z": 0.1, "visibility": 0.9},
        ...
      ],
      "hands_landmarks": [
        [
          {"x": 0.4, "y": 0.5, "z": 0.2},
          ...
        ],
        [
          {"x": 0.6, "y": 0.5, "z": 0.2},
          ...
        ]
      ],
      "face_landmarks": [
        [
          {"x": 0.45, "y": 0.4, "z": 0.15},
          ...
        ]
      ]
    },
    ...
  ]
}
```

### Структура данных по кадрам:

- `frame_index` (int): Индекс кадра

- `pose_landmarks` (list, optional): Landmarks позы тела (33 точки Mediapipe Pose)
  - Каждая точка: `{"x": float, "y": float, "z": float, "visibility": float}`
  - Координаты нормализованы (0..1 относительно размера кадра)
  - `visibility` — уверенность видимости точки (0..1)

- `hands_landmarks` (list, optional): Landmarks рук
  - Список рук (обычно 0-2 руки)
  - Каждая рука — список из 21 точки: `[{"x": float, "y": float, "z": float}, ...]`
  - Координаты нормализованы (0..1)

- `face_landmarks` (list, optional): Landmarks лица
  - Список лиц (обычно 0-4 лица)
  - Каждое лицо — список из 468 точек (если `refine_landmarks=True`) или 468 точек: `[{"x": float, "y": float, "z": float}, ...]`
  - Координаты нормализованы (0..1)

## Параметры запуска

```bash
python core/model_process/core_face_landmarks/main.py \
    --frames-dir <path_to_frames> \
    --rs-path <result_store_path> \
    --use-pose \
    --use-hands \
    --use-face-mesh
```

### Параметры

- `--frames-dir` (required): Путь к директории с кадрами
- `--rs-path` (required): Путь к result_store
- `--use-pose` (flag): Включить детекцию позы тела
- `--use-hands` (flag): Включить детекцию рук
- `--use-face-mesh` (flag): Включить детекцию лица (face mesh)

## Использование в модулях

Модули, которые используют core_face_landmarks:

- `behavioral` — для анализа поведения (поза, жесты, движения)
- `detalize_face_modules` — для детального анализа лица (геометрия, качество, освещение и т.д.)
- `frames_composition` — для композиционного анализа с учётом лиц
- `text_scoring` — для alignment текста с присутствием лиц

### Пример чтения данных

```python
import json
import os

def load_core_face_landmarks(rs_path: str):
    """Загружает landmarks из core_face_landmarks провайдера."""
    landmarks_path = os.path.join(rs_path, "core_face_landmarks", "landmarks.json")
    if not os.path.isfile(landmarks_path):
        return None
    
    with open(landmarks_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    frames = data.get("frames") or []
    
    # Преобразуем в dict[frame_index] -> frame_payload для удобства
    landmarks_dict = {int(f["frame_index"]): f for f in frames if "frame_index" in f}
    
    return {
        "landmarks": landmarks_dict,
        "total_frames": data.get("total_frames", 0),
        "use_pose": data.get("use_pose", False),
        "use_hands": data.get("use_hands", False),
        "use_face_mesh": data.get("use_face_mesh", False),
    }
```

## Версионирование

- **Версия 1.0**: Базовый формат с pose/hands/face landmarks

## Зависимости

- `mediapipe`
- `numpy`
- `opencv-python` (для конвертации цветов)

