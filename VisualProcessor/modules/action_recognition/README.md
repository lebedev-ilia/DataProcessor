# 🎬 Action Recognition with VideoMAE

## 📌 Описание

Модуль для распознавания действий в видео с использованием модели VideoMAE и motion-aware агрегации результатов. Особенностью реализации является использование optical flow для вычисления motion magnitude и взвешенной агрегации результатов с учетом движения в кадрах.

**Новые возможности:**
- ✅ Fine-grained actions (детальная классификация действий)
- ✅ Multi-person actions (групповые действия)
- ✅ Action complexity score (оценка сложности действий)
- ✅ Action planning & intent (предсказание будущих действий)
- ✅ Scene activity type (анализ уровня активности сцены)
- ✅ CLI интерфейс для обработки видео

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install torch transformers opencv-python numpy scipy
```

### CLI использование

Самый простой способ обработки видео:

```bash
python main.py --input video.mp4 --model /path/to/model --output results.json
```

**Параметры CLI:**

```bash
python main.py \
    --input video.mp4 \                    # Входное видео (обязательно)
    --model /path/to/model \               # Путь к модели VideoMAE (обязательно)
    --output results.json \                # Выходной JSON файл (опционально)
    --frames-dir /path/to/frames \         # Директория с кадрами (опционально)
    --clip-len 16 \                        # Длина клипа в кадрах
    --batch-size 8 \                       # Размер батча
    --frame-skip 1 \                       # Пропуск кадров
    --max-tracks 5                         # Максимальное количество треков
```

### Базовое использование (Python API)

```python
from pathlib import Path
import sys

# Добавляем путь для импорта FrameManager
emotion_face_path = Path(__file__).parent.parent / "emotion_face"
if str(emotion_face_path) not in sys.path:
    sys.path.insert(0, str(emotion_face_path))

from action_recognition_videomae import VideoMAEActionRecognizer
from utils import FrameManager

# Инициализация
fm = FrameManager(FRAMES_DIR)
recognizer = VideoMAEActionRecognizer(
    frame_manager=fm,
    model_name=MODEL_DIR,
    clip_len=16,
    batch_size=8
)

# Обработка треков
frame_indices_per_person = {
    1: list(range(10, 120, 2)),
    2: list(range(100, 220, 2))
}

results = recognizer.process(frame_indices_per_person)
print(results)

# Очистка ресурсов
fm.close()
```

## 🏗️ Архитектура

### Основные компоненты

#### `VideoMAEActionRecognizer`

Главный класс для распознавания действий.

**Параметры инициализации:**

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `frame_manager` | Экземпляр `FrameManager` для загрузки кадров | - |
| `model_name` | Путь к предобученной модели VideoMAE | - |
| `clip_len` | Длина клипа в кадрах | 16 |
| `stride` | Шаг для скользящего окна | `clip_len // 2` |
| `batch_size` | Размер батча для inference | 8 |
| `device` | Устройство для вычислений | 'auto' |

**Основные методы:**

- `process(frame_indices_per_person)` — главный метод для обработки треков
  - Принимает словарь `{track_id: [список индексов кадров]}`
  - Возвращает словарь с результатами для каждого трека

### Обработка данных

#### 1. Загрузка кадров и вычисление motion

Метод `_load_frames_with_motion()`:
- Загружает кадры через `FrameManager`
- Вычисляет optical flow между соседними кадрами (Farneback algorithm)
- Вычисляет magnitude движения для каждого кадра
- Обрабатывает различные форматы кадров (grayscale, RGB, RGBA)

#### 2. Создание клипов

Метод `_make_clips()`:
- Применяет скользящее окно с заданным stride
- Для коротких треков дополняет последним кадром до `clip_len`
- Вычисляет средний motion magnitude для каждого клипа

#### 3. Inference

Метод `_infer()`:
- Обрабатывает клипы батчами
- Использует `VideoMAEFeatureExtractor` для предобработки
- Возвращает вероятности для всех классов действий

#### 4. Агрегация результатов

Метод `_aggregate()`:
- **Motion-weighted averaging** — клипы с большим движением получают больший вес
- **Exponential Moving Average (EMA)** — временное сглаживание вероятностей
- **Статистики:**
  - `dominant_action` — действие с максимальной вероятностью
  - `mode_action` — наиболее частое действие по клипам
  - `stability` — доля самого длинного непрерывного действия
  - `switch_rate_per_sec` — частота смены действий в секунду
  - `mean_entropy` — средняя энтропия распределения вероятностей
  - `topk_labels` и `topk_vals` — топ-5 действий с вероятностями

## 📊 Формат выходных данных

```python
{
    track_id: {
        # Базовые статистики
        "num_clips": int,                    # Количество обработанных клипов
        "mean_entropy": float,               # Средняя энтропия
        "dominant_action": int,              # ID доминирующего действия
        "dominant_confidence": float,        # Уверенность в доминирующем действии
        "dominant_action_label": str,        # Название действия (если доступно)
        "topk_labels": List[int],            # Топ-5 ID действий
        "topk_vals": List[float],           # Вероятности топ-5 действий
        "stability": float,                  # Стабильность (0-1)
        "switch_rate_per_sec": float,        # Частота смены действий
        "mode_action": int,                  # Наиболее частое действие
        "mode_confidence": float,            # Уверенность в mode действии
        "ema_confidence": float,             # EMA уверенность
        
        # Fine-grained actions
        "category_scores": dict,             # Оценки по категориям детальных действий
        "fine_grained_detections": list,     # Детекции fine-grained действий
        "num_fine_grained_actions": int,     # Количество обнаруженных fine-grained действий
        
        # Action complexity
        "complexity_score": float,           # Общая оценка сложности действия (0-1)
        "entropy_complexity": float,         # Сложность на основе энтропии
        "diversity_score": float,            # Разнообразие действий
        "coordination_level": float,         # Уровень координации
        "action_precision": float,           # Точность действий
        "motion_complexity": float,          # Сложность движения
        "num_unique_actions": int,           # Количество уникальных действий
        
        # Action planning & intent
        "predicted_action": int,             # Предсказанное следующее действие
        "predicted_action_label": str,       # Название предсказанного действия
        "prediction_confidence": float,      # Уверенность в предсказании
        "action_preparation_time": float,    # Время подготовки к действию (сек)
        "intent_score": float,               # Оценка намерения
        
        # Scene activity
        "scene_activity_type": str,          # Тип активности: "high_action_intensity", "low_action_intensity", "chaotic_motion", "static"
        "scene_activity_score": float,       # Общая оценка активности сцены
        "activity_scores": dict,             # Оценки по всем типам активности
        "action_entropy": float,             # Энтропия действий
        "action_change_rate": float,        # Частота смены действий
        "dominant_action_ratio": float,      # Доля доминирующего действия
        "weighted_activity_entropy": float,  # Взвешенная энтропия активности
        
        # Multi-person actions (если несколько треков)
        "multi_person_context": dict,        # Контекст групповых действий
            # {
            #     "is_multi_person": bool,
            #     "num_persons": int,
            #     "multi_person_actions": list,  # Обнаруженные групповые действия
            #     "group_activity_type": str,     # Тип групповой активности
            #     "action_synchronization": float  # Синхронизация действий (0-1)
            # }
    }
}
```

## ⚙️ Настройки и параметры

### Рекомендуемые значения

| Параметр | Рекомендуемое значение |
|----------|----------------------|
| `clip_len` | 16 кадров (стандарт для VideoMAE) |
| `stride` | `clip_len // 2` или `clip_len // 3` для более плотного покрытия |
| `batch_size` | 4-8 (зависит от доступной памяти GPU) |

### Оптимизация производительности

- ✅ Используйте GPU для ускорения inference
- ✅ Увеличьте `batch_size` если есть свободная память
- ✅ Для длинных треков можно увеличить `stride` для уменьшения количества клипов

## 🔧 Требования к данным

### FrameManager

Модуль ожидает, что `FrameManager`:
- Имеет метод `get(idx: int) -> np.ndarray` для получения кадра по индексу
- Имеет атрибут `fps` (опционально) для правильного вычисления временных метрик
- Возвращает кадры в формате RGB (H, W, 3) или grayscale (H, W)

### Формат входных данных

```python
frame_indices_per_person: Dict[int, List[int]]
```

- **Ключ:** уникальный ID трека (person_id)
- **Значение:** список индексов кадров, где присутствует этот трек

## 🎯 Возможные улучшения

### 1. Динамический stride

Вместо фиксированного stride использовать адаптивный:
- Для коротких треков: `stride = 1` (максимальная детализация)
- Для длинных треков: `stride = clip_len // 2` или `clip_len // 3`

### 2. Интеллектуальный выбор кадров

Фильтрация статичных кадров:
- Использовать motion magnitude для отбора ключевых кадров
- Удалять кадры с движением ниже порога

### 3. Multi-scale клипы

Обработка клипов разной длины:
- Короткие клипы (16 кадров) для быстрых действий
- Длинные клипы (32-64 кадра) для медленных действий
- Агрегация результатов через усреднение или энтропию

### 4. Дополнительные статистики

Расширение метрик в `_aggregate()`:
- **Skewness / Kurtosis** — асимметрия распределения вероятностей
- **Top-k entropy** — энтропия только по топ-3 вероятностям
- **Action diversity index** — `1 - Σ(p_i²)` (индекс Джини)
- **Temporal autocorrelation** — корреляция меток с задержкой 1-2 клипа

### 5. Интерполяция для коротких треков

Вместо дублирования последнего кадра:
- Использовать линейную интерполяцию между кадрами
- Дублировать кадры с наибольшим движением

### 6. Data augmentation при inference

Для повышения robustness:
- Легкие трансформации: crop, flip, brightness adjustment
- Test-time augmentation с усреднением результатов

### 7. Интеграция с трекингом

Дополнительные признаки из трекинга:
- Средняя скорость движения (по центру bbox)
- Изменение размера bbox (приближение/удаление)
- Стабильность позиции в кадре

## 🆕 Новые функции

### Fine-grained Actions

Модуль анализирует детальные (fine-grained) действия, такие как:
- `face_touch` - прикосновения к лицу
- `hair_touch` - прикосновения к волосам
- `pointing` - указание
- `waving` - махание рукой
- `nodding` - кивание головой
- `shrugging` - пожимание плечами
- `clapping` - хлопки
- `lip_sync` - синхронизация губ
- `scrolling` - прокрутка на устройстве
- `reacting` - реакции на экран

### Multi-person Actions

При обработке нескольких треков модуль автоматически определяет групповые действия:
- `group_walking` - групповая ходьба
- `fighting` - драка
- `hugging` - объятия
- `handshakes` - рукопожатия
- `arguing` - спор
- `teaching` - обучение
- `collaborating` - сотрудничество
- `crowds_running` - бег толпы
- `dancing_together` - совместные танцы

### Action Complexity Score

Оценка сложности действия на основе:
- Энтропии распределения вероятностей
- Разнообразия действий
- Координации (плавность переходов)
- Точности (уверенность в действиях)
- Сложности движения

### Action Planning & Intent

Предсказание будущих действий и анализ намерений:
- Предсказание следующего действия на основе временной последовательности
- Время подготовки к действию
- Оценка намерения (intent score)

### Scene Activity Type

Классификация общего уровня активности сцены:
- `high_action_intensity` - высокая интенсивность действий
- `low_action_intensity` - низкая интенсивность действий
- `chaotic_motion` - хаотичное движение
- `static` - статичная сцена

## 🐛 Известные проблемы и ограничения

1. **Цветовое пространство** — модуль предполагает RGB формат кадров. Если кадры в BGR, нужно изменить `COLOR_RGB2GRAY` на `COLOR_BGR2GRAY` в методе `_load_frames_with_motion()`.

2. **Память** — для больших батчей может потребоваться много GPU памяти. Уменьшите `batch_size` при нехватке памяти.

3. **Производительность optical flow** — вычисление optical flow для каждого кадра может быть медленным. Для ускорения можно уменьшить разрешение кадров перед вычислением flow.

## 📝 Примеры использования

### Обработка одного трека

```python
frame_indices = list(range(0, 100, 2))  # Каждый второй кадр
results = recognizer.process({1: frame_indices})
print(f"Доминирующее действие: {results[1]['dominant_action_label']}")
print(f"Уверенность: {results[1]['dominant_confidence']:.2f}")
```

### Обработка нескольких треков

```python
tracks = {
    1: list(range(10, 120, 2)),
    2: list(range(100, 220, 2)),
    3: list(range(50, 150, 3))
}

results = recognizer.process(tracks)

for track_id, features in results.items():
    print(f"Track {track_id}:")
    print(f"  Действие: {features.get('dominant_action_label', 'unknown')}")
    print(f"  Стабильность: {features['stability']:.2f}")
    print(f"  Смен действий: {features['switch_rate_per_sec']:.2f} в сек")
```

## 📚 Зависимости

- `torch` — PyTorch для работы с моделью
- `transformers` — Hugging Face transformers для VideoMAE
- `opencv-python` — OpenCV для optical flow
- `numpy` — NumPy для работы с массивами
- `scipy` — SciPy для статистических функций (опционально)

## 🔗 Связанные модули

- `emotion_face/utils.py` — реализация `FrameManager`
- `behavior/` — модули трекинга и детекции объектов

## 📄 Лицензия

См. основной LICENSE проекта.
