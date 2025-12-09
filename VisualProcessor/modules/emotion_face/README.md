# Emotion Face Module

Модуль для анализа эмоций из лиц в видео. Извлекает эмоциональные признаки, включая классические эмоции Ekman, валентность/активацию (valence/arousal), динамику эмоций и ключевые кадры.

## Описание

Модуль `emotion_face` предназначен для глубокого анализа эмоционального состояния лиц в видео. Он использует современные модели машинного обучения для детекции лиц и распознавания эмоций, а также предоставляет расширенную аналитику эмоциональных паттернов.

## Основные возможности

- **Детекция лиц**: Использует InsightFace для обнаружения лиц в видео
- **Распознавание эмоций**: Модель EmoNet для классификации 8 базовых эмоций (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt)
- **Valence/Arousal**: Извлечение непрерывных значений валентности (позитивность) и активации (возбуждение)
- **Динамика эмоций**: Анализ переходов, скорости изменений и эмоциональной турбулентности
- **Ключевые кадры**: Автоматическое обнаружение ключевых моментов с эмоциональными переходами
- **Валидация качества**: Проверка качества извлеченной последовательности эмоций
- **Оптимизация памяти**: Эффективная обработка больших видео с использованием memory-mapped файлов

### Расширенные возможности (новое)

- **Микроэмоции**: Детекция резких эмоциональных изменений длительностью 0.03-0.5 секунды для анализа искренности и реакции на события
- **Физиологические сигналы**: Оценка уровня стресса, уверенности, нервозности и напряжения на основе паттернов эмоций
- **Асимметрия лица**: Анализ симметрии лица для оценки естественности и искренности выражения эмоций
- **Индивидуальность выражения**: Анализ стиля и интенсивности выражения эмоций, индекс выразительности

## Архитектура

Модуль состоит из следующих компонентов:

- **`process_video.py`**: Точка входа для обработки видео
- **`core/video_processor.py`**: Основной класс `VideoEmotionProcessor` для обработки
- **`core/advanced_emotion_features.py`**: Модуль расширенных фичей (микроэмоции, физиологические сигналы, асимметрия, индивидуальность)
- **`core/`**: Вспомогательные модули (конфигурация, валидация, управление памятью, retry стратегии)
- **`utils.py`**: Утилиты для обработки кадров, сегментации, анализа эмоций
- **`models/emonet/`**: Модель EmoNet для распознавания эмоций

## Установка

### Зависимости

```bash
pip install torch torchvision
pip install insightface
pip install opencv-python
pip install numpy
pip install psutil
```

### Модели

1. **EmoNet**: Модель должна быть размещена в `models/emonet/pretrained/emonet_8.pth`
2. **InsightFace**: Автоматически загружается при первом использовании

## Использование

### Базовый пример

```python
from process_video import init_face_app, load_emonet
from core.video_processor import VideoEmotionProcessor

# Инициализация
video_path = "path/to/video.mp4"
model = load_emonet("models/emonet/pretrained/emonet_8.pth")
face_app = init_face_app()

# Обработка
processor = VideoEmotionProcessor()
result = processor.process(
    video_path=video_path,
    model=model,
    face_app=face_app,
    target_length=256,  # Целевая длина последовательности
    chunk_size=64       # Размер чанка для обработки
)

if result.get("success"):
    emotions = result.get("emotions", [])
    keyframes = result.get("keyframes", {})
    print(f"Обработано {len(emotions)} кадров")
    print(f"Найдено {len(keyframes)} ключевых кадров")
```

### Конфигурация

Модуль поддерживает конфигурацию через `config.yaml`:

```yaml
validation:
  min_frames_ratio: 0.8
  min_keyframes: 3
  min_transitions: 2
  min_diversity_threshold: 0.2

caching:
  ttl_enabled: false
  ttl_seconds: 1800
  cache_size_limit: 10

logging:
  enable_structured_metrics: true
```

## Формат выходных данных

### Структура результата

```python
{
    "success": bool,
    "emotions": [
        {
            "valence": float,      # -1.0 до 1.0
            "arousal": float,      # -1.0 до 1.0
            "emotions": {
                "Neutral": float,
                "Happy": float,
                "Sad": float,
                "Surprise": float,
                "Fear": float,
                "Disgust": float,
                "Anger": float,
                "Contempt": float
            }
        }
    ],
    "keyframes": {
        frame_index: {
            "type": str,           # "transition" или "emotion_peak"
            "score": float,
            "valence_change": float,
            "arousal_change": float
        }
    },
    "indices": [int],              # Индексы обработанных кадров
    "quality_metrics": {
        "is_valid": bool,
        "overall_score": float,
        "diversity_score": float,
        "transition_score": float,
        "monotonicity_score": float,
        "variance_score": float
    },
    "processing_stats": {
        "total_frames": int,
        "faces_found": int,
        "dominant_emotion": str,
        "neutral_percentage": float,
        "valence_avg": float,
        "arousal_avg": float
    },
    "advanced_features": {
        "microexpressions": {
            "microexpressions_count": int,
            "microexpression_rate": float,
            "avg_duration": float,
            "microexpressions": [...]
        },
        "physiological_signals": {
            "stress_level_score": float,
            "confidence_face_score": float,
            "tension_face_index": float,
            "nervousness_score": float
        },
        "emotional_individuality": {
            "emotional_intensity_baseline": float,
            "expressivity_index": float,
            "emotional_range": float,
            "dominant_style": str,
            "emotional_style_vector": {...}
        },
        "face_asymmetry": {
            "asymmetry_score": float,
            "eyebrow_asymmetry": float,
            "mouth_asymmetry": float,
            "eye_asymmetry": float,
            "overall_symmetry": float,
            "sincerity_score": float
        }
    }
}
```

## Ключевые функции

### Сегментация и выборка кадров

- `segmentation()`: Разбивает временную линию на сегменты с лицами
- `select_from_segments()`: Адаптивная выборка кадров из сегментов
- `uniform_time_coverage()`: Равномерная выборка по всему видео

### Анализ эмоций

- `build_emotion_curve()`: Строит кривые валентности и активации
- `detect_keyframes()`: Находит ключевые кадры с эмоциональными переходами
- `analyze_emotion_profile()`: Анализирует профиль эмоций (доминантная эмоция, распределение)

### Расширенные фичи

- `detect_micro_expressions()`: Детектирует микроэмоции (резкие изменения длительностью 0.03-0.5 сек)
- `compute_physiological_signals()`: Вычисляет физиологические индексы (стресс, уверенность, нервозность)
- `compute_face_asymmetry()`: Анализирует асимметрию лица для оценки искренности
- `compute_emotional_individuality()`: Анализирует индивидуальность и стиль выражения эмоций

### Валидация и оптимизация

- `validate_sequence_quality()`: Проверяет качество последовательности эмоций
- `compress_sequence()`: Сжимает длинную последовательность до целевой длины
- `expand_sequence()`: Расширяет короткую последовательность с интерполяцией

## Особенности обработки

### Адаптивная обработка

Модуль автоматически адаптируется к типу видео:
- **STATIC_FACE**: Статичное лицо (большая часть кадров содержит лицо)
- **CONTINUOUS_FACE**: Непрерывное присутствие лица
- **DYNAMIC_FACES**: Динамические сцены с несколькими лицами

### Управление памятью

- Использование memory-mapped файлов для больших видео
- Батчевая обработка с автоматическим определением размера батча
- Кэширование результатов сканирования лиц
- Автоматическая очистка памяти между попытками

### Retry стратегия

Модуль поддерживает автоматические повторные попытки с адаптацией параметров:
- Снижение требований к качеству для монотонных видео
- Увеличение выборки для статичных лиц
- Адаптация порогов для различных типов видео

## Производительность

- **Обработка**: ~30-60 FPS (зависит от GPU и размера видео)
- **Память**: Оптимизировано для работы с видео любого размера
- **Точность**: Высокая точность распознавания эмоций благодаря EmoNet

## Ограничения

- Требует наличия лиц в видео для работы
- Лучше работает с фронтальными видами лиц
- Требует GPU для оптимальной производительности (поддерживается CPU fallback)

## Дополнительная информация

Подробное описание извлекаемых фичей и методологии доступно в файле `FEATURES.MD`.

## Лицензия

См. основной файл лицензии проекта.

