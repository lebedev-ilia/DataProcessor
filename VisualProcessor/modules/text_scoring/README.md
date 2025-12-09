# 📝 Text-Video Interaction & Emphasis Feature Extraction

## 📌 Описание

Модуль предназначен для извлечения признаков взаимодействия текста с визуальными событиями видео, а также оценки мультимодальной важности текста. Он **не анализирует содержание текста** (OCR / семантику), а фокусируется на том, **как текст влияет на визуальный и мультимодальный контекст**:

- **Синхронизация текста** с движением, действиями, событиями
- **Мультимодальная эмфаза** (motion + лицо + аудио пики)
- **Call-to-action (CTA) detection** и усиление
- **Динамика появления текста** на экране

## 🔹 Установка

```bash
pip install numpy scipy opencv-python-headless
```

> **Примечание:** GPU не требуется, можно использовать полностью на CPU, но для больших видео желательно минимизировать размер кадров.

## 🔧 Входные данные

### OCR Data

Список словарей с информацией по кадрам:

```python
ocr_data = [
    {
        "frame": 10,
        "bbox": (x1, y1, x2, y2),
        "text": "...",
        "confidence": 0.95,
        "is_cta": False
    },
    {
        "frame": 12,
        "bbox": (x1, y1, x2, y2),
        "text": "...",
        "confidence": 0.98,
        "is_cta": True
    },
    # ...
]
```

### Дополнительные сигналы

- `motion_peaks` — список float, motion intensity по кадрам
- `face_peaks` — список float, face presence peaks по кадрам
- `audio_peaks` (опционально) — аудио энергия по кадрам

## 📊 Выходные фичи

| Категория | Фичи | Описание |
|-----------|------|----------|
| **Text-Video Interaction** | `text_action_sync_score` | Синхронизация текста с движением / действиями |
| | `text_motion_alignment` | Усиленный мультимодальный сигнал (motion + face + audio) |
| **Multimodal Emphasis** | `multimodal_attention_boost_score` | Максимальный мультимодальный акцент текста |
| | `text_emphasis_peak_flags` | Кадры с сильной визуальной эмфазой текста |
| **CTA** | `cta_presence` | Наличие CTA текста (0/1) |
| | `cta_timestamp` | Время появления CTA (с) |
| | `cta_strength` | Мультимодальное усиление CTA |
| **Temporal Text Dynamics** | `text_on_screen_continuity` | Средняя длительность появления текста |
| | `text_switch_rate` | Частота смены текста на экране |

## 🔹 Используемые алгоритмы и улучшения

### Text → Action Correlation

Пересечение текста с motion / face / audio сигналами для оценки влияния текста на видео.

### Multimodal Emphasis

Комбинирует motion, face, audio пики для вычисления `multimodal_attention_boost_score`.

### Сглаживание сигналов

Gaussian smoothing для уменьшения шумов и ложных пиков.

### Нормализация

Позволяет сравнивать видео разной длины и интенсивности движения.

### CTA Detection

Усиление текста, который является call-to-action, с учетом мультимодальных сигналов.

### Динамика текста

`text_on_screen_continuity` и `text_switch_rate` показывают, как долго и как часто текст появляется на экране.

## 🔹 Пример использования

```python
from text_video_pipeline import TextVideoInteractionPipeline

# Подготовка данных
ocr_data = [
    {
        "frame": 10,
        "bbox": (50, 50, 300, 100),
        "text": "Subscribe!",
        "confidence": 0.99,
        "is_cta": True
    },
    {
        "frame": 12,
        "bbox": (60, 55, 310, 110),
        "text": "Subscribe!",
        "confidence": 0.98,
        "is_cta": True
    },
]

motion_peaks = [0.1]*15 + [0.8, 0.9, 0.85] + [0.1]*20
face_peaks = [0.0]*15 + [0.5, 0.6, 0.55] + [0.0]*20
audio_peaks = [0.1]*15 + [0.7, 0.8, 0.75] + [0.1]*20

# Инициализация
pipeline = TextVideoInteractionPipeline(video_fps=30)

# Извлечение фичей
features = pipeline.extract_features(
    ocr_data,
    motion_peaks,
    face_peaks,
    audio_peaks
)

# Вывод результатов
for k, v in features.items():
    print(f"{k}: {v}")
```

## 💡 Особенности

- ✅ Оптимизирован для мультимодального анализа текста на видео
- ✅ Легко интегрируется с остальными модулями (Frame Composition, Shot Quality, Story Structure)
- ✅ Позволяет извлекать синхронизацию текста с ключевыми визуальными событиями
- ✅ Рассчитан для YouTube Shorts / TikTok, где текст → акцент / CTA / мемы

## 🎯 Применение

- Анализ эффективности текстовых элементов
- Оценка синхронизации текста с действиями
- Детекция и анализ CTA элементов
- Мультимодальный анализ важности текста
- Оптимизация размещения текста в видео
