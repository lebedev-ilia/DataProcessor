# 📘 Shot Quality Pipeline

## 📌 Описание

Модуль выполняет автоматический анализ технического качества видеосъёмки. На вход подаётся последовательность кадров (numpy RGB изображения), на выход — детальные фичи, описывающие:

- **Резкость** — качество фокусировки и детализация
- **Шум** — уровень шумов и артефактов
- **Экспозицию** — баланс яркости и контраста
- **Цвет** — качество цветопередачи и баланс белого
- **Сжатие** — артефакты компрессии
- **Оптику** — качество объектива
- **Загрязнения** — дефекты объектива и атмосферные эффекты
- **Временную стабильность** — стабильность качества по времени
- **Профессиональный уровень** — общая оценка через NIMA

## 📦 Возможности

### 🔍 1. Sharpness & Clarity

**Метрики:**
- Laplacian Variance
- Tenengrad
- SMD2 (Sum of Modified Differences)
- Spatial Frequency Mean
- Blur Score (общий показатель размытия)
- DnCNN → детектор размытия
- Motion blur probability
- Focus accuracy score
- Edge clarity index

### 🧼 2. Noise Estimation

**Методы:**
- DnCNN noise estimation
- CBDNet noise model (готовая обученная сеть)
- Noise level (Luma) — уровень шума в яркостном канале
- Noise level (Chroma) — уровень шума в цветовых каналах
- ISO estimated value — оценка ISO на основе уровня шума
- Grain strength — сила зерна
- Noise spatial entropy — пространственная энтропия шума

### 🌑 3. Exposure

**Анализ через гистограммы:**
- `underexposure_ratio` — доля недоэкспонированных пикселей
- `overexposure_ratio` — доля переэкспонированных пикселей
- `exposure_histogram_skewness` — асимметрия экспозиции
- `midtones_balance` — баланс средних тонов
- `highlight_recovery_potential` — возможность восстановления светов
- `shadow_recovery_potential` — возможность восстановления теней
- `exposure_consistency_over_time` — стабильность экспозиции во времени

### 🔅 4. Contrast

**Типы контраста:**
- Global contrast
- Local contrast
- Dynamic range — динамический диапазон контраста
- Contrast clarity score — оценка четкости контраста
- Microcontrast — микроконтраст (важный показатель качества линзы)

### 🎨 5. Color Quality

**Метрики:**
- White balance shift (R, G, B каналы)
- Color cast type (red/green/blue/neutral)
- Skin tone accuracy score — точность передачи тона кожи
- Color fidelity index — индекс цветовой точности
- Color noise level — уровень цветового шума
- Color uniformity score — оценка равномерности цвета

### 🎚 6. Compression Artifacts

**Детекция артефактов:**
- Blockiness score — артефакты блочности
- Banding intensity — интенсивность полос
- Ringing artifacts level — уровень артефактов ringing (звон)
- Bitrate estimation score — оценка битрейта
- Codec artifact entropy — энтропия артефактов кодека

### 📐 7. Lens Quality

**Оценка оптики:**
- Chromatic aberration level — уровень хроматических аберраций
- Vignetting level — уровень виньетирования
- Distortion type — тип дисторсии (barrel/pincushion/none)
- Lens sharpness drop-off — снижение резкости к краям
- Lens obstruction probability — вероятность препятствий на объективе
- Lens dirt probability — вероятность грязи на объективе
- Veiling glare score — оценка veiling glare (засветка)

### 🌫 8. Dirt / Fog / Obstructions

**Детекция дефектов:**
- Lens dirt probability
- Foggy/haze score
- Veiling glare

### 🎞 9. Temporal Quality

**Временная стабильность:**
- `temporal_sharpness_stability` — стабильность резкости во времени
- `temporal_noise_variation` — вариация шума во времени
- `temporal_exposure_stability` — стабильность экспозиции во времени
- `temporal_flicker_score` — мерцание между кадрами
- `rolling_shutter_artifacts_score` — эффект rolling shutter (искажения при быстром движении)

### ⭐ 10. Shot Quality Classifier

Используется **CLIP (ViT-L/14)** для zero-shot классификации качества + эстетическая оценка.

**Выходные вероятности:**
- `quality_cinematic_prob` — кинематографическое качество
- `quality_lowlight_cinematic_prob` — кинематографическое качество в условиях низкой освещенности
- `quality_smartphone_good_prob` — хорошее качество смартфона
- `quality_smartphone_poor_prob` — плохое качество смартфона
- `quality_webcam_prob` — качество веб-камеры
- `quality_screenrecord_prob` — качество записи экрана
- `quality_surveillance_prob` — качество видеонаблюдения
- `aesthetic_score` — эстетический score (0-1)
- `clip_embedding` — CLIP embedding вектор (768 dims)

## ⚙️ Архитектура Pipeline

```
1. Анализ каждого кадра (frame-level metrics)
   ↓
2. Агрегация (mean, std, trend → temporal metrics)
   ↓
3. NIMA quality classifier
   ↓
4. Формирование итогового словаря всех метрик
```

## 🚀 Использование

### CLI интерфейс

Самый простой способ обработки видео:

```bash
python main.py --input video.mp4 --output results.json
```

**Параметры CLI:**

```bash
python main.py \
    --input video.mp4 \                    # Входное видео (обязательно)
    --output results.json \                 # Выходной JSON файл (опционально)
    --frame-skip 1 \                        # Обрабатывать каждый N-й кадр (по умолчанию: 1)
    --max-frames 1000 \                    # Максимальное количество кадров (опционально)
    --device cuda                           # Устройство: cuda или cpu (по умолчанию: cuda)
```

### Программный интерфейс

```python
from shot_quality_pipline import ShotQualityPipeline
import cv2

# Инициализация
pipeline = ShotQualityPipeline(device="cuda")

# Последовательность RGB кадров
frames = [
    cv2.cvtColor(cv2.imread("frame_001.jpg"), cv2.COLOR_BGR2RGB),
    cv2.cvtColor(cv2.imread("frame_002.jpg"), cv2.COLOR_BGR2RGB)
]

# Обработка последовательности кадров
results = pipeline.process(frames, frame_skip=1)

# Результат содержит:
# - results["frames"] - покадровые метрики
# - results["frame_features"] - агрегированные метрики (avg, std, min, max)
# - results["temporal_features"] - временные метрики

# Обработка отдельного кадра
frame = cv2.imread("frame.jpg")
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
result = pipeline.process_frame(frame_bgr)
```

## 📁 Структура выходных данных

```python
{
    # Покадровые метрики (список для каждого кадра)
    "frames": [
        {
            "sharpness_laplacian": float,
            "sharpness_tenengrad": float,
            "blur_score": float,
            "noise_level_luma": float,
            "noise_level_chroma": float,
            "underexposure_ratio": float,
            "overexposure_ratio": float,
            "contrast_global": float,
            "microcontrast": float,
            "rolling_shutter_artifacts_score": float,
            # ... и другие (70+ метрик на кадр)
        },
        # ... для каждого кадра
    ],
    
    # Агрегированные метрики (статистики по всем кадрам)
    "frame_features": {
        "avg_sharpness_laplacian": float,
        "std_sharpness_laplacian": float,
        "min_sharpness_laplacian": float,
        "max_sharpness_laplacian": float,
        "avg_noise_level_luma": float,
        # ... для всех метрик (avg, std, min, max)
    },
    
    # Временные метрики
    "temporal_features": {
        "temporal_sharpness_stability": float,
        "temporal_noise_variation": float,
        "temporal_exposure_stability": float,
        "exposure_consistency_over_time": float
    },
    
    # Метаданные
    "metadata": {
        "video_path": str,
        "fps": float,
        "total_frames": int,
        "processed_frames": int,
        "frame_skip": int
    },
    
    "total_frames_processed": int
}
```

**Всего от 70 до 120+ фичей на кадр**, полностью готовых к использованию в ML-модели (регрессия популярности, ранжирование качества, подбор кадров, QA и т.п.).

## 🧩 Зависимости

```bash
pip install opencv-python numpy torch torchvision clip-by-openai scipy pillow
```

**Опционально:**
- `aesthetic-predictor` — для расширенной эстетической оценки (если не установлен, используется упрощенная версия)

### Автоматическая загрузка моделей

Все модели загружаются автоматически:

| Модель | Источник |
|--------|----------|
| DnCNN | Torch Hub |
| CBDNet | Веса встроены |
| MiDaS | `torch.hub.load("intel-isl/MiDaS")` |
| NIMA | Локально встроенная MobileNetV2 + веса |

## 📄 Лицензии моделей

| Модель | Лицензия |
|--------|----------|
| DnCNN | MIT |
| CBDNet | Исследовательская лицензия от авторов |
| MiDaS | MIT License |
| NIMA | Research only |

## ❓ FAQ

**Q: Можно ли добавлять свои метрики?**  
A: Да, структура легко расширяется.

**Q: Можно ли заменить NIMA на CLIP?**  
A: Да, если нужна семантическая оценка.

**Q: Можно использовать для real-time?**  
A: Да, модель работает быстрее **40 FPS** на RTX 3060.

## 🎯 Применение

- Регрессия популярности видео
- Ранжирование качества съёмки
- Автоматический подбор лучших кадров
- QA для видеопроизводства
- Оценка технического качества контента
