# Micro Emotion Module

Модуль для анализа микроэмоций и Action Units (AU) с использованием OpenFace через Docker. Извлекает детальные признаки лица, включая мимические единицы, позу головы, направление взгляда и 3D landmarks.

## Описание

Модуль `micro_emotion` предназначен для глубокого анализа мимики лица с использованием OpenFace - инструмента для автоматического распознавания лицевых признаков. Модуль работает через Docker контейнер, что обеспечивает изоляцию и простоту развертывания.

## Основные возможности

- **Action Units (AU)**: Извлечение 45 мимических единиц с интенсивностью и наличием
- **Pose головы**: Оценка положения и ориентации головы (6 DOF)
- **Gaze direction**: Направление взгляда (углы по X и Y)
- **Facial Landmarks**: 68 точек 2D и 3D landmarks
- **Анализ кадров**: Обработка отдельных кадров или целых видео
- **Batch processing**: Поддержка пакетной обработки кадров

## Требования

### Docker

Модуль требует установленного Docker и образа OpenFace:

```bash
# Установка Docker (если не установлен)
sudo apt install docker.io

# Загрузка образа OpenFace
docker pull openface/openface:latest
```

### Python зависимости

```bash
pip install pandas numpy opencv-python
```

## Установка

1. Убедитесь, что Docker установлен и запущен
2. Загрузите образ OpenFace: `docker pull openface/openface:latest`
3. Модуль готов к использованию

## Использование

### Базовый пример

```python
from main import OpenFaceAnalyzer

# Инициализация анализатора
analyzer = OpenFaceAnalyzer()

# Анализ видео
results = analyzer.analyze_video(
    video_path="path/to/video.mp4",
    output_name="analysis_result",
    features="all"  # "all", "basic", "au", "pose", "gaze"
)

if results:
    print(f"Обработано кадров: {results['summary']['total_frames']}")
    print(f"Кадров с лицами: {results['summary']['frames_with_face']}")
    print(f"Найдено AU: {results['summary']['au_count']}")
```

### Анализ отдельных кадров

```python
import cv2

# Загрузка кадра
frame = cv2.imread("frame.jpg")

# Анализ одного изображения
result = analyzer.analyze_single_image(
    image_path="frame.jpg",
    output_prefix="frame_analysis"
)
```

### Анализ списка кадров

```python
frames = [cv2.imread(f"frame_{i}.jpg") for i in range(10)]
frame_indices = list(range(10))

results = analyzer.analyze_frames(
    frames=frames,
    frame_indices=frame_indices,
    output_prefix="batch_analysis"
)
```

### Сохранение результатов

```python
# Сохранение в JSON
json_path = analyzer.save_results(results, output_path="results.json")
```

## Формат выходных данных

### Структура результата

```python
{
    "success": bool,
    "face_count": int,
    "success_rate": float,
    "action_units": {
        "AU01": {
            "intensity_mean": float,
            "intensity_std": float,
            "presence_mean": float,
            "presence_std": float
        },
        # ... другие AU
    },
    "pose": {
        "pose_Rx": {"mean": float, "std": float, "min": float, "max": float},
        "pose_Ry": {...},
        "pose_Rz": {...},
        "pose_Tx": {...},
        "pose_Ty": {...},
        "pose_Tz": {...}
    },
    "gaze": {
        "gaze_angle_x": {"mean": float, "std": float},
        "gaze_angle_y": {"mean": float, "std": float}
    },
    "facial_landmarks_2d": [
        {
            "x_mean": float,
            "x_std": float,
            "y_mean": float,
            "y_std": float
        },
        # ... 68 точек
    ],
    "facial_landmarks_3d": [
        {
            "X_mean": float,
            "Y_mean": float,
            "Z_mean": float
        },
        # ... 68 точек
    ],
    "summary": {
        "total_frames": int,
        "frames_with_face": int,
        "au_count": int,
        "landmarks_2d_count": int,
        "landmarks_3d_count": int,
        "timestamp": str
    },
    "dataframe": pd.DataFrame,  # Полный DataFrame с данными OpenFace
    "csv_path": str
}
```

## Параметры анализа

### Типы признаков (features)

- **`"all"`**: Все признаки (pose, AU, gaze, 2D/3D landmarks, tracking)
- **`"basic"`**: Базовые признаки (pose, 2D landmarks)
- **`"au"`**: Только Action Units
- **`"pose"`**: Только поза головы
- **`"gaze"`**: Только направление взгляда

### Action Units (AU)

OpenFace извлекает следующие основные AU:
- **AU01**: Inner Brow Raiser
- **AU02**: Outer Brow Raiser
- **AU04**: Brow Lowerer
- **AU05**: Upper Lid Raiser
- **AU06**: Cheek Raiser
- **AU07**: Lid Tightener
- **AU09**: Nose Wrinkler
- **AU10**: Upper Lip Raiser
- **AU12**: Lip Corner Puller
- **AU14**: Dimpler
- **AU15**: Lip Corner Depressor
- **AU17**: Chin Raiser
- **AU20**: Lip Stretcher
- **AU23**: Lip Tightener
- **AU25**: Lips Part
- **AU26**: Jaw Drop
- **AU28**: Lip Suck
- И другие (всего до 45 AU)

## Архитектура

Модуль использует Docker для запуска OpenFace, что обеспечивает:

- **Изоляцию**: Не требует установки OpenFace в системе
- **Портативность**: Работает на любой системе с Docker
- **Версионность**: Легко переключаться между версиями OpenFace

### Структура директорий

```
micro_emotion/
├── main.py                 # Основной класс OpenFaceAnalyzer
├── input_videos/           # Входные видео (монтируются в Docker)
├── output_data/            # Выходные CSV и результаты
├── temp_frames/            # Временные кадры для анализа
└── results/                # Сохраненные JSON результаты
```

## Примеры использования

### Полный анализ видео

```python
analyzer = OpenFaceAnalyzer()
results = analyzer.analyze_video(
    video_path="video.mp4",
    features="all"
)

# Доступ к Action Units
for au_name, au_data in results['action_units'].items():
    print(f"{au_name}: интенсивность={au_data['intensity_mean']:.3f}")

# Доступ к позе
print(f"Поворот по X: {results['pose']['pose_Rx']['mean']:.3f}")
print(f"Поворот по Y: {results['pose']['pose_Ry']['mean']:.3f}")

# Доступ к полному DataFrame
df = results['dataframe']
print(df.head())
```

### Командная строка

```bash
python main.py --video_path video.mp4 --output analysis --features all
```

## Производительность

- **Скорость**: ~10-30 FPS (зависит от разрешения и сложности сцены)
- **Точность**: Высокая точность благодаря OpenFace
- **Ресурсы**: Требует значительных вычислительных ресурсов

## Ограничения

- Требует Docker и образ OpenFace
- Лучше работает с фронтальными видами лиц
- Может быть медленным для длинных видео
- Требует достаточного места на диске для временных файлов

## Интеграция с другими модулями

Результаты OpenFace могут быть использованы для:
- Анализа микроэмоций (в сочетании с модулем `emotion_face`)
- Оценки искренности и естественности
- Анализа физиологических сигналов (стресс, напряжение)
- Детекции асимметрии лица

## Дополнительные ресурсы

- [OpenFace Documentation](https://github.com/TadasBaltrusaitis/OpenFace)
- [Action Units Reference](https://www.cs.cmu.edu/~face/facs.htm)

## Лицензия

См. основной файл лицензии проекта.

