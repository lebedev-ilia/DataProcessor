# 👤 Detalize Face Modules

## 📌 Описание

Модульная архитектура для `DetalizeFaceExtractor` с использованием модульной архитектуры, аналогичной `EmotionEngine`. Рефакторинг обеспечивает гибкость, расширяемость и легкость тестирования.

## 🏗️ Архитектура

```
detalize_face_modules/
├── __init__.py
├── detalize_face_refactored.py  # Рефакторинг главного класса
├── modules/
│   ├── __init__.py
│   ├── base_module.py           # Базовый интерфейс FaceModule
│   ├── geometry_module.py       # Геометрические фичи
│   ├── pose_module.py           # Поза головы
│   ├── quality_module.py        # Качество изображения
│   ├── lighting_module.py       # Освещение
│   ├── skin_module.py           # Кожа
│   ├── accessories_module.py    # Аксессуары
│   ├── eyes_module.py           # Глаза
│   ├── motion_module.py         # Движение
│   ├── structure_module.py      # Структура лица
│   ├── lip_reading_module.py    # Lip reading features
│   ├── face_3d_module.py        # 3D face reconstruction
│   └── professional_module.py  # Профессиональные фичи
└── utils/
    ├── __init__.py
    ├── landmarks_utils.py       # Утилиты для landmarks
    ├── bbox_utils.py            # Утилиты для bbox
    └── face_helpers.py          # Вспомогательные функции
```

## 🚀 Использование

### Базовое использование

```python
from detalize_face_modules.detalize_face_refactored import DetalizeFaceExtractorRefactored
import cv2
import numpy as np

# Загружаем кадры
frames = [...]  # список numpy массивов (кадры)

# Создаем экстрактор со всеми модулями
extractor = DetalizeFaceExtractorRefactored(
    modules=None,  # None = загрузить все модули
    max_faces=4,
    refine_landmarks=True,
)

# Или выборочно загружаем модули
extractor = DetalizeFaceExtractorRefactored(
    modules=["geometry", "pose", "quality", "eyes"],
    module_configs={
        "motion": {"fps": 30.0},
    },
)

# Извлекаем фичи
results = extractor.extract(frames)

# Результаты содержат:
# - frame_results: список результатов по кадрам
# - Каждый результат содержит фичи от всех модулей
```

## 📦 Модули

### GeometryModule

**Извлекает геометрические фичи лица:**
- Размер и позиция bbox
- Морфометрические характеристики
- Форма лица

**Зависимости:** `coords`, `bbox`, `frame_shape`, `face_idx`

### PoseModule

**Извлекает фичи позы головы:**
- Yaw, pitch, roll
- Вариативность позы
- Частота поворотов головы
- Внимание к камере

**Зависимости:** `coords`, `frame_shape`, `face_idx`

### QualityModule

**Извлекает фичи качества изображения:**
- Размытие
- Резкость
- Шум
- Видимость лица

**Зависимости:** `roi`, `bbox`, `frame_shape`

### LightingModule

**Извлекает фичи освещения:**
- Яркость
- Равномерность
- Контраст
- Баланс белого

**Зависимости:** `roi`, `coords_roi` (опционально)

### SkinModule

**Извлекает фичи кожи:**
- Макияж
- Гладкость кожи
- Борода/усы
- Форма бровей

**Зависимости:** `roi`, `coords_roi`

### AccessoriesModule

**Извлекает фичи аксессуаров:**
- Очки/солнцезащитные очки
- Маска
- Шапка/шлем
- Серьги
- Украшения

**Зависимости:** `roi`, `coords_roi`

### EyesModule

**Извлекает фичи глаз:**
- Открытие глаз
- Частота моргания
- Направление взгляда
- Позиция радужки

**Зависимости:** `coords`, `pose`, `face_idx`

### MotionModule

**Извлекает фичи движения:**
- Скорость лица
- Микро-выражения
- Движение рта/челюсти
- Движение бровей

**Зависимости:** `coords`, `geometry`, `face_idx`

### StructureModule

**Извлекает структурные фичи:**
- Вектор формы лица
- Вектор идентичности
- Вектор выражения
- Симметрия

**Зависимости:** `coords`, `pose`

### LipReadingModule

**Извлекает продвинутые lip reading features:**
- Mouth shape parameters (ширина, высота, площадь)
- Lip contour features (компактность, форма)
- Phoneme-like features (округлая, широкая, узкая, открытая формы)
- Speech activity probability
- Temporal patterns (скорость движения губ, цикличность)
- Mouth motion intensity и velocity

**Зависимости:** `coords`, `motion`, `face_idx`

### Face3DModule

**Извлекает 3D face reconstruction features (упрощенная версия 3DMM):**
- 3D face mesh vector (100-300 параметров)
- Identity shape vector (структурные особенности лица)
- Expression vector (50+ параметров выражения)
- Jaw pose vector
- Eye pose vector
- Mouth shape params
- Face symmetry score (продвинутый расчет)
- Face uniqueness score

**Зависимости:** `coords`, `pose`

### ProfessionalModule

**Извлекает профессиональные фичи:**
- Face quality score (NIMA-like)
- Perceived attractiveness score
- Emotion intensity (expressiveness)
- **Улучшенный Lip reading features** (интеграция с LipReadingModule)
- **Улучшенный Fatigue score** (полный анализ):
  - Eye-based fatigue indicators (закрытость, асимметрия, аномалии моргания)
  - Pose-based fatigue indicators (наклон головы, нестабильность)
  - Motion-based fatigue indicators (медленные движения, низкая активность)
  - Temporal patterns (тренды усталости во времени)
- Engagement level
- Alertness score
- Expressiveness score

**Зависимости:** `quality`, `eyes`, `motion`, `pose`, `lip_reading` (опционально)

## 🔧 Создание нового модуля

Чтобы создать новый модуль, наследуйтесь от `FaceModule`:

```python
from .base_module import FaceModule

class MyFaceModule(FaceModule):
    def required_inputs(self) -> List[str]:
        """Объявляем зависимости"""
        return ["coords", "bbox"]
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатываем данные"""
        coords = data["coords"]
        bbox = data["bbox"]
        
        # Ваша логика обработки
        result = my_feature_extraction(coords, bbox)
        
        # Возвращаем результаты с ключом, соответствующим имени модуля
        return {
            "my_module": result
        }
```

Затем зарегистрируйте модуль в `modules/__init__.py`:

```python
from .my_module import MyFaceModule

MODULE_REGISTRY = {
    # ... существующие модули
    "my_module": MyFaceModule,
}
```

## 🎯 Преимущества архитектуры

1. **Модульность** — модули можно добавлять, убирать, заменять без изменения кода
2. **Зависимости** — модули могут зависеть от результатов других модулей
3. **Гибкость** — модули могут работать независимо
4. **Масштабируемость** — легко добавлять новые алгоритмы
5. **Тестируемость** — каждый модуль можно тестировать отдельно

## 📊 Формат результатов

```python
[
    [  # Кадр 0
        {  # Лицо 0
            "frame_index": 0,
            "face_index": 0,
            "bbox": [x1, y1, x2, y2],
            "geometry": {...},
            "pose": {...},
            "quality": {...},
            "lighting": {...},
            "skin": {...},
            "accessories": {...},
            "eyes": {...},
            "motion": {...},
            "structure": {...},
            "lip_reading": {...},
            "face_3d": {...},
            "professional": {...},
        },
        # ... другие лица
    ],
    # ... другие кадры
]
```

## 🔄 Миграция с оригинального класса

Оригинальный класс `DetalizeFaceExtractor` остается без изменений. Новый рефакторинг находится в `detalize_face_refactored.py`.

### Для миграции

1. Замените импорт `DetalizeFaceExtractor` на `DetalizeFaceExtractorRefactored`
2. Обновите параметры инициализации (добавлены `modules` и `module_configs`)
3. Формат результатов остается совместимым

## 📝 Примечания

- Модули обрабатываются в порядке загрузки
- Результаты модулей автоматически добавляются в `shared_data` для зависимостей
- Модули, которые не могут обработать данные (недостаточно зависимостей), пропускаются
- Ошибки в модулях логируются, но не прерывают обработку других модулей

## 🎯 Применение

- Детальный анализ лиц в видео
- Извлечение признаков для ML-моделей
- Анализ качества и освещения
- Оценка профессионального уровня съёмки
- **Lip reading** для предсказания speech activity
- **3D face reconstruction** для анализа структуры лица
- **Fatigue detection** для мониторинга усталости (автоиндустрия, безопасность)

## ✨ Новые возможности

### Lip Reading Features
Модуль `LipReadingModule` предоставляет детальные фичи для анализа движения губ:
- Геометрические характеристики рта (ширина, высота, площадь)
- Контурные фичи (компактность, форма)
- Phoneme-like features для различения разных звуков
- Временные паттерны (скорость, цикличность)
- Вероятность речевой активности

### 3D Face Reconstruction
Модуль `Face3DModule` реализует упрощенную версию 3DMM (3D Morphable Model):
- 100-300 параметров 3D mesh
- Identity и expression векторы
- Параметризация формы лица через PCA
- Не требует отдельной установки DECA/EMOCA моделей
- Работает на основе MediaPipe landmarks

### Улучшенный Fatigue Score
Модуль `ProfessionalModule` теперь включает полный анализ усталости:
- **Eye-based indicators**: закрытость глаз, асимметрия, аномалии моргания
- **Pose-based indicators**: наклон головы вниз, нестабильность позы
- **Motion-based indicators**: медленные движения, низкая активность
- **Temporal patterns**: анализ трендов усталости во времени
- Детализированный breakdown по компонентам
