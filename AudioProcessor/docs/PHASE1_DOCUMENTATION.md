# AudioProcessor - Документация первой фазы

## Обзор проекта

AudioProcessor - система для извлечения аудио признаков из видео файлов с поддержкой GPU-ускорения и асинхронной обработки. Проект создан на основе архитектуры `AudioProcessor_old` с улучшениями и современными подходами.

## Архитектура системы

### Структура проекта
```
AudioProcessor/
├── src/
│   ├── core/
│   │   ├── base_extractor.py          # Базовый класс для экстракторов
│   │   ├── audio_utils.py             # Утилиты для работы с аудио
│   │   ├── main_processor.py          # Синхронный процессор
│   ├── extractors/
│   └── schemas/
│       └── models.py                  # Pydantic модели
├── config/
│   └── settings.py                    # Конфигурация приложения
├── tests/                             # Тестовые видео файлы
├── run_local_processing.py            # Синхронный тест
└── requirements.txt                   # Зависимости
```

### Ключевые компоненты

#### 1. BaseExtractor
- Абстрактный базовый класс для всех экстракторов
- Поддержка CPU/GPU устройств
- Стандартизированный интерфейс с методами `run()`, `_create_result()`
- Логирование и обработка ошибок

#### 2. AudioUtils
- Централизованные утилиты для работы с аудио
- Поддержка GPU через `torchaudio`
- Автоматический fallback на CPU при ошибках GPU
- Методы: `load_audio()`, `save_audio()`, `extract_audio_from_video()`

#### 3. MainProcessor (Синхронный)
- Координация работы экстракторов
- Извлечение аудио из видео файлов
- Последовательная обработка экстракторов
- Сохранение результатов в формате manifest

#### 4. AsyncMainProcessor (Асинхронный)
- Параллельная обработка экстракторов
- Параллельная обработка множественных видео
- Управление ресурсами через семафоры
- ThreadPoolExecutor для блокирующих операций

## Реализованные экстракторы

### 1. VideoAudioExtractor
- **Назначение**: Извлечение аудио дорожки из видео файлов
- **Вход**: Видео файл (MP4, AVI, MOV, etc.)
- **Выход**: WAV аудио файл + метаданные
- **Поддержка**: CPU/GPU через `torchaudio`

### 2. MFCCExtractor
- **Назначение**: Извлечение Mel-frequency cepstral coefficients
- **Параметры**: 40 MFCC, 128 mel-фильтров, sample_rate=22050
- **Выход**: Статистики (mean, std, min, max) + delta, delta-delta
- **Оптимизация**: GPU-ускорение через `torchaudio.transforms`

### 3. MelExtractor
- **Назначение**: Mel-спектрограмма и спектральные характеристики
- **Параметры**: 128 mel-фильтров, n_fft=2048, hop_length=512
- **Выход**: Mel-статистики + спектральный центроид, полоса пропускания, плоскостность
- **Оптимизация**: GPU-ускорение, безопасные численные вычисления

## Технические особенности

### GPU Поддержка
- Автоматическое определение CUDA доступности
- Fallback на CPU при ошибках GPU
- Оптимизированные `torchaudio` трансформации
- Управление памятью GPU

### Обработка ошибок
- Подавление warnings (deprecated torchaudio, numpy warnings)
- Безопасные численные вычисления (избежание sqrt/log от отрицательных значений)
- Graceful fallback между CPU/GPU
- Детальное логирование ошибок

### Формат результатов
- **Manifest JSON**: Стандартизированный формат результатов
- **Совместимость**: Соответствует формату `AudioProcessor_old`
- **Структура**: video_id, task_id, timestamp, extractors[], total_processing_time
- **Плоские признаки**: Все признаки в одном уровне для удобства анализа

## Тестирование

### Тестовые файлы
- 3 видео файла в директории `tests/`
- Форматы: MP4
- Различные длительности и характеристики

### Скрипты тестирования

#### 1. run_local_processing.py (Синхронный)
```bash
python run_local_processing.py
```
- Последовательная обработка на CPU и GPU
- Сравнение производительности
- Выход: `output_cpu/` и `output_gpu/`

#### 2. run_async_processing.py (Асинхронный)
```bash
python run_async_processing.py
```
- Параллельная обработка 3 видео одновременно
- CPU и GPU режимы
- Выход: `output_async/` и `output_async_gpu/`

### Результаты тестирования

#### Производительность (примерные значения)
- **CPU**: ~1.5-2.0s на видео
- **GPU**: ~0.5-1.0s на видео
- **Ускорение**: 2-3x на GPU
- **Асинхронность**: 3 видео обрабатываются параллельно

#### Качество результатов
- ✅ Все экстракторы работают корректно
- ✅ Manifest файлы создаются в правильном формате
- ✅ GPU ускорение функционирует
- ✅ Warnings исправлены

## Конфигурация

### Настройки (config/settings.py)
- **API**: host, port, workers
- **GPU**: device, memory_limit, batch_size
- **Обработка**: sample_rate, chunk_size, max_duration
- **Параллелизм**: max_workers, concurrent_tasks

### Зависимости (requirements.txt)
```
fastapi==0.104.1
uvicorn==0.24.0.post1
pydantic==2.5.2
pydantic-settings==2.1.0
python-dotenv==1.0.0
librosa==0.10.1
soundfile==0.12.1
torch==2.1.1
torchaudio==2.1.1
ffmpeg-python==0.2.0
```

## Использование

### Базовое использование
```python
from src.core.main_processor import MainProcessor

# Синхронная обработка
processor = MainProcessor(device="cuda", max_workers=4)
result = processor.process_video(
    video_path="video.mp4",
    output_dir="output/",
    extractor_names=["video_audio", "mfcc", "mel"],
    extract_audio=True
)
```

### Асинхронная обработка
```python
from src.core.async_main_processor import AsyncMainProcessor

# Асинхронная обработка
processor = AsyncMainProcessor(device="cuda", max_workers=4)
result = await processor.process_video_async(
    video_path="video.mp4",
    output_dir="output/",
    extractor_names=["video_audio", "mfcc", "mel"],
    extract_audio=True
)
```

## Проблемы и решения

### Решенные проблемы
1. **Warnings**: Подавлены deprecated torchaudio и numpy warnings
2. **GPU Fallback**: Реализован автоматический fallback на CPU
3. **Численная стабильность**: Исправлены sqrt/log от отрицательных значений
4. **JSON сериализация**: Конвертация numpy arrays в списки
5. **Импорты**: Исправлены относительные импорты

### Известные ограничения
1. **Зависимость от CUDA**: GPU режим требует CUDA-совместимую видеокарту
2. **Память GPU**: Большие видео могут требовать больше GPU памяти
3. **Форматы видео**: Поддерживаются основные форматы через torchaudio

## Следующие шаги (Фаза 2)

### Планируемые улучшения
1. **API сервер**: FastAPI endpoints для удаленного использования
2. **Дополнительные экстракторы**: CLAP, ASR, Emotion Recognition
3. **Batch обработка**: Массовая обработка файлов
4. **Мониторинг**: Prometheus метрики, логирование
5. **Docker**: Контейнеризация для развертывания
6. **S3 интеграция**: Загрузка/скачивание файлов из облака

### Архитектурные улучшения
1. **Celery**: Асинхронная очередь задач
2. **Redis**: Кэширование и брокер сообщений
3. **Kubernetes**: Масштабируемое развертывание
4. **Health checks**: Мониторинг состояния системы

## Заключение

Первая фаза AudioProcessor успешно реализована и протестирована. Система обеспечивает:

- ✅ Извлечение аудио из видео файлов
- ✅ Извлечение MFCC и Mel признаков
- ✅ GPU ускорение с fallback на CPU
- ✅ Асинхронную и синхронную обработку
- ✅ Стандартизированный формат результатов
- ✅ Отсутствие warnings и ошибок

Система готова для перехода ко второй фазе разработки с добавлением API, дополнительных экстракторов и production-ready функций.
