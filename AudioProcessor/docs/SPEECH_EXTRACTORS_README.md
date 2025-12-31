# 🎤 Экстракторы речи для AudioProcessor

## Обзор

Добавлены три новых экстрактора для анализа речи:

1. **ASRExtractor** - автоматическое распознавание речи с помощью Whisper
2. **SpeakerDiarizationExtractor** - диаризация спикеров с помощью Resemblyzer  
3. **SpeechAnalysisExtractor** - комбинированный анализ с сопоставлением ASR и диаризации

## 🔧 Установка зависимостей

```bash
pip install openai-whisper>=20231117
pip install resemblyzer>=0.1.1
pip install scikit-learn>=1.3.0
```

## 📁 Структура файлов

```
src/extractors/
├── asr_extractor.py                    # ASR экстрактор
├── speaker_diarization_extractor.py    # Диаризация спикеров
├── speech_analysis_extractor.py        # Комбинированный анализ
└── __init__.py                         # Обновлен с импортами
```

## 🚀 Использование

### 1. ASR экстрактор

```python
from src.extractors import ASRExtractor

# Инициализация
asr = ASRExtractor(
    device="auto",           # "cuda", "cpu", "auto"
    model_size="small",      # "tiny", "base", "small", "medium", "large"
    language=None,           # None для автоопределения, "ru", "en", etc.
    task="transcribe"        # "transcribe" или "translate"
)

# Запуск
result = asr.run(audio_path, tmp_dir)

if result.success:
    print(f"Транскрипт: {result.payload['transcription']}")
    print(f"Язык: {result.payload['language']}")
    print(f"Сегменты: {len(result.payload['segments'])}")
```

### 2. Диаризация спикеров

```python
from src.extractors import SpeakerDiarizationExtractor

# Инициализация
diarization = SpeakerDiarizationExtractor(
    device="auto",
    segment_duration=2.0,    # Длительность сегмента в секундах
    min_speakers=1,          # Минимальное количество спикеров
    max_speakers=10,         # Максимальное количество спикеров
    sample_rate=16000
)

# Запуск
result = diarization.run(audio_path, tmp_dir)

if result.success:
    print(f"Спикеров: {result.payload['speaker_count']}")
    print(f"Сегментов: {len(result.payload['speaker_segments'])}")
```

### 3. Комбинированный анализ

```python
from src.extractors import SpeechAnalysisExtractor

# Инициализация
speech_analysis = SpeechAnalysisExtractor(
    device="auto",
    asr_model_size="small",
    asr_language=None,
    diarization_segment_duration=2.0
)

# Запуск
result = speech_analysis.run(audio_path, tmp_dir)

if result.success:
    aligned = result.payload['aligned_speech']
    print(f"Выровненных сегментов: {aligned['total_segments']}")
    print(f"Спикеров: {aligned['total_speakers']}")
    
    # Показываем первые сегменты с присвоенными спикерами
    for segment in aligned['aligned_segments'][:5]:
        print(f"[{segment['start']:.2f}-{segment['end']:.2f}] Speaker {segment['speaker_id']}: {segment['text']}")
```

## 🧪 Тестирование

Запустите тестовый скрипт:

```bash
cd /home/ilya/Рабочий\ стол/DataProcessor/AudioProcessor
python test_speech_extractors.py
```

## 📊 Результаты

### ASR экстрактор возвращает:
- `transcription` - полный текст транскрипции
- `segments` - массив сегментов с временными метками
- `language` - определенный язык
- `language_probability` - уверенность в языке

### Диаризационный экстрактор возвращает:
- `speaker_segments` - сегменты с метками спикеров
- `speaker_count` - количество спикеров
- `speaker_embeddings` - эмбеддинги спикеров

### Комбинированный экстрактор возвращает:
- `aligned_speech` - выровненные сегменты (ASR + диаризация)
- `statistics` - статистики по речи и спикерам
- `asr_result` - полный результат ASR
- `diarization_result` - полный результат диаризации

## ⚙️ Интеграция в основную систему

Экстракторы автоматически добавлены в `MainProcessor`:

```python
# Доступные экстракторы
extractors = ["asr", "speaker_diarization", "speech_analysis"]

# Запуск через API
POST /api/v1/process
{
    "video_path": "/path/to/video.mp4",
    "output_dir": "/path/to/output",
    "extractor_names": ["asr", "speaker_diarization", "speech_analysis"]
}
```

## 🔧 Настройки

### GPU поддержка
- ASR: предпочитает GPU, требует ~1GB памяти
- Диаризация: предпочитает GPU, требует ~500MB памяти  
- Комбинированный: предпочитает GPU, требует ~1.5GB памяти

### Производительность
- ASR: ~5-10 секунд на минуту аудио (GPU)
- Диаризация: ~2-5 секунд на минуту аудио (GPU)
- Комбинированный: ~8-15 секунд на минуту аудио (GPU)

## 🐛 Устранение неполадок

1. **Ошибка импорта whisper**: `pip install openai-whisper`
2. **Ошибка импорта resemblyzer**: `pip install resemblyzer`
3. **CUDA недоступна**: экстракторы автоматически переключатся на CPU
4. **Недостаточно памяти GPU**: уменьшите `gpu_memory_limit` в настройках

## 📝 Примеры использования

См. файл `test_speech_extractors.py` для полных примеров использования всех экстракторов.
