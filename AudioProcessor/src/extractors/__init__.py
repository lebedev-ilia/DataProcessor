"""
Экстракторы для AudioProcessor.

Содержит различные экстракторы:
- VideoAudioExtractor: извлечение аудио из видео
- AudioExtractors: различные аудио экстракторы признаков
- SpeechExtractors: экстракторы для анализа речи

Важно: Импорты сделаны опциональными, чтобы не блокировать Tier-0 extractors (clap/tempo/loudness)
при отсутствии тяжелых зависимостей (whisper и т.д.). MainProcessor использует ленивую загрузку через __import__.
"""

# Опциональные импорты экстракторов речи (ленивая загрузка, чтобы не блокировать Tier-0)
# MainProcessor использует __import__ внутри фабрик, так что эти импорты не обязательны
__all__ = []

# Попытка импорта (опционально, для обратной совместимости, если где-то есть прямые импорты)
try:
    from .asr_extractor import ASRExtractor
    __all__.append('ASRExtractor')
except ImportError:
    pass

try:
    from .speaker_diarization_extractor import SpeakerDiarizationExtractor
    __all__.append('SpeakerDiarizationExtractor')
except ImportError:
    pass

try:
    from .speech_analysis_extractor import SpeechAnalysisExtractor
    __all__.append('SpeechAnalysisExtractor')
except ImportError:
    pass

try:
    from .emotion_diarization_extractor import EmotionDiarizationExtractor
    __all__.append('EmotionDiarizationExtractor')
except ImportError:
    pass
