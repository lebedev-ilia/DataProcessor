"""
Экстракторы для AudioProcessor.

Содержит различные экстракторы:
- VideoAudioExtractor: извлечение аудио из видео
- AudioExtractors: различные аудио экстракторы признаков
- SpeechExtractors: экстракторы для анализа речи
"""

# Импорты экстракторов речи
from .asr_extractor import ASRExtractor
from .speaker_diarization_extractor import SpeakerDiarizationExtractor
from .speech_analysis_extractor import SpeechAnalysisExtractor
from .emotion_diarization_extractor import EmotionDiarizationExtractor

# Экспорт всех экстракторов
__all__ = [
    'ASRExtractor',
    'SpeakerDiarizationExtractor', 
    'SpeechAnalysisExtractor',
    'EmotionDiarizationExtractor'
]
