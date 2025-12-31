"""
Экстрактор для извлечения аудио из видео файлов.
"""
import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class VideoAudioExtractor(BaseExtractor):
    """Экстрактор для извлечения аудио из видео файлов."""
    
    name = "video_audio_extractor"
    version = "1.0.0"
    description = "Извлечение аудио дорожки из видео файлов"
    category = "video"
    dependencies = ["ffmpeg"]
    estimated_duration = 5.0
    
    # Не требует GPU, но может использовать для обработки извлеченного аудио
    gpu_required = False
    gpu_preferred = False
    
    # Поддерживаемые форматы видео
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv'}
    
    def __init__(self, device: str = "cpu", sample_rate: int = 22050):
        """
        Инициализация экстрактора видео.
        
        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации для извлеченного аудио
        """
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение аудио из видео файла.
        
        Args:
            input_uri: Путь к видео файлу
            tmp_path: Временная директория для обработки
            
        Returns:
            ExtractorResult с информацией об извлеченном аудио
        """
        start_time = time.time()
        
        try:
            # Валидация входного файла
            if not self._validate_input(input_uri):
                return self._create_result(
                    success=False,
                    error="Некорректный входной файл",
                    processing_time=time.time() - start_time
                )
            
            # Проверяем, что это видео файл
            if not self._is_video_file(input_uri):
                return self._create_result(
                    success=False,
                    error="Файл не является видео",
                    processing_time=time.time() - start_time
                )
            
            self._log_extraction_start(input_uri)
            
            # Создаем имя для выходного аудио файла
            video_path = Path(input_uri)
            audio_filename = f"{video_path.stem}_extracted_audio.wav"
            audio_output_path = os.path.join(tmp_path, audio_filename)
            
            # Извлекаем аудио
            extracted_audio_path = self.audio_utils.extract_audio_from_video(
                video_path=input_uri,
                output_path=audio_output_path
            )
            
            # Получаем информацию об извлеченном аудио
            audio_info = self.audio_utils.get_audio_info(extracted_audio_path)
            
            # Получаем информацию о видео
            video_info = self._get_video_info(input_uri)
            
            processing_time = time.time() - start_time
            
            # Создаем результат
            payload = {
                "extracted_audio_path": extracted_audio_path,
                "audio_info": audio_info,
                "video_info": video_info,
                "extraction_successful": True,
                "sample_rate": self.sample_rate,
                "original_video_path": input_uri
            }
            
            self._log_extraction_success(input_uri, processing_time)
            
            return self._create_result(
                success=True,
                payload=payload,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Ошибка извлечения аудио: {str(e)}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            
            return self._create_result(
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    def _is_video_file(self, file_path: str) -> bool:
        """Проверка, является ли файл видео."""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.SUPPORTED_FORMATS
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Получение информации о видео файле.
        
        Args:
            video_path: Путь к видео файлу
            
        Returns:
            dict: Информация о видео
        """
        try:
            import subprocess
            
            # Команда ffprobe для получения информации о видео
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            info = json.loads(result.stdout)
            
            # Извлекаем нужную информацию
            video_info = {
                "file_size": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
                "duration": 0.0,
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "codec": "unknown",
                "bitrate": 0
            }
            
            # Парсим информацию из ffprobe
            if 'format' in info:
                format_info = info['format']
                video_info['duration'] = float(format_info.get('duration', 0))
                video_info['bitrate'] = int(format_info.get('bit_rate', 0))
            
            if 'streams' in info:
                for stream in info['streams']:
                    if stream.get('codec_type') == 'video':
                        video_info['width'] = int(stream.get('width', 0))
                        video_info['height'] = int(stream.get('height', 0))
                        video_info['fps'] = eval(stream.get('r_frame_rate', '0/1'))
                        video_info['codec'] = stream.get('codec_name', 'unknown')
                        break
            
            return video_info
            
        except Exception as e:
            self.logger.warning(f"Не удалось получить информацию о видео: {e}")
            return {
                "file_size": os.path.getsize(video_path) if os.path.exists(video_path) else 0,
                "duration": 0.0,
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "codec": "unknown",
                "bitrate": 0
            }
    
    def _validate_input(self, input_uri: str) -> bool:
        """Валидация входного файла."""
        if not super()._validate_input(input_uri):
            return False
        
        # Проверяем существование файла
        if not os.path.exists(input_uri):
            self.logger.error(f"Файл не существует: {input_uri}")
            return False
        
        # Проверяем, что это видео файл
        if not self._is_video_file(input_uri):
            self.logger.error(f"Файл не является поддерживаемым видео форматом: {input_uri}")
            return False
        
        return True
    
    def get_supported_formats(self) -> set:
        """Получение списка поддерживаемых форматов."""
        return self.SUPPORTED_FORMATS.copy()
    
    def check_dependencies(self) -> bool:
        """Проверка наличия зависимостей."""
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
