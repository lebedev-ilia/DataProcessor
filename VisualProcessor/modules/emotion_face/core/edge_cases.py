"""
Обработка граничных случаев (edge cases) для обработки видео.
"""
from typing import Dict, Any, Tuple
from core.exceptions import FrameSelectionError


def check_video_duration(total_frames: int, fps: float) -> Dict[str, Any]:
    """
    Проверяет длительность видео и возвращает информацию о граничных случаях.
    
    Args:
        total_frames: Общее количество кадров.
        fps: FPS видео.
    
    Returns:
        Словарь с информацией о граничных случаях.
    """
    duration_seconds = total_frames / fps if fps > 0 else 0
    
    edge_cases = {
        "is_very_short": duration_seconds < 1.0,
        "is_short": 1.0 <= duration_seconds < 5.0,
        "is_very_long": duration_seconds > 3600.0,  # > 1 час
        "is_long": 1800.0 < duration_seconds <= 3600.0,  # 30 мин - 1 час
        "duration_seconds": duration_seconds,
        "total_frames": total_frames,
        "fps": fps
    }
    
    return edge_cases


def handle_empty_video(total_frames: int) -> None:
    """
    Обрабатывает случай пустого видео.
    
    Args:
        total_frames: Общее количество кадров.
    
    Raises:
        VideoFileError: Если видео пустое.
    """
    if total_frames == 0:
        raise

def handle_no_faces(timeline: list, total_frames: int, min_faces_ratio: float = 0.01) -> Tuple[bool, str]:
    """
    Обрабатывает случай видео без лиц или с очень малым количеством лиц.
    
    Args:
        timeline: Список индексов кадров с лицами.
        total_frames: Общее количество кадров.
        min_faces_ratio: Минимальная доля кадров с лицами (по умолчанию 1%).
    
    Returns:
        Кортеж (is_critical, message):
        - is_critical: True, если критическая ситуация (нет лиц вообще)
        - message: Сообщение о ситуации
    """
    faces_count = len(timeline)
    faces_ratio = faces_count / total_frames if total_frames > 0 else 0
    
    if faces_count == 0:
        return True, "No faces detected in video"
    
    if faces_ratio < min_faces_ratio:
        return False, f"Very few faces detected ({faces_count}/{total_frames}, {faces_ratio*100:.2f}%)"
    
    return False, f"Faces detected: {faces_count}/{total_frames} ({faces_ratio*100:.2f}%)"


def handle_very_short_video(total_frames: int, fps: float, min_frames: int = 30) -> Dict[str, Any]:
    """
    Обрабатывает случай очень короткого видео.
    
    Args:
        total_frames: Общее количество кадров.
        fps: FPS видео.
        min_frames: Минимальное количество кадров для нормальной обработки.
    
    Returns:
        Словарь с рекомендациями по обработке.
    """
    duration = total_frames / fps if fps > 0 else 0
    
    if duration < 1.0 or total_frames < min_frames:
        return {
            "is_very_short": True,
            "recommendation": "use_all_frames",
            "target_length": min(total_frames, 256),
            "message": f"Video is very short ({duration:.2f}s, {total_frames} frames). Using all available frames."
        }
    
    return {
        "is_very_short": False,
        "recommendation": "normal_processing"
    }


def handle_very_long_video(total_frames: int, fps: float, max_duration_hours: float = 1.0) -> Dict[str, Any]:
    """
    Обрабатывает случай очень длинного видео.
    
    Args:
        total_frames: Общее количество кадров.
        fps: FPS видео.
        max_duration_hours: Максимальная рекомендуемая длительность в часах.
    
    Returns:
        Словарь с рекомендациями по обработке.
    """
    duration_hours = (total_frames / fps) / 3600 if fps > 0 else 0
    
    if duration_hours > max_duration_hours:
        return {
            "is_very_long": True,
            "duration_hours": duration_hours,
            "recommendation": "increase_scan_stride",
            "suggested_scan_stride_multiplier": min(5.0, duration_hours / max_duration_hours),
            "message": f"Video is very long ({duration_hours:.2f} hours). Consider increasing scan stride."
        }
    
    return {
        "is_very_long": False,
        "recommendation": "normal_processing"
    }



def _collect_warnings(
    duration_info: Dict[str, Any],
    short_video_info: Dict[str, Any],
    long_video_info: Dict[str, Any],
    faces_count: int,
    total_frames: int
) -> list:
    """Собирает предупреждения о граничных случаях."""
    warnings = []
    
    if duration_info.get("is_very_short"):
        warnings.append("Video is very short (< 1 second)")
    
    if duration_info.get("is_very_long"):
        warnings.append(f"Video is very long ({duration_info.get('duration_seconds', 0)/3600:.2f} hours)")
    
    if short_video_info.get("is_very_short"):
        warnings.append(short_video_info.get("message", ""))
    
    if long_video_info.get("is_very_long"):
        warnings.append(long_video_info.get("message", ""))
    
    faces_ratio = faces_count / total_frames if total_frames > 0 else 0
    if faces_ratio < 0.05:  # Меньше 5% кадров с лицами
        warnings.append(f"Very few faces detected ({faces_ratio*100:.1f}%)")
    
    return warnings

