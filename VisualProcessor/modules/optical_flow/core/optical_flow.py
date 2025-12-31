"""
optical_flow_pipeline.py - Production пайплайн обработки видео с RAFT

Все TODO выполнены:
    1. ✅ Интеграция с внешними зависимостями через BaseModule (если потребуется)
    2. ✅ Модель RAFT используется напрямую (нет в core/model_process)
    3. ✅ Интеграция с BaseModule через класс OpticalFlowModule
    4. ✅ Единый формат вывода для сохранения в npz
"""

import os
import sys
import torch
import torchvision.transforms as T
import torchvision.models.optical_flow as models
import numpy as np
import cv2
import json
from typing import Dict, Tuple, Optional, Any, List
from datetime import datetime

from .config import FlowPipelineConfig

# Добавляем путь для импорта BaseModule
_MODULE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _MODULE_PATH not in sys.path:
    sys.path.append(_MODULE_PATH)

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

name = "OpticalFlowProcessor"

from utils.logger import get_logger
logger = get_logger(name)

class OpticalFlowProcessor:
    """Основной класс для обработки оптического потока."""
    
    def __init__(self, config = None):
        self.config = config
        self.model = None
        self.device = None
        
    def _initialize_model(self):
        """Инициализация модели RAFT."""
        logger.info(f"Инициализация модели RAFT {self.config.model_type}")
        
        try:
            if self.config.model_type == "large":
                self.model = models.raft_large(
                    weights=models.Raft_Large_Weights.DEFAULT,
                    progress=True
                ).to(self.config.device)
            else:
                self.model = models.raft_small(
                    weights=models.Raft_Small_Weights.DEFAULT,
                    progress=True
                ).to(self.config.device)
            
            self.model.eval()
            logger.info(f"Модель инициализирована на {self.config.device}")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации модели: {e}")
            raise
    
    @staticmethod
    def resize_frame(frame_tensor: torch.Tensor, max_dimension: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Ресайз кадра с сохранением соотношения сторон."""
        if frame_tensor.dtype != torch.float32:
            frame_tensor = frame_tensor.float()
        
        _, H, W = frame_tensor.shape
        
        if max(H, W) <= max_dimension:
            return frame_tensor, (H, W)
        
        if H > W:
            new_H = max_dimension
            new_W = int(W * (max_dimension / H))
        else:
            new_W = max_dimension
            new_H = int(H * (max_dimension / W))
        
        resized = torch.nn.functional.interpolate(
            frame_tensor.unsqueeze(0),
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return resized, (H, W)
    
    @staticmethod
    def preprocess_frame(frame_tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Предобработка кадра для RAFT."""
        transforms = T.Compose([
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        frame = transforms(frame_tensor)
        
        _, H, W = frame.shape
        pad_h = (8 - H % 8) % 8
        pad_w = (8 - W % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            frame = torch.nn.functional.pad(frame, (0, pad_w, 0, pad_h), 
                                           mode='constant', value=0)
        
        return frame, (H, W)
    
    @staticmethod
    def resize_flow(flow_tensor: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Ресайз тензора оптического потока."""
        if flow_tensor.dim() == 4:
            flow_tensor = flow_tensor.squeeze(0)
        
        h, w = flow_tensor.shape[1], flow_tensor.shape[2]
        new_h, new_w = target_size
        
        if h == new_h and w == new_w:
            return flow_tensor
        
        flow_resized = torch.zeros(2, new_h, new_w, 
                                  device=flow_tensor.device, 
                                  dtype=flow_tensor.dtype)
        
        for i in range(2):
            flow_resized[i:i+1] = torch.nn.functional.interpolate(
                flow_tensor[i:i+1].unsqueeze(0),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Масштабирование значений потока
        scale_h = new_h / h
        scale_w = new_w / w
        flow_resized[0] *= scale_w
        flow_resized[1] *= scale_h
        
        return flow_resized
    
    @staticmethod
    def flow_to_color_map(flow_tensor: torch.Tensor, max_flow: float = 50.0) -> np.ndarray:
        """Конвертация потока в цветовую карту."""
        flow_np = flow_tensor.permute(1, 2, 0).cpu().numpy()
        h, w = flow_np.shape[:2]
        
        magnitude, angle = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1], 
                                          angleInDegrees=True)
        
        magnitude_normalized = np.clip(magnitude / max_flow, 0, 1)
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = angle / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(magnitude_normalized, None, 0, 255, 
                                   cv2.NORM_MINMAX)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    @staticmethod
    def _compute_fb_metrics(flow_fwd: torch.Tensor,
                            flow_bwd: torch.Tensor,
                            fb_thresh: float = 1.0,
                            occlusion_thresh: float = 1.5) -> Dict[str, float]:
        """Оценивает forward-backward согласованность и доверие."""
        if flow_fwd.dim() == 4:
            flow_fwd = flow_fwd.squeeze(0)
        if flow_bwd.dim() == 4:
            flow_bwd = flow_bwd.squeeze(0)

        device = flow_fwd.device
        _, h, w = flow_fwd.shape

        # Координатная сетка
        y, x = torch.meshgrid(
            torch.arange(h, device=device, dtype=flow_fwd.dtype),
            torch.arange(w, device=device, dtype=flow_fwd.dtype),
            indexing='ij'
        )
        x2 = x + flow_fwd[0]
        y2 = y + flow_fwd[1]

        # Нормализация координат для grid_sample
        x2_norm = 2.0 * (x2 / max(w - 1, 1)) - 1.0
        y2_norm = 2.0 * (y2 / max(h - 1, 1)) - 1.0
        grid = torch.stack((x2_norm, y2_norm), dim=-1).unsqueeze(0)

        flow_bwd_warp = torch.nn.functional.grid_sample(
            flow_bwd.unsqueeze(0),
            grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        ).squeeze(0)

        fb_sum = flow_fwd + flow_bwd_warp
        fb_error = torch.sqrt(torch.sum(fb_sum ** 2, dim=0))

        valid_mask = (x2 >= 0) & (x2 <= w - 1) & (y2 >= 0) & (y2 <= h - 1)
        valid_ratio = float(valid_mask.float().mean().item())
        fb_error_valid = fb_error[valid_mask] if valid_mask.any() else fb_error

        fb_error_mean = float(fb_error_valid.mean().item())
        fb_error_fraction = float((fb_error > fb_thresh).float().mean().item())
        occlusion_fraction = float((fb_error > occlusion_thresh).float().mean().item())
        flow_confidence_mean = float((1.0 / (1.0 + fb_error_valid)).mean().item())

        return {
            'fb_error_mean': fb_error_mean,
            'fb_error_fraction': fb_error_fraction,
            'occlusion_fraction': occlusion_fraction,
            'flow_confidence_mean': flow_confidence_mean,
            'valid_ratio': valid_ratio
        }
    
    def process_video(self, frame_manager, frame_indices) -> Dict[str, Any]:
        """
        Основной метод обработки видео.
        
        Args:
            video_path: Путь к видеофайлу
            
        Returns:
            Словарь с результатами обработки
        """
        import os

        # Инициализация модели
        if self.model is None:
            self._initialize_model()

        flow_dir = f"{self.config.output_dir}/flow"
        overlay_dir = f"{self.config.output_dir}/overlay" if self.config.save_overlay else None
        quality_dir = f"{self.config.output_dir}/flow/quality"
        
        os.makedirs(flow_dir, exist_ok=True)
        if overlay_dir:
            os.makedirs(overlay_dir, exist_ok=True)
        os.makedirs(quality_dir, exist_ok=True)
        
        # Получение свойств видео
        fps = frame_manager.fps
        total_frames = frame_manager.total_frames
        width = frame_manager.width
        height = frame_manager.height
        
        # Основной цикл обработки
        frame_buffer = []
        processed_pairs = 0
        flow_data = []

        frame_step_est = frame_indices[1] - frame_indices[0] if len(frame_indices) > 1 else 1

        for frame_idx in frame_indices:
            frame = frame_manager.get(frame_idx)
            
            # Конвертация BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(self.config.device)
            
            # Ресайз
            frame_resized, orig_size = self.resize_frame(frame_tensor, self.config.max_dimension)
            
            frame_buffer.append({
                'tensor_resized': frame_resized,
                'orig_size': orig_size,
                'original_idx': frame_idx
            })
            
            # Обработка пары кадров
            if len(frame_buffer) == 2:
                frame1 = frame_buffer[0]
                frame2 = frame_buffer[-1]
                
                # Предобработка
                frame1_processed, _ = self.preprocess_frame(frame1['tensor_resized'])
                frame2_processed, _ = self.preprocess_frame(frame2['tensor_resized'])
                
                # Расчет оптического потока
                with torch.no_grad():
                    list_of_flows = self.model(
                        frame1_processed.unsqueeze(0),
                        frame2_processed.unsqueeze(0)
                    )
                    flow_tensor = list_of_flows[-1].squeeze(0)

                    flow_tensor_bwd = None
                    if self.config.enable_forward_backward:
                        list_of_flows_bwd = self.model(
                            frame2_processed.unsqueeze(0),
                            frame1_processed.unsqueeze(0)
                        )
                        flow_tensor_bwd = list_of_flows_bwd[-1].squeeze(0)
                
                # Масштабирование к оригинальному размеру
                flow_resized = self.resize_flow(flow_tensor, frame1['orig_size'])
                flow_resized_bwd = self.resize_flow(flow_tensor_bwd, frame1['orig_size']) if flow_tensor_bwd is not None else None
                
                # Сохранение тензора потока
                if self.config.save_flow_tensors:
                    flow_filename = f"flow_{frame1['original_idx']:06d}.pt"
                    flow_path = f"{flow_dir}/{flow_filename}"
                    torch.save(flow_resized.cpu(), flow_path)
                    if self.config.save_backward_flow and flow_resized_bwd is not None:
                        bwd_filename = f"flow_bwd_{frame1['original_idx']:06d}.pt"
                        torch.save(flow_resized_bwd.cpu(), f"{flow_dir}/{bwd_filename}")
                
                # Визуализация
                if self.config.save_overlay:
                    flow_rgb = self.flow_to_color_map(flow_resized)
                    frame_display = self._tensor_to_display(frame1['tensor_resized'])
                    
                    # Ресайз flow для overlay
                    if flow_rgb.shape[:2] != frame_display.shape[:2]:
                        flow_rgb = cv2.resize(
                            flow_rgb,
                            (frame_display.shape[1], frame_display.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )
                    
                    overlay = cv2.addWeighted(frame_display, 0.6, flow_rgb, 0.4, 0)
                    overlay_path = f"{overlay_dir}/overlay_{frame1['original_idx']:06d}.png"
                    cv2.imwrite(str(overlay_path), overlay)
                
                # Сбор данных для статистик
                flow_data.append({
                    'frame_idx': frame1['original_idx'],
                    'flow_tensor': flow_resized.cpu(),
                    'orig_size': frame1['orig_size']
                })

                # Качество/консистентность
                if flow_resized_bwd is not None:
                    quality = self._compute_fb_metrics(
                        flow_resized,
                        flow_resized_bwd,
                        fb_thresh=self.config.fb_error_threshold,
                        occlusion_thresh=self.config.occlusion_error_threshold
                    )
                    quality_path = f"{quality_dir}/quality_{frame1['original_idx']:06d}.json"
                    with open(quality_path, 'w', encoding='utf-8') as f:
                        json.dump(quality, f, ensure_ascii=False, indent=2)
                
                # Обновление буфера
                frame_buffer = [frame_buffer[-1]]
                processed_pairs += 1
            
            frame_idx += 1
            
            # Периодическая очистка кэша CUDA
            if frame_idx % 50 == 0 and self.config.device == "cuda":
                torch.cuda.empty_cache()
        
        # Создание метаданных
        metadata = self._create_metadata(
            output_dir=self.config.output_dir,
            fps=fps,
            total_frames=total_frames,
            processed_frames=processed_pairs,
            original_resolution=(width, height),
            processed_resolution=self._get_processed_size(height, width),
            frame_skip=frame_step_est
        )
        
        logger.info(f"Обработка завершена. Обработано пар: {processed_pairs}")
        
        return {
            'flow_dir': str(flow_dir),
            'overlay_dir': str(overlay_dir) if overlay_dir else None,
            'metadata': metadata,
            'flow_data': flow_data,
            'processed_pairs': processed_pairs
        }
    
    def _create_metadata(self, output_dir, fps: float, total_frames: int, processed_frames: int,
                        original_resolution: Tuple[int, int],
                        processed_resolution: Tuple[int, int],
                        frame_skip: int = 1) -> Dict[str, Any]:
        """Создание метаданных видео."""
        
        metadata = {
            'processing_date': datetime.now().isoformat(),
            
            'processing_parameters': {
                'model': self.config.model_type,
                'max_dimension': self.config.max_dimension,
                'device': self.config.device,
                'pipeline_version': '2.0.0',
                'frame_skip': frame_skip
            },
            
            'video_properties': {
                'original_resolution': original_resolution,
                'processed_resolution': processed_resolution,
                'total_frames': total_frames,
                'processed_frames': processed_frames,
                'fps': fps,
                'duration_seconds': total_frames / fps if fps > 0 else 0
            },
            
            'output_structure': {
                'flow_format': 'torch_tensor',
                'flow_extension': '.pt',
                'flow_naming': 'flow_{frame_idx:06d}.pt',
                'overlay_format': 'png' if self.config.save_overlay else None,
                'overlay_naming': 'overlay_{frame_idx:06d}.png' if self.config.save_overlay else None
            }
        }
        
        # Сохранение метаданных
        metadata_path = f"{output_dir}/metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata
    
    @staticmethod
    def _tensor_to_display(tensor: torch.Tensor) -> np.ndarray:
        """Конвертация тензора в numpy для отображения."""
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        
        tensor = tensor.squeeze(0)
        
        if tensor.max() <= 1.0:
            tensor = (tensor * 255).clamp(0, 255)
        
        tensor = tensor.to(torch.uint8)
        frame_np = tensor.permute(1, 2, 0).cpu().numpy()
        
        if frame_np.shape[2] == 3:
            frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        
        return frame_np
    
    @staticmethod
    def _get_processed_size(height: int, width: int, max_dim: int = 512) -> Tuple[int, int]:
        """Вычисление размера после ресайза."""
        if max(height, width) <= max_dim:
            return (height, width)
        
        if height > width:
            new_h = max_dim
            new_w = int(width * (max_dim / height))
        else:
            new_w = max_dim
            new_h = int(height * (max_dim / width))
        
        return (new_h, new_w)


class OpticalFlowModule(BaseModule):
    """
    Модуль для анализа оптического потока в видео.
    
    Наследуется от BaseModule для интеграции с системой зависимостей и единым форматом вывода.
    Использует OpticalFlowProcessor и FlowStatisticsAnalyzer для обработки видео.
    """
    
    def __init__(
        self,
        rs_path: Optional[str] = None,
        model: str = "small",
        max_dim: int = 256,
        use_overlay: bool = False,
        run_stats: bool = False,
        device: Optional[str] = None,
        enable_forward_backward: bool = False,
        save_backward_flow: bool = False,
        fb_error_threshold: Optional[float] = None,
        occlusion_error_threshold: Optional[float] = None,
        stats_config: Optional[Any] = None,
        **kwargs: Any
    ):
        """
        Инициализация OpticalFlowModule.
        
        Args:
            rs_path: Путь к хранилищу результатов
            model: Модель RAFT ('small' или 'large')
            max_dim: Максимальный размер стороны
            use_overlay: Сохранять overlay визуализацию
            run_stats: Запустить статистический анализ
            device: Устройство для обработки (cuda/cpu)
            enable_forward_backward: Включить forward-backward проверку
            save_backward_flow: Сохранять обратный поток
            fb_error_threshold: Порог ошибки forward-backward
            occlusion_error_threshold: Порог ошибки окклюзии
            stats_config: Конфигурация для FlowStatisticsAnalyzer (опционально)
            **kwargs: Дополнительные параметры для BaseModule
        """
        super().__init__(rs_path=rs_path, **kwargs)
        
        self.run_stats = run_stats
        self.stats_config = stats_config
        
        # Настройка конфигурации flow pipeline
        flow_config = FlowPipelineConfig(
            model_type=model,
            max_dimension=max_dim,
            save_overlay=use_overlay,
            device=device,
            enable_forward_backward=enable_forward_backward,
            save_backward_flow=save_backward_flow,
            fb_error_threshold=fb_error_threshold,
            occlusion_error_threshold=occlusion_error_threshold,
        )
        
        # Устанавливаем output_dir
        if rs_path:
            flow_config.output_dir = os.path.join(rs_path, self.module_name)
        
        # Инициализируем процессор
        self.processor = OpticalFlowProcessor(flow_config)
    
    def required_dependencies(self) -> List[str]:
        """
        Возвращает список зависимостей модуля.
        
        В данный момент модуль не имеет зависимостей от других модулей.
        """
        return []
    
    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Основной метод обработки (интерфейс BaseModule).
        
        Args:
            frame_manager: Менеджер кадров
            frame_indices: Список индексов кадров для обработки
            config: Конфигурация модуля:
                - stats_config: конфигурация для FlowStatisticsAnalyzer (если run_stats=True)
                
        Returns:
            Словарь с результатами в формате для сохранения в npz:
            - features: агрегированные фичи оптического потока
            - flow_metadata: метаданные потока
            - stats_features: статистические фичи (если run_stats=True)
            - summary: метаданные обработки
        """
        try:
            # Обрабатываем видео через OpticalFlowProcessor
            flow_results = self.processor.process_video(
                frame_manager=frame_manager, frame_indices=frame_indices
            )
            
            # Статистический анализ (если включен)
            stats_results = None
            if self.run_stats and self.stats_config:
                from .flow_statistics import FlowStatisticsAnalyzer
                stats_analyzer = FlowStatisticsAnalyzer(self.stats_config)
                stats_results = stats_analyzer.analyze_video(
                    flow_results["flow_dir"], flow_results["metadata"]
                )
            
            # Преобразуем результаты в единый формат для npz
            formatted_result = self._format_results_for_npz(flow_results, stats_results)
            
            self.logger.info(
                f"OpticalFlowModule | Обработка завершена: "
                f"обработано {len(frame_indices)} кадров"
            )
            
            return formatted_result
            
        except Exception as e:
            self.logger.exception(f"OpticalFlowModule | Ошибка обработки: {e}")
            return self._empty_result()
    
    def _format_results_for_npz(
        self,
        flow_results: Dict[str, Any],
        stats_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Преобразует результаты OpticalFlowProcessor в формат для сохранения в npz.
        
        Args:
            flow_results: Результаты из processor.process_video()
            stats_results: Результаты из stats_analyzer.analyze_video() (опционально)
            
        Returns:
            Словарь в формате для npz
        """
        # Извлекаем основные данные
        flow_metadata = flow_results.get("metadata", {})
        flow_dir = flow_results.get("flow_dir", "")
        
        # Подготавливаем features (агрегированные фичи)
        features = {}
        
        # Добавляем метаданные потока
        if flow_metadata:
            for key, value in flow_metadata.items():
                if isinstance(value, (int, float, bool)):
                    features[f"flow_{key}"] = float(value) if isinstance(value, bool) else value
                elif isinstance(value, (list, tuple)):
                    try:
                        features[f"flow_{key}"] = np.asarray(value, dtype=np.float32)
                    except Exception:
                        features[f"flow_{key}"] = np.asarray(value, dtype=object)
                elif isinstance(value, dict):
                    # Сохраняем словари как есть
                    features[f"flow_{key}"] = value
        
        # Добавляем статистические фичи (если есть)
        stats_features = {}
        if stats_results:
            # Извлекаем фичи из stats_results
            if isinstance(stats_results, dict):
                for key, value in stats_results.items():
                    if isinstance(value, (int, float, bool)):
                        stats_features[f"stats_{key}"] = float(value) if isinstance(value, bool) else value
                    elif isinstance(value, (list, tuple)):
                        try:
                            stats_features[f"stats_{key}"] = np.asarray(value, dtype=np.float32)
                        except Exception:
                            stats_features[f"stats_{key}"] = np.asarray(value, dtype=object)
                    elif isinstance(value, dict):
                        stats_features[f"stats_{key}"] = value
                    elif isinstance(value, np.ndarray):
                        stats_features[f"stats_{key}"] = value
        
        # Подготавливаем summary
        summary = {
            "flow_dir": flow_dir,
            "has_stats": stats_results is not None,
            "success": True,
        }
        
        # Формируем итоговый результат
        formatted_result = {
            "features": {**features, **stats_features},
            "flow_metadata": flow_metadata,
            "stats_features": stats_features if stats_features else {},
            "summary": summary,
        }
        
        return formatted_result
    
    def _empty_result(self) -> Dict[str, Any]:
        """Возвращает пустой результат в правильном формате."""
        return {
            "features": {},
            "flow_metadata": {},
            "stats_features": {},
            "summary": {
                "flow_dir": "",
                "has_stats": False,
                "success": False,
            },
        }