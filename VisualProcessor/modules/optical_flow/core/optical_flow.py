"""
optical_flow_pipeline.py - Production пайплайн обработки видео с RAFT
"""

import torch
import torchvision.transforms as T
import torchvision.models.optical_flow as models
import numpy as np
import cv2
import json
from typing import Dict, Tuple, Optional, Any
from datetime import datetime

from .config import FlowPipelineConfig

name = "OpticalFlowProcessor"

from utils.logger import get_logger
logger = get_logger(name)

class OpticalFlowProcessor:
    """Основной класс для обработки оптического потока."""
    
    def __init__(self, config: Optional[FlowPipelineConfig] = None):
        self.config = config or FlowPipelineConfig()
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
        
        os.makedirs(flow_dir, exist_ok=True)
        if overlay_dir:
            os.makedirs(overlay_dir, exist_ok=True)
        
        # Получение свойств видео
        fps = frame_manager.fps
        total_frames = frame_manager.total_frames
        width = frame_manager.width
        height = frame_manager.height
        
        # Основной цикл обработки
        frame_buffer = []
        processed_pairs = 0
        flow_data = []

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
                
                # Масштабирование к оригинальному размеру
                flow_resized = self.resize_flow(flow_tensor, frame1['orig_size'])
                
                # Сохранение тензора потока
                if self.config.save_flow_tensors:
                    flow_filename = f"flow_{frame1['original_idx']:06d}.pt"
                    flow_path = f"{flow_dir}/{flow_filename}"
                    torch.save(flow_resized.cpu(), flow_path)
                
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
            processed_resolution=self._get_processed_size(height, width)
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
                        processed_resolution: Tuple[int, int]) -> Dict[str, Any]:
        """Создание метаданных видео."""
        
        metadata = {
            'processing_date': datetime.now().isoformat(),
            
            'processing_parameters': {
                'model': self.config.model_type,
                'max_dimension': self.config.max_dimension,
                'device': self.config.device,
                'pipeline_version': '2.0.0'
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