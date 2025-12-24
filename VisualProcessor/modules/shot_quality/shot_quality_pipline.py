import cv2
import torch
import clip
import numpy as np
from PIL import Image
from scipy.stats import entropy
from typing import Dict, Optional
try:
    from aesthetic_predictor import AestheticPredictor
except ImportError:
    # Fallback: simple aesthetic predictor using CLIP
    class AestheticPredictor:
        def __init__(self, model_name):
            self.model_name = model_name
        def predict(self, pil_image):
            # Simple heuristic based on image properties
            img_array = np.array(pil_image)
            brightness = np.mean(img_array) / 255.0
            contrast = np.std(img_array) / 255.0
            # Combine brightness and contrast for aesthetic score
            return float(0.5 + 0.3 * brightness + 0.2 * contrast)

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    category=torch.serialization.SourceChangeWarning
)

name = "ShotQualityZeroShot"

import os
import sys
_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

from utils.logger import get_logger
logger = get_logger(name)

# -----------------------------
# HELPERS ДЛЯ ЧТЕНИЯ CORE ПРОВАЙДЕРОВ
# -----------------------------

def _load_core_clip_embeddings(rs_path: Optional[str], frame_index: int) -> Optional[np.ndarray]:
    """
    Загружает CLIP эмбеддинги из core_clip для конкретного кадра.
    Возвращает None, если core данные недоступны.
    """
    if not rs_path:
        return None
    
    core_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
    if not os.path.isfile(core_path):
        return None
    
    try:
        data = np.load(core_path)
        emb = data.get("frame_embeddings")
        if emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32)
        
        # Проверяем, что frame_index в пределах массива
        if frame_index < 0 or frame_index >= emb.shape[0]:
            return None
        
        return emb[frame_index]
    except Exception as e:
        logger.warning(f"{name} | _load_core_clip_embeddings | Error loading core data: {e}")
        return None

# -----------------------------
# DNN MODELS: DnCNN, CBDNet, MiDaS
# -----------------------------

def load_midas(device="cuda"):
    import torch

    model = torch.hub.load(
        "intel-isl/MiDaS",
        "MiDaS_small",
        pretrained=True,
        trust_repo=True,
        verbose=False
    ).to(device).eval()

    transforms = torch.hub.load(
        "intel-isl/MiDaS",
        "transforms",
        trust_repo=True,
        verbose=False
    )

    _midas = {
        "model": model,
        "transform": transforms.small_transform
    }

    return _midas["model"], _midas["transform"]

def load_dncnn(device: str = "cuda") -> torch.nn.Module:
    """
    Загружает модель DnCNN с предобученными весами.
    
    Поддерживает:
    - DataParallel (удаляет префикс 'module.')
    - Цветные RGB изображения
    - Корректное размещение на cpu/cuda

    Возвращает:
        torch.nn.Module: модель DnCNN в eval режиме
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.join(base_dir, "models", "DnCNN")

    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    try:
        from TrainingCodes.dncnn_pytorch.main_test import DnCNN
    except ImportError as e:
        raise ImportError(f"Не удалось импортировать класс DnCNN: {e}")

    sys.modules['__main__'].DnCNN = DnCNN

    weights_path = os.path.join(repo_dir, "TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth")

    with torch.serialization.safe_globals([DnCNN]):
        model = torch.load(weights_path, map_location=device, weights_only=False)

    model.to(device)
    model.eval()

    return model

def load_cbdnet(device="cuda"):
    import os
    import sys
    import subprocess
    import torch
    import torch.nn as nn

    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    repo_dir = os.path.join(models_dir, "CBDNet_pytorch")

    os.makedirs(models_dir, exist_ok=True)

    # Клонируем репозиторий
    if not os.path.exists(repo_dir):
        subprocess.run(
            ["git", "clone", "https://github.com/IDKiro/CBDNet-pytorch.git", repo_dir],
            check=True
        )

    # Добавляем в PYTHONPATH
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    checkpoint_path = os.path.join(models_dir, "checkpoint.pth.tar")
    if not os.path.exists(checkpoint_path):
        raise RuntimeError("CBDNet: checkpoint.pth.tar не найден")

    from model.cbdnet import Network # type: ignore

    model = Network()
    model.to(device)
    model = nn.DataParallel(model)

    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    return model


# ---------------------------------------------------------------
# SHARPNESS METRICS
# ---------------------------------------------------------------

def sharpness_tenengrad(frame):
    gx = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
    g = gx*gx + gy*gy
    return np.mean(g)

def _sharpness_laplacian(frame):
    """Вспомогательная Laplacian‑резкость (для secondary метрики)."""
    return cv2.Laplacian(frame, cv2.CV_64F).var()

def _sharpness_smd2(frame):
    """Вспомогательный SMD2 (для secondary метрики)."""
    diff1 = np.abs(frame[:, 1:] - frame[:, :-1])
    diff2 = np.abs(frame[1:, :] - frame[:-1, :])
    return np.mean(diff1) + np.mean(diff2)

# Motion blur CNN (simple heuristics)
def motion_blur_probability(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray)
    mag = np.log(np.abs(fft) + 1)
    blur = 1 - (np.mean(mag) / np.max(mag))
    return float(np.clip(blur, 0, 1))

# Edge clarity
def _edge_clarity(frame):
    edges = cv2.Canny(frame, 100, 200)
    return np.mean(edges) / 255

def _focus_gradient_score(gray):
    """Градиент‑основанный proxy фокуса (для secondary метрики)."""
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    return float(np.mean(gradient_magnitude))

def _spatial_frequency_mean(frame):
    """Нормализованная средняя пространственная частота (масштаб‑инвариантная)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h, w = gray.shape
    if h == 0 or w == 0:
        return 0.0
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    # Нормируем радиусы на диагональ, чтобы учесть разное разрешение
    diag = np.sqrt(h ** 2 + w ** 2)
    norm_dist = distances / (diag + 1e-8)
    weighted_freq = np.sum(magnitude * norm_dist) / (np.sum(magnitude) + 1e-10)
    return float(weighted_freq)

def sharpness_secondary(gray, frame_bgr):
    """
    Компактная secondary‑резкость, агрегирующая несколько метрик:
    Laplacian, SMD2, edge clarity и градиент‑фокус.
    """
    lap = _sharpness_laplacian(gray)
    smd = _sharpness_smd2(gray)
    edge = _edge_clarity(frame_bgr)
    grad = _focus_gradient_score(gray)

    # Простейшая нормализация/сжатие в один скор
    # (лог‑scale для lap/smd, среднее по всем компонентам)
    lap_n = np.log1p(lap)
    smd_n = np.log1p(smd)
    grad_n = np.log1p(grad)
    sec = (lap_n + smd_n + grad_n + edge) / 4.0
    return float(sec)

# ---------------------------------------------------------------
# NOISE METRICS (DnCNN, CBDNet)
# ---------------------------------------------------------------

def noise_estimation_dncnn(
    dncnn: torch.nn.Module,
    frame_rgb: np.ndarray,
    device: str = "cuda"
) -> float:
    """
    Оценка уровня шума через DnCNN (grayscale).
    """

    # RGB → Gray (perceptual)
    if frame_rgb.ndim == 3:
        gray = (
            0.299 * frame_rgb[..., 0] +
            0.587 * frame_rgb[..., 1] +
            0.114 * frame_rgb[..., 2]
        )
    else:
        gray = frame_rgb

    gray = gray.astype(np.float32) / 255.0

    tensor = (
        torch.from_numpy(gray)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        denoised = dncnn(tensor)
        noise_map = torch.abs(denoised - tensor)
        noise_level = float(noise_map.mean().item())

    return noise_level

def noise_estimation_cbdnet(cbdnet, frame_rgb, device="cuda"):
    import torch
    import numpy as np

    # [H, W, C] → [1, C, H, W]
    img = frame_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        noise_map, _ = cbdnet(tensor)

    # Noise statistics
    noise_mean = noise_map.mean().item()
    noise_std = noise_map.std().item()
    noise_energy = (noise_map ** 2).mean().item()

    return {
        "noise_mean": float(noise_mean),
        "noise_std": float(noise_std),
        "noise_energy": float(noise_energy)
    }

def noise_level_luma(gray):
    """Уровень шума в яркостном канале"""
    # Простой метод через вариацию локальных средних
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    return float(np.mean(diff) / 255.0)

def noise_level_chroma(frame):
    """Уровень шума в цветовых каналах"""
    # Конвертируем в YUV и анализируем U и V каналы
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    u = yuv[:, :, 1].astype(np.float32)
    v = yuv[:, :, 2].astype(np.float32)
    # Анализируем вариацию в цветовых каналах
    u_blur = cv2.GaussianBlur(u, (5, 5), 0)
    v_blur = cv2.GaussianBlur(v, (5, 5), 0)
    u_noise = np.mean(np.abs(u - u_blur))
    v_noise = np.mean(np.abs(v - v_blur))
    return float((u_noise + v_noise) / 2.0 / 255.0)

def iso_estimated_value(frame):
    """Оценка ISO на основе уровня шума"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    noise = noise_level_luma(gray)
    # Эвристика: больше шума = выше ISO
    # Нормализуем к примерному диапазону ISO 100-6400
    iso_est = 100 + noise * 6300
    return float(np.clip(iso_est, 100, 6400))

def grain_strength(frame):
    """Сила зерна (grain) в изображении"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    # Анализируем высокочастотные компоненты
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    grain = np.std(filtered)
    return float(np.clip(grain / 50.0, 0, 1))

def noise_spatial_entropy(gray):
    """Пространственная энтропия шума"""
    # Разбиваем на блоки и вычисляем энтропию вариации
    h, w = gray.shape
    block_size = 8
    entropies = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            hist = cv2.calcHist([block], [0], None, [256], [0, 256]).ravel()
            hist = hist / (hist.sum() + 1e-10)
            entropies.append(entropy(hist))
    return float(np.mean(entropies) if entropies else 0.0)

# ---------------------------------------------------------------
# EXPOSURE METRICS
# ---------------------------------------------------------------

def exposure_metrics(gray):
    """
    Робастные exposure‑метрики на основе percentiles, а не жёстких уровней 0–255.

    Используем p5/p95 как динамический диапазон:
    - underexposure_ratio: доля пикселей ниже p5
    - overexposure_ratio: доля пикселей выше p95
    - midtones_balance: доля пикселей в [p5, p95]
    """
    gray_f = gray.astype(np.float32)
    flat = gray_f.flatten()
    if flat.size == 0:
        return {
            "underexposure_ratio": 0.0,
            "overexposure_ratio": 0.0,
            "midtones_balance": 1.0,
            "exposure_histogram_skewness": 0.0,
            "highlight_recovery_potential": 1.0,
            "shadow_recovery_potential": 1.0,
        }

    p5 = np.percentile(flat, 5)
    p95 = np.percentile(flat, 95)

    # Безопасность на вырожденных случаях
    if p95 <= p5:
        p5 = max(0.0, p5 - 1.0)
        p95 = min(255.0, p95 + 1.0)

    total = float(flat.size)
    under = float(np.mean(flat < p5))
    over = float(np.mean(flat > p95))
    mid = float(1.0 - under - over)

    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-10)
    skew = entropy(hist)

    # Recovery potential: анализируем, можно ли восстановить детали
    highlight_recovery = 1.0 - over  # Меньше переэкспонирования = больше потенциал восстановления
    shadow_recovery = 1.0 - under    # Меньше недоэкспонирования = больше потенциал восстановления

    return {
        "underexposure_ratio": float(under),
        "overexposure_ratio": float(over),
        "midtones_balance": float(mid),
        "exposure_histogram_skewness": float(skew),
        "highlight_recovery_potential": float(highlight_recovery),
        "shadow_recovery_potential": float(shadow_recovery),
    }

# ---------------------------------------------------------------
# CONTRAST METRICS
# ---------------------------------------------------------------

def contrast_global(gray):
    return float(gray.std())

def contrast_local(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.mean(np.abs(lap)))

def contrast_dynamic_range(gray):
    """Динамический диапазон контраста"""
    min_val = np.min(gray)
    max_val = np.max(gray)
    if max_val > min_val:
        return float((max_val - min_val) / 255.0)
    return 0.0

def contrast_clarity_score(gray):
    """Оценка четкости контраста"""
    # Комбинация локального контраста и градиентов
    local_contrast = contrast_local(gray)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.mean(np.sqrt(gx**2 + gy**2))
    return float(np.clip((local_contrast + gradient_mag / 100.0) / 2.0, 0, 1))

def microcontrast(gray):
    """Микроконтраст - очень важный показатель качества линзы"""
    # Анализируем контраст на очень малых масштабах
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    micro = np.std(filtered)
    return float(np.clip(micro / 30.0, 0, 1))

# ---------------------------------------------------------------
# COLOR QUALITY
# ---------------------------------------------------------------

def white_balance_shift(frame):
    means = frame.mean(axis=(0, 1))
    return {
        "wb_r": float(means[2]),
        "wb_g": float(means[1]),
        "wb_b": float(means[0])
    }

def color_cast(frame):
    b, g, r = frame.mean(axis=(0, 1))
    if r > g and r > b:
        return "red"
    if g > r and g > b:
        return "green"
    if b > g and b > r:
        return "blue"
    return "neutral"

def skin_tone_accuracy_score(frame):
    """Оценка точности передачи тона кожи (упрощенная)"""
    # Ищем области, похожие на кожу (в диапазоне HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Типичный диапазон тона кожи в HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
    # Если есть кожа, оцениваем её цветовую точность
    if skin_ratio > 0.01:
        skin_pixels = frame[mask > 0]
        # Проверяем, насколько близок цвет к естественному тону кожи
        # Упрощенная эвристика: проверяем баланс RGB
        r_mean = np.mean(skin_pixels[:, 2])
        g_mean = np.mean(skin_pixels[:, 1])
        b_mean = np.mean(skin_pixels[:, 0])
        # Естественный тон кожи: R > G > B
        if r_mean > g_mean > b_mean:
            return float(1.0 - abs(r_mean - g_mean) / 255.0)
    return 0.5  # Нейтральная оценка, если кожи нет

def color_fidelity_index(frame):
    """Индекс цветовой точности"""
    # Анализируем цветовую вариацию и баланс
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Хорошая цветовая точность = равномерное распределение оттенков
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).ravel()
    h_hist = h_hist / (h_hist.sum() + 1e-10)
    color_entropy = entropy(h_hist)
    # Нормализуем к [0, 1]
    return float(np.clip(color_entropy / 7.0, 0, 1))

def color_noise_level(frame):
    """Уровень цветового шума"""
    # Анализируем вариацию цвета в локальных областях
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    # Локальная вариация в цветовых каналах
    a_blur = cv2.GaussianBlur(a, (5, 5), 0)
    b_blur = cv2.GaussianBlur(b, (5, 5), 0)
    a_noise = np.std(a - a_blur)
    b_noise = np.std(b - b_blur)
    return float((a_noise + b_noise) / 2.0 / 128.0)

def color_uniformity_score(frame):
    """Оценка равномерности цвета"""
    # Анализируем вариацию цвета по кадру
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_std = np.std(hsv[:, :, 0])
    s_std = np.std(hsv[:, :, 1])
    v_std = np.std(hsv[:, :, 2])
    # Меньше вариация = больше равномерность
    uniformity = 1.0 - np.clip((h_std + s_std + v_std) / (180 + 255 + 255), 0, 1)
    return float(uniformity)

# ---------------------------------------------------------------
# COMPRESSION ARTIFACTS
# ---------------------------------------------------------------

def blockiness(frame):
    diffs = np.abs(frame[:, 8:] - frame[:, :-8]).mean()
    return float(diffs)

def banding(gray):
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    diff = np.abs(gray - blurred).mean()
    return float(1 - diff / 255)

def ringing_artifacts_level(frame):
    """Уровень артефактов ringing (звон)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    # Ringing проявляется как осцилляции вокруг резких краев
    # Используем фильтр для детекции осцилляций
    kernel = np.array([[-1, -1, -1, -1, -1],
                       [-1,  2,  2,  2, -1],
                       [-1,  2,  8,  2, -1],
                       [-1,  2,  2,  2, -1],
                       [-1, -1, -1, -1, -1]])
    filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    ringing = np.std(filtered)
    return float(np.clip(ringing / 50.0, 0, 1))

def bitrate_estimation_score(frame):
    """Оценка битрейта на основе артефактов сжатия"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    # Комбинация blockiness и banding
    block = blockiness(frame)
    band = banding(gray)
    # Низкий битрейт = больше артефактов
    artifacts = (block + band) / 2.0
    # Инвертируем: больше артефактов = ниже оценка битрейта
    bitrate_score = 1.0 - np.clip(artifacts, 0, 1)
    return float(bitrate_score)

def codec_artifact_entropy(frame):
    """Энтропия артефактов кодека"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    # Анализируем паттерны артефактов через DCT-подобный анализ
    # Разбиваем на блоки 8x8 (стандартный размер для JPEG/MPEG)
    h, w = gray.shape
    block_size = 8
    entropies = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            # Вычисляем вариацию в блоке
            block_std = np.std(block)
            entropies.append(block_std)
    if entropies:
        hist = np.histogram(entropies, bins=20)[0]
        hist = hist / (hist.sum() + 1e-10)
        return float(entropy(hist))
    return 0.0

# ---------------------------------------------------------------
# LENS QUALITY
# ---------------------------------------------------------------

def vignetting(frame):
    h, w = frame.shape[:2]
    center = frame[h//4:3*h//4, w//4:3*w//4].mean()
    corners = np.mean([frame[0:h//4, 0:w//4],
                       frame[0:h//4, 3*w//4:w],
                       frame[3*h//4:h, 0:w//4],
                       frame[3*h//4:h, 3*w//4:w]])
    return float(center - corners)

def chromatic_aberration(frame):
    shift = np.abs(cv2.Canny(frame[:,:,2],100,200) - cv2.Canny(frame[:,:,0],100,200))
    return float(shift.mean())

def distortion_type(frame):
    """Тип дисторсии объектива (barrel/pincushion)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h, w = gray.shape
    # Анализируем кривизну линий от центра к краям
    center_y, center_x = h // 2, w // 2
    # Проверяем прямые линии (если есть)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    if lines is not None and len(lines) > 0:
        curvatures = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Вычисляем расстояние от центра до линии
            dist1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
            dist2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)
            # Barrel: линии выгнуты наружу (больше расстояние в центре)
            # Pincushion: линии выгнуты внутрь (меньше расстояние в центре)
            if dist1 > 0 and dist2 > 0:
                curvature = (dist1 + dist2) / 2.0
                curvatures.append(curvature)
        
        if curvatures:
            avg_curvature = np.mean(curvatures)
            # Эвристика: если среднее расстояние большое = barrel, малое = pincushion
            if avg_curvature > np.sqrt(h**2 + w**2) * 0.4:
                return "barrel"
            elif avg_curvature < np.sqrt(h**2 + w**2) * 0.2:
                return "pincushion"
    
    return "none"

def lens_sharpness_drop_off(frame):
    """Снижение резкости к краям объектива"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    
    # Резкость в центре
    center_region = gray[center_y-h//4:center_y+h//4, center_x-w//4:center_x+w//4]
    center_sharpness = cv2.Laplacian(center_region, cv2.CV_64F).var()
    
    # Резкость в углах
    corners = [
        gray[0:h//4, 0:w//4],
        gray[0:h//4, 3*w//4:w],
        gray[3*h//4:h, 0:w//4],
        gray[3*h//4:h, 3*w//4:w]
    ]
    corner_sharpness = np.mean([cv2.Laplacian(c, cv2.CV_64F).var() for c in corners])
    
    if center_sharpness > 0:
        drop_off = 1.0 - (corner_sharpness / center_sharpness)
        return float(np.clip(drop_off, 0, 1))
    return 0.0

def lens_obstruction_probability(frame):
    """Вероятность препятствий на объективе"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    # Препятствия обычно создают темные области или размытие
    # Анализируем локальные минимумы яркости
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    diff = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    # Большие локальные отклонения могут указывать на препятствия
    obstruction_score = np.mean(diff > 30) / 255.0
    return float(np.clip(obstruction_score, 0, 1))

def lens_dirt_probability(frame):
    """Вероятность грязи на объективе"""
    # Аналогично obstruction, но ищем более специфичные паттерны
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    # Грязь создает темные пятна
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
    # Ищем маленькие темные области
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_dark_areas = sum(1 for c in contours if cv2.contourArea(c) < 100)
    dirt_prob = min(small_dark_areas / 50.0, 1.0)  # Нормализуем
    return float(dirt_prob)

def veiling_glare_score(frame):
    """Оценка veiling glare (засветка объектива)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    # Veiling glare создает общее снижение контраста
    # Анализируем распределение яркости
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-10)
    # Glare приводит к смещению гистограммы в сторону ярких значений
    bright_ratio = hist[200:].sum()
    # Но также снижает контраст
    contrast = gray.std()
    # Комбинируем: высокий bright_ratio + низкий contrast = glare
    glare_score = bright_ratio * (1.0 - contrast / 128.0)
    return float(np.clip(glare_score, 0, 1))

# ---------------------------------------------------------------
# OBSTRUCTIONS (DIRT / FOG)
# ---------------------------------------------------------------

def fog_score(frame):
    lap = cv2.Laplacian(frame, cv2.CV_64F).var()
    return float(1.0 / (lap + 1))

# ---------------------------------------------------------------
# TEMPORAL METRICS
# ---------------------------------------------------------------

def temporal_flicker(prev_gray, gray):
    if prev_gray is None:
        return 0.0
    return float(np.mean(np.abs(prev_gray - gray)))

def rolling_shutter_artifacts_score(prev_frame, curr_frame):
    """Оценка артефактов rolling shutter"""
    if prev_frame is None or curr_frame is None:
        return 0.0
    
    # Rolling shutter создает искажения при быстром движении камеры
    # Анализируем вертикальные искажения между кадрами
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
    
    # Вычисляем optical flow для детекции искажений
    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Rolling shutter создает нелинейные искажения по вертикали
    # Анализируем вариацию вертикального потока по горизонтали
    h, w = flow.shape[:2]
    vertical_flow = flow[:, :, 1]  # Вертикальная компонента
    
    # Разбиваем на вертикальные полосы и сравниваем поток
    num_strips = 5
    strip_width = w // num_strips
    strip_flows = []
    for i in range(num_strips):
        strip = vertical_flow[:, i*strip_width:(i+1)*strip_width]
        strip_flows.append(np.mean(np.abs(strip)))
    
    # Большая вариация между полосами = rolling shutter
    flow_variation = np.std(strip_flows) if len(strip_flows) > 1 else 0.0
    rolling_shutter_score = float(np.clip(flow_variation / 5.0, 0, 1))
    
    return rolling_shutter_score

# ---------------------------------------------------------------
# SHOT QUALITY CLASSIFIER (CLIP + AESTHETICS)
# ---------------------------------------------------------------

class ShotQualityZeroShot:
    def __init__(self, device="cuda", rs_path: Optional[str] = None):
        self.device = device
        self.rs_path = rs_path
        # CLIP удалён - используем только core_clip
        # Если core данные недоступны, будет ошибка при predict
        self.aesthetic = AestheticPredictor("vit-l-14")

        self.prompts = [
            "cinematic shot, high-quality professional footage",
            "professional low-light cinematic footage",
            "good smartphone video quality",
            "poor smartphone video quality, grainy, noisy",
            "webcam low resolution footage",
            "screen recording of display",
            "cctv surveillance camera footage low quality"
        ]
        
        # Загружаем текстовые эмбеддинги только один раз (они не зависят от кадра)
        # Но для этого нужен CLIP модель, поэтому временно оставляем локальную загрузку
        # TODO: вынести текстовые промпты в core провайдер или использовать предрасчитанные
        try:
            self.clip_model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.text = clip.tokenize(self.prompts).to(device)
            with torch.no_grad():
                self.txt_feat = self.clip_model.encode_text(self.text)
                self.txt_feat /= self.txt_feat.norm(dim=-1, keepdim=True)
        except Exception as e:
            logger.warning(f"{name} | ShotQualityZeroShot.__init__ | Failed to load CLIP for text embeddings: {e}")
            self.clip_model = None
            self.txt_feat = None

    def predict(self, frame, frame_index: Optional[int] = None):
        """
        Предсказывает качество кадра. Использует только core_clip для изображений.
        """
        if frame_index is None:
            raise RuntimeError(
                f"{name} | ShotQualityZeroShot.predict | frame_index обязателен для чтения core_clip"
            )
        
        if not self.rs_path:
            raise RuntimeError(
                f"{name} | ShotQualityZeroShot.predict | rs_path обязателен для чтения core_clip"
            )
        
        # Загружаем CLIP эмбеддинг из core_clip
        img_feat = _load_core_clip_embeddings(self.rs_path, frame_index)
        if img_feat is None:
            raise RuntimeError(
                f"{name} | ShotQualityZeroShot.predict | core_clip embeddings не найдены для frame {frame_index}. "
                f"Убедитесь, что core провайдер core_clip запущен перед этим модулем. "
                f"rs_path: {self.rs_path}"
            )

        img_feat = img_feat.reshape(1, -1)  # [1, D]
        img_feat = img_feat / (np.linalg.norm(img_feat, axis=1, keepdims=True) + 1e-9)
        
        # Вычисляем logits с текстовыми эмбеддингами
        if self.txt_feat is not None:
            logits = (torch.from_numpy(img_feat).to(self.device) @ self.txt_feat.T).softmax(dim=-1).cpu().numpy()[0]
        else:
            # Fallback: равномерное распределение
            logits = np.ones(7) / 7.0
        
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        aesthetic = float(self.aesthetic.predict(pil))

        return {
            "quality_cinematic_prob": float(logits[0]),
            "quality_lowlight_cinematic_prob": float(logits[1]),
            "quality_smartphone_good_prob": float(logits[2]),
            "quality_smartphone_poor_prob": float(logits[3]),
            "quality_webcam_prob": float(logits[4]),
            "quality_screenrecord_prob": float(logits[5]),
            "quality_surveillance_prob": float(logits[6]),
            "aesthetic_score": aesthetic,
            "clip_embedding": img_feat[0].tolist()
        }

# ---------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------

class ShotQualityPipeline:
    def __init__(self, device="cuda", rs_path: Optional[str] = None):
        self.device = device
        self.rs_path = rs_path

        # MiDaS удалён - может использоваться core_depth_midas (опционально, если нужно)
        # try:
        #     self.midas, self.midas_tf = load_midas(device)
        #     logger.info("Загружена модель midas")
        # except Exception as e:
        #     logger.error(f"Ошибка загрузки midas : {e}")
        try:
            self.dncnn = load_dncnn(device)
            logger.info("Загружена модель dncnn")
        except Exception as e:
            logger.error(f"Ошибка загрузки dncnn : {e}")
        try:
            self.cbdnet = load_cbdnet(device)
            logger.info("Загружена модель cbdnet")
        except Exception as e:
            logger.error(f"Ошибка загрузки cbdnet : {e}")
        try:
            self.classifier = ShotQualityZeroShot(device, rs_path=rs_path)
            logger.info("Загружена модель ShotQualityZeroShot")
        except Exception as e:
            logger.error(f"Ошибка загрузки ShotQualityZeroShot : {e}")

        self.prev_gray = None
        self.prev_frame = None
        # Для temporal метрик
        self.frame_history = {
            "sharpness": [],
            "exposure": [],
            "noise": []
        }

    def process_frame(self, frame, frame_index: Optional[int] = None):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Sharpness (основная + secondary)
        tenengrad = sharpness_tenengrad(gray)
        motion_blur = motion_blur_probability(frame)
        sharp_sec = sharpness_secondary(gray, frame)
        spatial_freq = _spatial_frequency_mean(frame)

        # Noise (сведённые фичи)
        noise_dncnn = noise_estimation_dncnn(self.dncnn, frame_rgb, self.device)
        noise_cbdnet_stats = noise_estimation_cbdnet(self.cbdnet, frame_rgb, self.device)
        noise_model_mean = float(noise_dncnn)
        if isinstance(noise_cbdnet_stats, dict):
            noise_model_mean = float(
                0.5 * noise_model_mean + 0.5 * noise_cbdnet_stats.get("noise_mean", noise_model_mean)
            )

        noise_luma = noise_level_luma(gray)
        noise_chroma = noise_level_chroma(frame)
        noise_chroma_ratio = float(
            noise_chroma / (noise_luma + 1e-6)
        ) if noise_luma > 0 else float(noise_chroma)
        iso_est = iso_estimated_value(frame)
        grain = grain_strength(frame)
        noise_entropy = noise_spatial_entropy(gray)

        # Exposure
        exp = exposure_metrics(gray)

        # Contrast
        c_global = contrast_global(gray)
        c_local = contrast_local(gray)
        c_dynamic = contrast_dynamic_range(gray)
        c_clarity = contrast_clarity_score(gray)
        micro = microcontrast(gray)

        # Color
        wb = white_balance_shift(frame)
        cast = color_cast(frame)
        skin_tone = skin_tone_accuracy_score(frame)
        color_fidelity = color_fidelity_index(frame)
        color_noise = color_noise_level(frame)
        color_uniformity = color_uniformity_score(frame)

        # Compression
        block = blockiness(frame)
        band = banding(gray)
        ringing = ringing_artifacts_level(frame)
        bitrate = bitrate_estimation_score(frame)
        codec_entropy = codec_artifact_entropy(frame)

        # Lens
        vign = vignetting(frame)
        ca = chromatic_aberration(frame)
        distortion = distortion_type(frame)
        sharpness_drop = lens_sharpness_drop_off(frame)
        obstruction = lens_obstruction_probability(frame)
        dirt = lens_dirt_probability(frame)
        glare = veiling_glare_score(frame)

        # Fog/dirt
        fog = fog_score(frame)

        # Temporal
        flicker = temporal_flicker(self.prev_gray, gray)
        rolling_shutter = rolling_shutter_artifacts_score(self.prev_frame, frame)
        
        # Сохраняем для temporal метрик
        self.frame_history["sharpness"].append(tenengrad)
        self.frame_history["exposure"].append(exp.get("midtones_balance", 0.5))
        self.frame_history["noise"].append(noise_luma)
        
        self.prev_gray = gray.copy()
        self.prev_frame = frame.copy()

        # Quality classifier
        if frame_index is not None:
            qc = self.classifier.predict(frame, frame_index=frame_index)
        else:
            qc = {}

        return {
            # Sharpness (компактные признаки)
            "sharpness_tenengrad": tenengrad,
            "sharpness_secondary": sharp_sec,
            "motion_blur_probability": motion_blur,
            "spatial_frequency_mean": spatial_freq,

            # Noise (сведённые)
            "noise_model_mean": noise_model_mean,
            "noise_cbdnet_stats": noise_cbdnet_stats,
            "noise_level_luma": noise_luma,
            "noise_level_chroma": noise_chroma,
            "noise_chroma_ratio": noise_chroma_ratio,
            "iso_estimated_value": iso_est,
            "grain_strength": grain,
            "noise_spatial_entropy": noise_entropy,

            # Exposure
            **exp,

            # Contrast
            "contrast_global": c_global,
            "contrast_local": c_local,
            "contrast_dynamic_range": c_dynamic,
            "contrast_clarity_score": c_clarity,
            "microcontrast": micro,

            # Color
            **wb,
            "color_cast_type": cast,
            "skin_tone_accuracy_score": skin_tone,
            "color_fidelity_index": color_fidelity,
            "color_noise_level": color_noise,
            "color_uniformity_score": color_uniformity,

            # Compression
            "compression_blockiness_score": block,
            "banding_intensity": band,
            "ringing_artifacts_level": ringing,
            "bitrate_estimation_score": bitrate,
            "codec_artifact_entropy": codec_entropy,

            # Lens
            "vignetting_level": vign,
            "chromatic_aberration_level": ca,
            "distortion_type": distortion,
            "lens_sharpness_drop_off": sharpness_drop,
            "lens_obstruction_probability": obstruction,
            "lens_dirt_probability": dirt,
            "veiling_glare_score": glare,

            # Obstruction
            "fog_haziness_score": fog,

            # Temporal
            "temporal_flicker_score": flicker,
            "rolling_shutter_artifacts_score": rolling_shutter,

            # Shot Quality Classifier
            **qc
        }

    def process(self, frame_manager, frame_indices) -> Dict:
        frame_results: Dict[int, Dict] = {}

        # --- Frame-level обработка ---
        for i, idx in enumerate(frame_indices):
            frame = frame_manager.get(idx)

            if frame.ndim == 3 and frame.shape[2] == 3:
                frame_bgr = (
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if frame.dtype == np.uint8
                    else frame
                )
            else:
                frame_bgr = frame

            result = self.process_frame(frame_bgr, frame_index=idx)
            frame_results[idx] = result

            logger.info(f"Обработано кадров: {i + 1}/{len(frame_indices)}")

        # --- Агрегация frame-level метрик ---
        frame_features: Dict[str, float] = {}

        numeric_keys = set()
        for result in frame_results.values():
            for k, v in result.items():
                if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                    numeric_keys.add(k)

        for key in numeric_keys:
            values = [
                float(r[key])
                for r in frame_results.values()
                if key in r and isinstance(r[key], (int, float, np.number))
            ]

            if not values:
                continue

            values_np = np.array(values, dtype=np.float32)

            frame_features[f"avg_{key}"] = float(values_np.mean())
            frame_features[f"std_{key}"] = float(values_np.std())
            frame_features[f"min_{key}"] = float(values_np.min())
            frame_features[f"max_{key}"] = float(values_np.max())

        # --- Temporal метрики ---
        temporal_features: Dict[str, float] = {}

        def robust_cv(x):
            """
            Робастный коэффициент вариации: IQR / median.
            Менее чувствителен к выбросам, чем std/mean.
            """
            x = np.array(x, dtype=np.float32)
            if x.size == 0:
                return 0.0
            med = np.median(x)
            if med == 0:
                return 0.0
            q75, q25 = np.percentile(x, [75, 25])
            iqr = q75 - q25
            return float(iqr / (abs(med) + 1e-8))

        if len(self.frame_history["sharpness"]) > 1:
            rc = robust_cv(self.frame_history["sharpness"])
            temporal_features["temporal_sharpness_stability"] = float(1.0 - np.clip(rc, 0.0, 1.0))

        if len(self.frame_history["exposure"]) > 1:
            rc_exp = robust_cv(self.frame_history["exposure"])
            temporal_features["temporal_exposure_stability"] = float(1.0 - np.clip(rc_exp, 0.0, 1.0))
            temporal_features["exposure_consistency_over_time"] = float(
                np.clip(1.0 - np.std(self.frame_history["exposure"]), 0.0, 1.0)
            )

        if len(self.frame_history["noise"]) > 1:
            noise = np.array(self.frame_history["noise"], dtype=np.float32)
            temporal_features["temporal_noise_variation"] = float(
                np.std(noise) / (np.mean(noise) + 1e-8)
            )

        # --- Reset ---
        self.frame_history = {"sharpness": [], "exposure": [], "noise": []}
        self.prev_gray = None
        self.prev_frame = None

        return {
            "frames": frame_results,
            "frame_features": frame_features,
            "temporal_features": temporal_features,
            "total_frames_processed": len(frame_results),
        }
