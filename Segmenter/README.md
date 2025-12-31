## Segmenter (реальный контракт для пайплайна)

Этот README содержит много идей/псевдо-алгоритмов, но ниже — **реальный контракт**, который сейчас реализован в коде `Segmenter/segmenter.py` и нужен для baseline.

### Что Segmenter генерирует

При запуске Segmenter для видео `video_path` он создаёт директорию:

- `output/<video_id>/video/metadata.json`
- `output/<video_id>/video/batch_*.npy`
- `output/<video_id>/audio/metadata.json`
- `output/<video_id>/audio/<video_id>.wav` (если включено/доступно ffmpeg)

### Union-sampled frames_dir (DEFAULT)

По новым контрактам **frames_dir хранит только union sampled кадры** (не все кадры видео).

- Segmenter считает `frame_indices` отдельно для каждого компонента (core/module).
- Строит `union_frame_indices_source` (source indices) и сохраняет **только эти кадры**.
- В `metadata.json` per-component `frame_indices` записаны **в union domain** (0..N-1) — именно их потом используют модули через `FrameManager.get()`.

Также фиксируются:
- `color_space="RGB"` (все кадры в `.npy` — RGB)
- `union_timestamps_sec` (mapping к исходному времени)

### Legacy режим (НЕ рекомендуется)

Можно извлечь все кадры (дорого по диску/IO):
- `--legacy-full-extract`

### Как запускать (пример)

Запуск с автоматической генерацией per-component budgets на основе `VisualProcessor/config.yaml`:

```bash
python Segmenter/segmenter.py \
  --video-path path/to/video.mp4 \
  --output Segmenter/data \
  --visual-cfg-path VisualProcessor/config.yaml
```

Опционально можно задать `run_id`/`video_id`/`sampling_policy_version`, чтобы они попали в `metadata.json`.

---

Философия: Не "нарезать", а "понимать" структуру
Цель — не просто получить равные куски, а выделить семантически целостные единицы, которые станут токенами для трансформера. Это могут быть: кадры, сцены, эпизоды, смысловые блоки (intro, main part, outro).

Уровень 1: Быстрая грубая сегментация (Shot Boundary Detection)
Задача: Найти технические склейки (резкие смены кадра) и плавные переходы (dissolve, fade).

Для ВСЕХ видео, независимо от длины. Это базовый, быстрый и надежный метод.

Методы:

Histogram-based: Сравнение цветовых гистограмм соседних кадров. Быстро, устойчиво к движению объектов.

Pixel-based: SSD (Sum of Squared Differences) между кадрами.

Pre-trained модели: Есть легкие CNN (например, на ResNet), обученные именно на детекции склеек. Для 100к видео это оптимально.

Результат: Видео разбивается на shots (планы). Для 4-секундного видео может быть 1 shot, для 20-минутного — сотни.

Проблема: Shots слишком короткие и их слишком много. Нужна группировка.

Уровень 2: Семантическая агрегация в сцены (Scene Detection)
Задача: Объединить визуально и смыслово связанные shots в сцены (scene). Это ключевой уровень для вашей задачи.

Визуально-временная кластеризация:

Для каждого shot (например, берем его средний кадр или усредненный эмбеддинг) извлекаем фич-вектор через предобученную CNN (например, Place365, или взяв слой из torchvision.models).

Строим граф смежности shots: вершины — shots, ребра — временная близость + визуальное сходство.

Применяем алгоритм кластеризации на графе (например, Spectral Clustering или Affinity Propagation). Shots, попавшие в один кластер и идущие подряд во времени, образуют scene.

Плюс: Хорошо работает для склейки диалогов (лицо A, лицо B, лицо A...), спортивных событий, лекций.

Адаптивный порог по времени:

Нельзя использовать фиксированную длину сцены. Вместо этого — правило: сцена не может быть короче X секунд и длиннее Y секунд.

Например: min_scene_duration=3 сек, max_scene_duration=60 сек. Если алгоритм выделил сцену в 90 секунд — принудительно разбиваем ее на две по самому слабому визуальному переходу внутри.

Использование аудио-подсказок (Audio Scene Detection):

Резкая смена звуковой дорожки (тишина -> музыка, речь -> музыка) — сильный маркер смены сцены.

Метод: Вычислите энергию аудио (RMS) или вектор аудио-фичей (из librosa) и найдите в нем точки резкого изменения (через обнаружение выбросов или анализ производной).

Уровень 3: Контент-специфичные эвристики (Очень важно!)
Поскольку у вас 100к видео, наверняка есть мета-данные (категория: "музыкальный клип", "интервью", "геймплей", "инфографика"). Используйте их!

Для коротких видео (до 30 сек): Отказывайтесь от сегментации вовсе! Там весь контент — одна смысловая единица. Ваш "сегмент" = все видео. Либо используйте фиксированное разбиение на 2-3 равных отрезка.

Для музыкальных клипов, тиктоков: Сегментация по ритму и биту. Используйте библиотеку librosa для обнаружения beats (librosa.beat.beat_track()). Смена кадра часто совпадает с битом.

Для лекций, презентаций, интервью: Детекция слайдов/заголовков через OCR (Tesseract) или детекция "чистых" фонов с текстом. Смена слайда = новый сегмент.

Для геймплея, спорта: Используйте детекцию счетчика/таймера (опять же OCR) или смены игровой карты/площадки (через сравнение эмбеддингов сцен).

Уровень 4: Иерархия и финальное принятие решения (Fusion Layer)
Создайте систему голосования или приоритетов:

Первичные детекторы работают параллельно:

Детектор технических склеек (Level 1).

Визуальный scene detector (Level 2).

Аудио scene detector (Level 2).

Контент-специфичный детектор (Level 3), если применимо.

Точки сегментации от всех детекторов сводятся на единую временную шкалу.

Правило слияния: Если две или более детектора указывают на переход в окне delta_t (например, 1 секунда), это сильный кандидат на границу сегмента. Если детектор один — проверяем его уверенность.

Пост-обработка:

Удаление слишком близких границ (мержим сегменты, если расстояние между границами < min_duration).

Обязательное разбиение слишком длинных сегментов (как описано выше).

Гарантированное минимальное количество сегментов: Даже для 4-секундного видео — минимум 1 сегмент. Для 20-минутного — не более, скажем, 50 сегментов (чтобы не перегружать трансформер).




# Пайплайн обработки видео (псевдо-экстракторы)
# Весь код и комментарии — на русском.
# Этот скрипт демонстрирует архитектуру: ресэмплинг -> локальные энкодеры -> pooling -> фиксированное представление.

import os
import random
from typing import Dict, Any, List
import numpy as np

# -----------------------------
# Конфигурация
# -----------------------------
BASE_DT = 0.5           # базовый шаг временной сетки (в секундах)
WINDOW_SIZE = 8         # размер окна локальной агрегации (в шагах BASE_DT). 8 * 0.5s = 4s
SUMMARY_TOKENS = 64     # число итоговых summary-токенов (фиксированное представление)

VIDEO_DIR = "./videos"  # Путь к папке с видео-файлами (в демонстрации файлы не читаются)

# -----------------------------
# Псевдо-экстракторы
# Каждый экстрактор выдаёт последовательность признаков с собственной частотой.
# В реальной системе здесь будут вызовы реальных моделей (face detector, yolo, wav2vec и т.д.).
# -----------------------------
class PseudoExtractor:
    def __init__(self, name: str, feat_dim: int, step: float):
        """
        name: имя экстрактора
        feat_dim: размер выходного вектора на одну точку времени
        step: собственный шаг экстракции (в секундах)
        """
        self.name = name
        self.feat_dim = feat_dim
        self.step = step

    def extract(self, duration: float) -> np.ndarray:
        """
        Симулирует извлечение последовательности фичей для видео длительностью duration.
        Возвращает массив формы [N, feat_dim], где N = floor(duration / step).
        """
        n = max(1, int(np.floor(duration / self.step)))
        return np.random.randn(n, self.feat_dim)

# Примеры экстракторов: эмоции, яркость, объекты, аудио-эмбеддинги
EXTRACTORS = [
    PseudoExtractor('emotion', 10, 1.0),     # эмоции каждые 1 сек
    PseudoExtractor('brightness', 6, 0.5),   # яркость каждые 0.5 сек
    PseudoExtractor('objects', 20, 2.0),     # объекты каждые 2 сек
    PseudoExtractor('audio', 32, 0.1),       # аудио-эмбедды каждые 0.1 сек
]

# -----------------------------
# Вспомогательные функции
# -----------------------------

def simulate_video_duration(min_s: int = 15, max_s: int = 600) -> float:
    """Симулирует длительность видео в секундах (для демонстрации)."""
    return float(random.randint(min_s, max_s))


def resample_to_base_grid(seq: np.ndarray, src_step: float, duration: float, base_dt: float = BASE_DT) -> np.ndarray:
    """
    Ресэмплирует входную последовательность seq (shape [N_src, F]) с шагом src_step
    на общую базовую сетку с шагом base_dt.

    Простейшая реализация: nearest-neighbor по времени (для демонстрации).
    В продакшене разумно использовать интерполяцию или агрегацию (mean, max, std).

    Возвращает массив shape [T, F], где T = floor(duration / base_dt).
    """
    T = max(1, int(np.floor(duration / base_dt)))
    base_times = np.arange(T) * base_dt
    src_times = np.arange(seq.shape[0]) * src_step

    res = np.zeros((T, seq.shape[1]), dtype=np.float32)
    for i, t in enumerate(base_times):
        idx = int(np.argmin(np.abs(src_times - t)))
        res[i] = seq[idx]
    return res


def local_encode(window: np.ndarray) -> np.ndarray:
    """
    Локальный энкодер: получает окно размером [W, F] и выдаёт вектор фиксированной длины.
    В реальном решении тут может быть маленький Transformer, 1D-CNN или MLP.

    Для демонстрации: используем простые статистики (mean, std, max).
    Возвращаем вектор размерности 3 * F.
    """
    mean = window.mean(axis=0)
    std = window.std(axis=0)
    mx = window.max(axis=0)
    return np.concatenate([mean, std, mx], axis=0)


# -----------------------------
# Learnable Attention Pooling (PyTorch)
# -----------------------------
import torch
import torch.nn as nn

class LearnableAttentionPooling(nn.Module):
    """
    Реализация обучаемого attention-pooling:
    - обучаемые query-вектора
    - attention(Q, K) -> softmax -> взвешенная сумма V
    """
    def __init__(self, d_model: int, num_queries: int = SUMMARY_TOKENS):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.queries = nn.Parameter(torch.randn(num_queries, d_model) * 0.02)

    def forward(self, local_tokens: torch.Tensor) -> torch.Tensor:
        if local_tokens.numel() == 0:
            return torch.zeros(self.num_queries, self.d_model, device=self.queries.device)
        Q = self.queries
        K = local_tokens
        V = local_tokens
        scores = torch.matmul(Q, K.transpose(0,1)) / (self.d_model ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.matmul(weights, V)
        return pooled(local_tokens: np.ndarray, k: int = SUMMARY_TOKENS) -> np.ndarray:
    """
    Псевдо-реализация pooling-а на фиксированное число summary-токенов.

    Идея: создаём k "query" векторов (random в демо), считаем скалярные веса
    и формируем k свёрнутых векторов size D.

    В продакшене: эти query параметризуются и обучаются (Cross-attention).
    """
    if local_tokens.size == 0:
        return np.zeros((k, 1), dtype=np.float32)

    M, D = local_tokens.shape
    # Случайные query (в реальном коде learnable параметры)
    queries = np.random.randn(k, D).astype(np.float32)
    # Скалярные произведения -> [k, M]
    scores = queries @ local_tokens.T
    # softmax по локальным токенам
    scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    weights = scores_exp / (scores_exp.sum(axis=1, keepdims=True) + 1e-9)
    # Взвешенная сумма: [k, D]
    pooled = weights @ local_tokens
    return pooled

# -----------------------------
# Процессинг одного видео
# -----------------------------

def process_single_video(video_path: str) -> Dict[str, Any]:
    """
    Полный pipeline для одного видео (симуляция):
      1) Определяем длительность видео
      2) Запускаем все экстракторы -> получаем разные последовательности
      3) Ресэмплируем каждую последовательность на базовую сетку
      4) Конкатенируем признаки по feature-оси -> матрица [T, F_all]
      5) Разбиваем на окна размера WINDOW_SIZE и применяем local_encode -> [M, D_local]
      6) Применяем attention_pooling -> фиксированные [K, D_local]

    Возвращаем словарь с промежуточными результатами для отладки.
    """
    duration = simulate_video_duration()

    # 2) Запуск экстракторов
    raw = {}
    for ex in EXTRACTORS:
        raw[ex.name] = ex.extract(duration)

    # 3) Ресэмплинг на базовую сетку и конкатенация
    resampled_parts = []
    for ex in EXTRACTORS:
        seq = raw[ex.name]
        res = resample_to_base_grid(seq, ex.step, duration)
        resampled_parts.append(res)

    # Проверка: все части должны иметь одинаковое количество временных шагов T
    T = resampled_parts[0].shape[0]
    for part in resampled_parts:
        assert part.shape[0] == T, "Несовпадающие временные размеры после ресэмплинга"

    # Конкатенируем по feature-оси => X shape [T, F_all]
    X = np.concatenate(resampled_parts, axis=1)

    # 5) Локальное кодирование по окнам
    local_tokens = []
    for start in range(0, T, WINDOW_SIZE):
        window = X[start:start + WINDOW_SIZE]
        if window.shape[0] < 1:
            continue
        enc = local_encode(window)
        local_tokens.append(enc)

    if len(local_tokens) == 0:
        local_tokens_arr = np.zeros((0, X.shape[1] * 3), dtype=np.float32)
    else:
        local_tokens_arr = np.stack(local_tokens, axis=0)

    # 6) Pooling в фиксированное число токенов
    summary = attention_pooling(local_tokens_arr, SUMMARY_TOKENS)

    return {
        'video_path': video_path,
        'duration': duration,
        'T': T,
        'X_shape': X.shape,
        'local_tokens_shape': local_tokens_arr.shape,
        'summary_shape': summary.shape,
        'summary': summary
    }

# -----------------------------
# Пакетная обработка папки с видео
# -----------------------------

def process_folder(folder: str) -> List[Dict[str, Any]]:
    """Обрабатывает все файлы в папке и возвращает список результатов."""
    results = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            continue
        path = os.path.join(folder, fname)
        print(f"Обрабатываю: {fname}")
        res = process_single_video(path)
        results.append(res)
    return results

# -----------------------------
# Запуск (пример)
# -----------------------------
if __name__ == '__main__':
    # Для демонстрации можно создать папку ./videos с пустыми файлами с нужными расширениями,
    # код не будет пытаться реально читать видео, а симулирует длительность и фичи.
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR, exist_ok=True)
        # создаём несколько пустых имитируемых файлов
        for i in range(5):
            open(os.path.join(VIDEO_DIR, f"video_{i}.mp4"), 'a').close()

    dataset = process_folder(VIDEO_DIR)

    print('\nРезультаты обработки:')
    for item in dataset:
        print(item['video_path'], 'duration=', item['duration'], 'T=', item['T'],
              'X_shape=', item['X_shape'], 'local=', item['local_tokens_shape'], 'summary=', item['summary_shape'])

    print('\nГотово. Фиксированные summary-токены для каждого видео находятся в поле item[\'summary\']')
