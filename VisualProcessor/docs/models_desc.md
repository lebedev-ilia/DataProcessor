# Описание модулей и их моделей

## Архитектура core-слоя

VisualProcessor использует двухфазную архитектуру:

1. **Фаза 1 — core-провайдеры моделей** (`core/model_process/*`):
   - Тяжёлые модели запускаются **один раз на видео**
   - Сохраняют **сырые/универсальные фичи** в `result_store/core_*/`
   - Каждый провайдер имеет свою виртуальную среду и запускается через subprocess

2. **Фаза 2 — модульные анализаторы** (`modules/*`):
   - Читают результаты из `result_store/core_*` и других модулей
   - Не инициализируют модели заново, работают с уже готовыми данными
   - Агрегируют и пост-обрабатывают фичи

### Core-провайдеры

#### core_clip
- **Модель**: OpenAI CLIP (ViT-B/32, ViT-L/14 и др.)
- **Выход**: `result_store/core_clip/embeddings.npz` (per-frame embeddings)
- **Используют**: `video_pacing`, `story_structure`, `high_level_semantic`, `cut_detection`, `shot_quality`
- **Документация**: `docs/FEATURES_DESCRIPTION_core_clip.md`

#### core_optical_flow
- **Модель**: RAFT (small/large) или Farneback
- **Выход**: `result_store/optical_flow/statistical_analysis.json` (motion statistics)
- **Используют**: `video_pacing`, `story_structure`, `cut_detection`, `text_scoring`
- **Документация**: `docs/FEATURES_DESCRIPTION_core_optical_flow.md`

#### core_face_landmarks
- **Модель**: Mediapipe (pose, hands, face_mesh)
- **Выход**: `result_store/core_face_landmarks/landmarks.json` (pose/hands/face landmarks)
- **Используют**: `behavioral`, `detalize_face_modules`, `frames_composition`, `text_scoring`
- **Документация**: `docs/FEATURES_DESCRIPTION_core_face_landmarks.md`

#### core_depth_midas
- **Модель**: MiDaS (depth estimation)
- **Выход**: `result_store/core_depth_midas/depth.json` (depth statistics)
- **Используют**: `frames_composition`, `shot_quality`
- **Документация**: `docs/FEATURES_DESCRIPTION_core_depth_midas.md`

#### core_object_detections
- **Модель**: YOLO (yolo11x.pt и др.) или OWL-ViT/OWLv2 (open-vocabulary)
- **Выход**: `result_store/core_object_detections/detections.json` (object detections per frame)
- **Используют**: `frames_composition`, `object_detection`, `scene_classification`
- **Документация**: `docs/FEATURES_DESCRIPTION_core_object_detections.md`

### Миграция модулей на core-слой

Модули постепенно переходят на использование core-провайдеров:

- ✅ **behavioral**: использует `core_face_landmarks` (с fallback на локальный Mediapipe)
- ✅ **video_pacing**: использует `core_clip` и `core_optical_flow` (с fallback)
- ✅ **story_structure**: использует `core_clip` и `core_optical_flow` (с fallback)
- ✅ **detalize_face_modules**: использует `core_face_landmarks` (с fallback)
- ✅ **text_scoring**: использует `core_optical_flow` и `core_face_landmarks` (с fallback)

Подробнее о плане рефакторинга: `docs/core_refactor_plan.md`

---

## object_detection

### 1 Вариант - Если изначально неизвестно какие могут быть объекты на видео

### Модели:

```py
from ultralytics import YOLO

# Доступные модели YOLO:
# yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
model = YOLO("yolo11x.pt", task="detect")
model.to(device)
```

YOLO детектирует 80 классов COCO (person, car, bicycle, и т.д.)

### 2 Вариант - Open-vocabulary detection по текстовым запросам

### Модели:

```py
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)

# OWL-ViT варианты:
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16").to(device)

# или
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)

# или
processor = OwlViTProcessor.from_pretrained("google/owlvit-large-patch14")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14").to(device)

# OWLv2 варианты:
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16").to(device)
```

OWL-ViT/OWLv2 позволяет детектировать объекты по произвольным текстовым описаниям без переобучения.

## scene_classification

### Модели:

```py
from torchvision import models
from transformers import CLIPProcessor, CLIPModel  # опционально
import timm  # опционально

# Places365 модели (по умолчанию)
# ResNet18
model = models.resnet18(pretrained=False)
# Загрузка весов Places365: resnet18_places365.pth.tar

# ResNet50
model = models.resnet50(pretrained=False)
# Загрузка весов Places365: resnet50_places365.pth.tar
```

```py
# Современные архитектуры через timm
# EfficientNet
model = timm.create_model("efficientnet_b0", pretrained=True)
# или efficientnet_b1, efficientnet_b2, efficientnet_b3

# ConvNeXt
model = timm.create_model("convnext_tiny", pretrained=True)
# или convnext_small, convnext_base

# Vision Transformers
model = timm.create_model("vit_base_patch16_224", pretrained=True)
# или vit_large_patch16_224

# RegNet
model = timm.create_model("regnetx_002", pretrained=True)
# или regnetx_004, regnetx_006

# ResNet через timm
model = timm.create_model("resnet50", pretrained=True)
# или resnet101
```

```py
# Опционально: CLIP для семантических фичей
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
```

```py
# Категории Places365 (365 классов сцен)
# Файл: categories_places365.txt
# Загружается автоматически или из локального файла
```

Доступные модели:
- **Places365**: `resnet18`, `resnet50`
- **TIMM**: `efficientnet_b0/b1/b2/b3`, `convnext_tiny/small/base`, `vit_base_patch16_224`, `vit_large_patch16_224`, `regnetx_002/004/006`, `resnet50`, `resnet101`

## face_detection

### Модели:

```py
from insightface.app import FaceAnalysis

# Инициализация с GPU
app = FaceAnalysis(providers=["CUDAExecutionProvider"])

# Или с CPU
app = FaceAnalysis(providers=["CPUExecutionProvider"])

app.prepare(ctx_id=0, det_size=(640, 640))
```

InsightFace автоматически загружает модели детекции и распознавания лиц.

## detalize_face_modules
    
### Модели:

```py
import mediapipe as mp
_FACE_MESH = mp.solutions.face_mesh
face_mesh = _FACE_MESH.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
```

## emotion_face
    
### Модели:

```py
from models.emonet.emonet.models.emonet import EmoNet

# Загрузка модели EmoNet
model = EmoNet(n_expression=8).to(device)

# Загрузка весов
# Путь к весам: models/emonet/pretrained/emonet_8.pth
checkpoint = torch.load("models/emonet/pretrained/emonet_8.pth", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
```

Параметры:
- `n_expression=8`: количество классов эмоций (Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt)

## behavioral

### Модели:

Модуль `behavioral` теперь работает поверх core‑слоя:

- **Primary**: использует предрасчитанные Mediapipe‑landmarks из `core_face_landmarks/landmarks.json`  
  (провайдер `core_face_landmarks` в фазе core считает pose/hands/face_mesh один раз на видео).
- **Fallback**: если core‑данные недоступны, инициализирует Mediapipe локально, как раньше:

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=pose_model_complexity,
    smooth_landmarks=pose_smooth_landmarks,
    min_detection_confidence=pose_min_detection_confidence,
    min_tracking_confidence=pose_min_tracking_confidence
)

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=hands_max_num_hands,
    model_complexity=hands_model_complexity,
    min_detection_confidence=hands_min_detection_confidence,
    min_tracking_confidence=hands_min_tracking_confidence
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=face_max_num_faces,
    refine_landmarks=face_refine_landmarks,
    min_detection_confidence=face_min_detection_confidence,
    min_tracking_confidence=face_min_tracking_confidence
)
```

## optical_flow

### Модели:

```py
import torchvision.models.optical_flow as models

if model_type == "large":
    model = models.raft_large(
        weights=models.Raft_Large_Weights.DEFAULT,
        progress=True
    )
else:
    model = models.raft_small(
        weights=models.Raft_Small_Weights.DEFAULT,
        progress=True
    )
```

## action_recognition

### Модели:

```py
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

# model_name - путь к предобученной модели VideoMAE
processor = VideoMAEImageProcessor.from_pretrained(model_name)
model = VideoMAEForVideoClassification.from_pretrained(model_name).to(device)
model.eval()
```

Примеры моделей:
- `MCG-NJU/videomae-base-finetuned-kinetics` (Kinetics-400)
- `MCG-NJU/videomae-large-finetuned-kinetics` (Kinetics-400)
- Другие предобученные VideoMAE модели из Hugging Face

## color_light

### Модели:

```py
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
kmeans.fit(sampled)
colors = kmeans.cluster_centers_.astype(int)
```

## frames_composition

### Модели:

```py
yolo_model = YOLO("yolo11n.pt")
```

```py
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=self.config.max_num_faces,
    min_detection_confidence=self.config.min_detection_confidence
)
```

```py
torch.hub.set_dir("./models")
midas_model = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small",
    pretrained=True,
    trust_repo=True,
    verbose=False
)
transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms",
    trust_repo=True,
    verbose=False
)
```

```py
style_model = models.resnet50(pretrained=True)
style_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

## shot_quality

### Модели:

```py
try:
    from aesthetic_predictor import AestheticPredictor
except ImportError:
    class AestheticPredictor:
        def __init__(self, model_name):
            self.model_name = model_name
        def predict(self, pil_image):
            img_array = np.array(pil_image)
            brightness = np.mean(img_array) / 255.0
            contrast = np.std(img_array) / 255.0
            return float(0.5 + 0.3 * brightness + 0.2 * contrast)
```

```py
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
```

```py
try:
    from TrainingCodes.dncnn_pytorch.main_test import DnCNN
except ImportError as e:
    raise ImportError(f"Не удалось импортировать класс DnCNN: {e}")

sys.modules['__main__'].DnCNN = DnCNN

weights_path = os.path.join(repo_dir, "TrainingCodes/dncnn_pytorch/models/DnCNN_sigma25/model.pth")

with torch.serialization.safe_globals([DnCNN]):
    model = torch.load(weights_path, map_location=device, weights_only=False)
```

```py
checkpoint_path = os.path.join(models_dir, "checkpoint.pth.tar")

from model.cbdnet import Network # type: ignore

model = Network()
model.to(device)
model = nn.DataParallel(model)

ckpt = torch.load(checkpoint_path, map_location=device)
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
else:
    model.load_state_dict(ckpt)
```

```py
self.clip_model, self.preprocess = clip.load("ViT-L/14", device=device)
```

## text_scoring

### Модели:

```py
import easyocr

reader = easyocr.Reader(['en', 'ru'], gpu=True)
results = reader.readtext(frame)
```

Альтернатива:
```py
import pytesseract

text = pytesseract.image_to_string(frame, lang='eng+rus')
```

### Зависимость от core‑слоя:

- **motion**: при наличии `core_optical_flow` используется кривая движения из  
  `result_store/optical_flow/statistical_analysis.json` (`statistics.frame_statistics[*].magnitude_mean_px_sec_norm`  
  или `magnitude_mean_px_sec` / `magnitude_mean` в качестве `motion_peaks`).  
- **face**: на текущем этапе модуль читает face‑эмоции из модуля `emotion_face`  
  (`result_store/emotion_face/*.json`, поле `emotion_curve`) — планируется переход на core‑провайдер  
  поверх `core_face_landmarks`.  
- **audio**: источник audio‑сигналов зарезервирован под будущий `core_audio_embeddings` и пока не активен.  

## high_level_semantic

### Модели:

```py
import clip  # openai/clip
# или
import open_clip  # open_clip

# OpenAI CLIP
model, preprocess = clip.load("ViT-B/32", device=device)
# или
model, preprocess = clip.load("ViT-L/14", device=device)

# OpenCLIP
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", 
    pretrained='laion2b_s13b_b90k'
)
model.to(device).eval()
```

Доступные модели CLIP:
- OpenAI CLIP: `ViT-B/32`, `ViT-B/16`, `ViT-L/14`, `RN50`, `RN101`
- OpenCLIP: `ViT-B-32`, `ViT-L-14`, `ViT-H-14`, и другие

## similarity_metrics

### Модели:

Модуль не использует модели напрямую, работает с embeddings и фичами из других модулей. Использует библиотеки для вычисления метрик:

```py
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from scipy.stats import wasserstein_distance
from sklearn.metrics import jaccard_score
```

## uniqueness

### Модели:

Модуль не использует модели напрямую, работает с embeddings и фичами из других модулей. Использует библиотеки для вычисления метрик:

```py
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, entropy
from scipy.stats import wasserstein_distance
```

## micro_emotion

### Модели:

```bash
# Docker образ OpenFace
docker pull openface/openface:latest
```

Используется через Docker контейнер:
```py
# OpenFace запускается через Docker
docker run --rm \
    -v /input:/input \
    -v /output:/output \
    openface/openface:latest \
    /usr/local/bin/FeatureExtraction \
    -f /input/video.mp4 \
    -out_dir /output \
    -pose -aus -gaze -2Dfp -3Dfp -tracked
```

OpenFace извлекает:
- Action Units (AU01-AU45)
- Head pose (6 DOF)
- Gaze direction
- Facial landmarks (68 точек в 2D и 3D)

## cut_detection

### Модели:

```py
# Опционально: CLIP для zero-shot классификации переходов
import clip
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
```

```py
# Опционально: Deep features через ResNet
from torchvision import models

# ResNet18
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final FC
model.eval().to(device)

# или ResNet50
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval().to(device)
```

```py
# Опционально: MediaPipe для дополнительных фичей
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh
```

Библиотеки для обработки:
```py
from skimage.metrics import structural_similarity as ssim
import librosa  # для аудио анализа
from sklearn.cluster import KMeans, DBSCAN
```

## video_pacing

### Модели:

```py
import clip

clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model.eval()
```

Библиотеки для обработки:
```py
from skimage.metrics import structural_similarity as ssim
from scipy.stats import entropy
```

## story_structure

### Модели:

```py
import clip
from sentence_transformers import SentenceTransformer

# CLIP для визуальных embeddings
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# SentenceTransformer для текстовых embeddings (субтитры)
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
# Альтернативы: 'all-mpnet-base-v2', 'paraphrase-MiniLM-L6-v2'
```

```py
# MediaPipe для детекции лиц
import mediapipe as mp

mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
```

Библиотеки для обработки:
```py
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import cv2  # для optical flow
```