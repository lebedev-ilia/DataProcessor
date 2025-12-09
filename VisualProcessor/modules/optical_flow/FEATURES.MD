# 🎬 1. Объекты, сцена, окружение

## 1.1. Детектирование объектов (расширенное)✅

### Модели для детектирования

1. Grounding DINO 1.5 / 2.0
2. OWL-ViT (Google)
3. YOLO-World (2024)
4. Florence-2 / OmniVL
5. InternVL 1.5

### 🔷 G. Полный финальный список фичей из 1.1

**Ядерные фичи (на каждый кадр и покадрово):**

- количество объектов каждого типа
- вероятность наличия конкретных объектов
- размеры объектов
- позиции объектов
- движение объектов
- длительность присутствия
- объектная плотность
- overlapping объектов
- разнообразие объектов
- редкость объектов
- entropy / Gini объектов
- object turnover rate
- object birth / object death events
- популярные категории (people, vehicles, animals, food)
- бренды и логотипы
- семантика объектов ("luxury", "danger", "cute")
- атрибуты объектов (цвет, стиль, размер)

## 1.2. Сценарное распознавание (Scene Classification)✅

### Модели для распознавания сцен

1. 🟦 Places365 (MIT)
2. 🟪 Places-Extended (Places450 / Places-MS)
3. 🟥 CLIP (Scene Embedding)
4. 🟧 Google OpenImages + Open Vocabulary Scene Tags
5. 🟩 VideoMAE / InternVideo Scene Features

### 🔥 Полный список фичей Scene Classification (итоговый)

#### 🎯 Статические

- top-1 scene class
- top-k scene distribution
- indoor/outdoor
- nature/urban
- time of day
- environment type
- aesthetic score
- luxury score
- atmosphere sentiment (cozy, scary, epic)

#### 🎯 Динамические

- number of scene changes
- average scene duration
- scene duration variance
- scene transition speed
- scene variety count
- scene entropy
- presence timeline per scene type

#### 🎯 Семантические

- genre scene (vlog, sports, travel, cooking)
- emotional scene characteristics
- professional/studio look
- activity-scene linking (kitchen → cooking)

#### 🎯 Геометрические

- openness
- clutter
- layout geometry
- depth cues

## 1.3. Контентные темы

### Модели для контентных тем

1. 🟥 CLIP (image-level semantic embedding)
2. 🟦 VideoCLIP / InternVid / VideoMAE
3. 🟧 LLaVA-Video / GPT-4o-mini-video / Flamingo
4. 🟩 BLIP-2 / Florence-2 / Gemini-Vision

### 🔷 A. Какие фичи входят в "контентные темы"?

#### 1) Тема / жанр / категория видео

Это главный набор признаков.

**Модели могут определять:**

- cooking / recipes
- fitness / workout
- beauty / makeup
- comedy
- news
- vlog
- gaming
- educational
- tech review
- unboxing
- DIY / crafts
- fashion
- nature / travel
- car-related
- pets
- music performance
- ASMR
- dance
- motivation
- personal story
- science
- movie scenes
- sports
- relationship advice
- business
- finance
- psychology
- prank
- meme video
- tutorial
- documentary style

> 💡 Можно держать 200–500 тем.

#### 2) Тематические кластеры

Используем CLIP для создания embedding 1536 → кластеризация (KMeans, HDBSCAN, Faiss).

**Получаем:**

- 50–200 тематических кластеров
- каждый кластер = тип контента (например, "cars at night", "cooking indoors", "nature drone footage")

**Фичи:**

- `cluster_id`
- `distance_to_cluster_centroid`
- `cluster_density`
- `cluster_rarity`

> 📌 Сильная фича для популярности: редкие темы → либо очень популярны, либо проваливаются.

#### 3) Темы высокого уровня (semantic attributes)

Через CLIP similarity к текстовым запросам:

- видео про путешествия
- видео про отношения
- мотивационное видео
- страшный/опасный контент
- эмоциональное видео
- расслабляющий контент
- романтический контент
- спорный контент
- острые конфликты
- смешное видео
- видео с обучением/пояснением

#### 4) Профессиональный контент vs любительский

Через CLIP-тексты:
- "professional video production"
- "cinematic video"
- "studio interview setup"

**Фича:**

- `professional_score` ∈ [0,1]

#### 5) Интенсивность действия/сюжета

**Модели определяют:**

- calm content
- high-intensity action
- dynamic editing
- slow-paced storytelling

#### 6) Narrative Arc (сюжетная дуга)

С помощью VideoCLIP-встраиваний детектируется:

- intro → setup
- conflict point
- climax
- resolution

**Фичи:**

- наличие/отсутствие сюжетной дуги
- её длина
- резкость конфликтов

#### 7) Emotional theme (эмоциональная тема)

> ⚠️ Не путать с эмоциями лица!

Это эмоциональное содержание видео:

- мотивация
- вдохновение
- страх
- расслабление
- тревога
- радость
- грусть
- ностальгичность

#### 8) Контент векторизованный: semantic embedding

**Самая сильная фича:**

- 📌 CLIP embedding 1536 dims на ключевых кадрах
  → агрегируем (mean, max, temporal transformer)
- 📌 VideoMAE embedding 1024–4096 dims
  → учитывает движение и глубину контента
- 📌 Общий fused semantic embedding 3000–10000 dims

> 💡 Твои будущие модели могут работать с этим как с универсальным "представлением видео".

### 🔷 B. Числовые фичи для обучения

1. `genre_probs` (вектор вероятностей по 200+ темам)
2. `scene semantic tags`
3. `emotional theme score`
4. `narrativity score`
5. `professional look score`
6. `dynamic intensity score`
7. `cluster ID` (из embedding)
8. `cluster rarity score`
9. `semantic entropy` (сколько разных смыслов в видео)
10. `similarity to viral content embeddings`
11. `similarity to trends`
12. `similarity to top videos of category`
13. `uniqueness` (насколько тема выделяется)
14. `controversy score`
15. `meme-ness score`
16. `trigger score` (есть ли шоковый, интересный или странный контент)

### 🔥 C. Особые фичи, важные для популярности

#### 1) Тема vs Тренд

Может быть сильная фича:

```
similarity(video, trending_category_X)
```

#### 2) Trigger Moments

Определяются LLM по сценам.

**Триггеры:**

- аварии
- прыжки
- падения
- конфликты
- красивые виды
- животные
- неожиданные переходы
- раскрытие секрета
- выводы в конце

### 🔷 D. Контентные темы, которые особенно важны для YouTube prediction

**Топ категорий, сильно влияющих:**

- cooking
- DIY
- gaming
- beauty
- finance
- fitness
- travel
- cars
- music
- children content
- motivation
- funny
- conflict/drama
- controversial
- life hacks

# 🧑‍🦰 2. Люди, лица, эмоции и поведение

## 🎯 2.1. Детализация лиц — полный профессиональный набор фичей✅

---

### 🔵 2.1.1 Геометрия лица

**Фичи:**

- `face_bbox_area` — площадь бокса
- `face_relative_size` — площадь относительно кадра
- `face_box_ratio` (w/h)
- `face_bbox_position` (cx, cy)
- `face_dist_from_center`
- `face_rotation_in_frame` (относительное поворот относительно камеры)
- `aspect_ratio_stability` — стабильность размеров по времени
- `face_shape_vector` (landmarks → PCA)
- `jaw_width`, `cheekbone_width`, `forehead_height`

**Как получить:**

Использовать:
- InsightFace (landmarks 106)
- MediaPipe (468 landmarks) → ещё точнее

```python
face = app.get(frame)
landmarks = face.landmark_3d_68
```

> 💡 Можно извлечь до 400 геометрических метрик.

---
### 🔵 2.1.2 Поза лица (Head Pose Estimation)

**Фичи:**

- `yaw`, `pitch`, `roll` — углы
- `avg_pose_angle`
- `head_pose_variability`
- `looking_direction_vector`
- `pose_stability_score` (how stable head is)
- `head_turn_frequency`
- `attention_to_camera_ratio`

**Как получить:**

- SolvePnP по 68 landmark points
- Или готовые модели:
  - Hopenet
  - SynergyNet

### 🔵 2.1.3 Качество изображения лица

Это важно для понимания:
- профессиональная съёмка или нет
- блюр
- хейло
- плохое освещение
- недофокус

**Фичи:**

- `face_blur_score` (можно считать LAPLACIAN variance)
- `sharpness_score`
- `texture_quality`
- `focus_metric`
- `noise_level`
- `motion_blur_score` (по edge-streaks)
- `artifact_score` (compression artifacts)
- `resolution_of_face` (px → водит к качеству объектива)
- `face_visibility_ratio`
- `occlusion_score`

---
### 🔵 2.1.4 Свет и цвет лица

**Фичи:**

- `average_lighting_on_face` (mean pixel value)
- `light_uniformity_score` (разница освещения левой/правой части)
- `face_contrast`
- `white_balance_shift`
- `skin_color_vector` (Lab/RGB)
- `highlight_intensity`
- `shadow_depth`
- `skin_tone_index` (Fitzpatrick classification)

**Как получить:**

Разобрать лицо по маске из landmarks:
- left-half
- right-half
- forehead
- cheeks
- chin

### 🔵 2.1.5 Состояние кожи / атрибуты / косметика

**Фичи:**

- `makeup_presence_prob`
- `lipstick_intensity`
- `eye_shadow_prob`
- `skin_smoothness`
- `skin_defect_count` (acne, wrinkles)
- `beard_prob`
- `mustache_prob`
- `face_hair_density`
- `eyebrow_shape_vector`
- `eyelid_openness`

**Модели:**

FaceParsing (BiSeNet) для сегментации:
- глаза, губы, волосы, кожа, брови, шея, одежда

---
### 🔵 2.1.6 Очки / Маски / Аксессуары

**Фичи:**

- `glasses_prob`
- `sunglasses_prob`
- `mask_prob` (medical mask)
- `hat_prob`
- `earrings_presence`
- `jewelry_probability`
- `hair_accessories`
- `helmet_prob`

> Получаем через сегментацию (FaceParsing) или YOLO-face-extended.

### 🔵 2.1.7 Глаза и взгляд

> ⚠️ Очень важная часть!

**Фичи:**

- `eye_opening_ratio` (left/right)
- `blink_rate`
- `blink_intensity`
- `gaze_vector` (3D direction)
- `gaze_at_camera_prob`
- `attention_score`
- `eye_redness_prob`
- `iris_position` (relative)

**Модели:**

- MediaPipe Iris
- Gaze360
- DECA (3D face reconstruction)

---
### 🔵 2.1.8 Движение лица

**Фичи:**

- `face_speed` (bbox center displacement)
- `face_acceleration`
- `micro-expression_rate`
- `jaw_movement_intensity`
- `eyebrows_motion_score`
- `mouth_motion_score` (lips)
- `head_motion_energy`
- `talking_motion_score` (по губам)

**Модели:**

- 3D landmarks + optical flow
- или DECA

### 🔵 2.1.9 Структурные особенности лица

**Можно использовать:**

- 3DMM (3D Morphable Model)
- DECA / EMOCA models — современные SOTA

**Фичи:**

- 3D face mesh → 100–300 параметров
- `identity shape vector`
- `expression vector` (50+ dims)
- `jaw pose vector`
- `eye pose vector`
- `mouth shape params`
- `face symmetry score`
- `face uniqueness score`

> 💡 Это уже уровень студийной анимации — проф. аналитикам понравится.

---
### 🔵 2.1.10 Профессиональные (редкие) фичи

1. **Face Quality Score (NIMA)**
   - Оценивает красивость/качество фото/видео.

2. **Perceived attractiveness score**
   - Есть модели, но они спорные; можно делать скрытый feature-vector.

3. **Emotion intensity (expressiveness)**

4. **Lip reading features**
   - Контекст губ → можно предсказывать speech activity.

5. **Fatigue score**
   - глаза
   - поза
   - скорость движений
   - Используется в автоиндустрии.

6. **Engagement level**
   - Индекс вовлечённости:
     - взгляд в камеру
     - открытая поза
     - улыбка
     - активная мимика

## 2.2. Эмоции из лица✅

### ✅ КАТЕГОРИИ ФИЧЕЙ ИЗ ЭМОЦИЙ

#### 2.2.1. Классические эмоции (Ekman 7 basic)

**Стандартный набор:**

- angry
- disgust
- fear
- happy
- sad
- surprised
- neutral

**Как извлечь:**

- 🔹 DeepFace → `DeepFace.analyze(img, actions=['emotion'])`
- 🔹 FER+ (переобученный на эмоции)
- 🔹 Google MediaPipe FaceLandmark + классик-классификатор

#### 2.2.2. Динамика эмоций

> 💡 Это очень мощные фичи, т.к. важна не сама эмоция, а её переходы.

**Фичи:**

- скорость изменения эмоции
- амплитуда всплесков
- эмоциональная турбулентность
- стабильность эмоций
- среднее время удержания эмоции
- резкие эмоциональные переключения

**Пример фичей:**

- `emotion_change_per_second`
- `emotion_peak_frequency`
- `emotion_state_duration`
- `emotion_entropy`

**Как извлечь:**

1. Обработать эмоции по каждому кадру (или раз в 5 кадров).
2. Получить временной ряд эмоций.
3. Посчитать:
   - производную (скорость изменений)
   - вариацию
   - энтропию
   - количество смен эмоций

> → Всё это легко сделать в pandas + numpy.

#### 2.2.3. Arousal / Valence (2D эмоциональное пространство)

Это лучше, чем Ekman-модель, потому что:

- даёт непрерывные значения,
- работает на любых культурах,
- полезно для ML.

**Фичи:**

- `valence` (позитивность)
- `arousal` (возбуждение)

> 🎯 это лучшие предикторы для вовлечённости зрителя.

**Как извлечь:**

- Модель AffectNet (отличный датасет)
- Модели Affectiva (коммерческие)
- Есть открытые веса: "emotion2vec", "affectnet_resnet50", "DAN (Dual Attention Network)".

#### 2.2.4. Микроэмоции (micro-expressions)

Это резкие эмоции длительностью 0.03–0.5 секунды.

**Их можно использовать для анализа:**

- искренности
- реакции на события
- напряжения

**Фичи:**

- `microexpressions_count`
- `microexpression_duration`
- `microexpression_type`

**Как извлечь:**

**Модели:**

- CASME II dataset models
- SAMML micro-expression classifier
- Dual-Stream CNN for Microexpression Recognition

#### 2.2.5. Физиологические сигналы (из эмоций)

**Можно получить:**

- уровень стресса
- уровень уверенности
- уровень нервозности

**Как?**

Через сочетание:

- частоты микромимики,
- расширение/сужение глаз,
- частоту моргания,
- движения рта,
- напряжение мышц лица.

**Фичи:**

- `stress_level_score`
- `confidence_face_score`
- `tension_face_index`

**Модели:**

- OpenFace 2.0 (AU → эмоциональные индексы)
- FaceActionUnits models на BP4D dataset

#### 2.2.6. Индивидуальность выражения эмоций

**Можно анализировать:**

- насколько автор "эмоциональный"
- как он выражает разные эмоции
- стилевые паттерны

**Фичи:**

- `emotional_intensity_baseline`
- `emotional_style_vector`
- `expressivity_index`

> 💡 Это полезно для аналитиков: "автор очень выразителен, это повышает удержание".

#### 2.2.7. Асимметрия лица

**Используется для:**

- оценки естественности
- оценки искренности
- оценки уровня напряжения

**Фичи:**

- разница активности AU между левой/правой частью лица
- разница поднятия бровей
- разница улыбки

**Технически:**

- OpenFace выдаёт AU для каждой стороны лица
- делишь левую часть на правую → получаешь асимметрию

### 🛠️ КАК ПОЛУЧИТЬ ВСЁ ЭТО ИЗ ВИДЕО — ПАЙПЛАЙН

Ниже готовая схема, которую ты можешь прямо применять.

#### ✔ Шаг 1 — Детекция лица

**Использовать:**

- RetinaFace (лучший)
- Или YOLO-NAS-Face
- Или FaceX-Zoo (быстрый)

**Получаешь:**

- координаты лица
- ключевые точки

#### ✔ Шаг 2 — Трекинг лица по кадрам

Чтобы не детектить каждый кадр:

- SORT / DeepSORT
- ByteTrack
- Или Face Recognition + IoU matching

**Получаешь:**

- стабильные face-id
- траекторию лица

#### ✔ Шаг 3 — Извлечение эмоций каждую N-ую секунду

**Использовать:**

🔥 **Рекомендуем:**

- AffectNet-based emotion classifiers (valence/arousal)
- DeepFace (удобно тестировать)
- OpenFace 2.0 (AU → эмоции + физиология)

#### ✔ Шаг 4 — Извлечение Action Units (AU)

**OpenFace даёт:**

- AU1 .. AU45
- интенсивность и наличие
- скорость изменения AU

> 💡 AU — это лучший источник глубокой эмоциональной информации.

#### ✔ Шаг 5 — Построение временных рядов

Для каждого лица:

- эмоции
- valence/arousal
- AU
- микроэмоции

#### ✔ Шаг 6 — Агрегирование фичей

**Ты получаешь:**

- средние значения
- максимумы
- динамику
- энтропию
- распределения
- пики
- продолжительность
- вариативность

> 💡 В итоге для одного лица → 200–400 фичей.

#### ✔ Шаг 7 — Для нескольких лиц

Для видео:

- по каждому лицу
- плюс глобальные фичи:
  - средняя эмоция кадра
  - эмоция "главного героя"
  - эмоции публики
  - согласованность эмоций всех лиц

## 2.3. Поведение людей (behavioral analytics)✅

### 🧠 2.3. Поведение людей — Полный разбор

#### 2.3.1. Жесты рук (Hand Gestures)

**Фичи:**

- количество жестов в минуту
- амплитуда движения рук
- частота поднятия рук
- типы жестов:
  - указание рукой
  - раскрытые ладони
  - руки в боки
  - "self-touch gestures" (поглаживание, почёсывание — индикатор стресса)
- симметрия движений рук
- избыточная жестикуляция (over-gesturing)
- отсутствие жестикуляции (under-gesturing)

> ⚡ Польза: эмоциональность, уверенность, уровень энергии.

#### 2.3.2. Body Pose & Body Language (язык тела)

**Фичи:**

- поза тела (стоя, сидя, наклон, наклон вперёд)
- открытая/закрытая поза
- доминантность позы (power pose)
- напряжённость (rigidity)
- расслабленность (relaxed posture)
- изменения позы:
  - смена позы
  - скорость смены
  - внезапные сдвиги (тревожность)
- баланс тела (center of mass)

> ⚡ Польза: уверенность → влияет на удержание зрителя.

#### 2.3.3. Head Pose (положение головы)

**Фичи:**

- наклон головы (roll, pitch, yaw)
- частота наклонов
- микродвижения головы
- кивки (head nods)
- покачивание головой "нет"
- резкие повороты головы
- скорость поворота

> ⚡ Польза: вовлечённость, реакция на события, уровень нервозности.

#### 2.3.4. Eye Behavior (поведение глаз)

**Фичи:**

- направление взгляда:
  - в камеру
  - в сторону
  - вниз (часто → неуверенность)
- dwell-time (сколько времени смотришь на объект)
- saccades (быстрые движения глаз)
- частота морганий
- расширение зрачков (pupil dilation — признак эмоций)

> ⚡ Польза: уровень вовлечённости, стресс, искренность.

#### 2.3.5. Speech-driven behavior (если человек говорит на видео)

**Фичи:**

- синхронность движений губ со звуком
- частота открывания рта
- зоны артикуляции
- несинхронность → признак монтажа / фейков

> ⚡ Полезно для тикток/ютуба, где много talking-head.

#### 2.3.6. Movement Patterns (структура движения человека)

**Фичи:**

- скорость движения тела
- ускорение движения
- smoothness (плавность)
- chaotic index (хаотичность)
- trajectory entropy
- distance traveled by landmarks

> ⚡ Это сильно коррелирует с удержанием внимания зрителя.

#### 2.3.7. Interaction behavior (взаимодействие с окружением и объектами)

**Фичи:**

- взаимодействие с предметами (телефон, еда, микрофон)
- взаимодействие с другими людьми
- дистанция между людьми (proxemics)
- передача объектов
- слежение за объектом глазами/головой
- использование реквизита

> ⚡ Полезно для аналитики студий, интервью, подкастов.

#### 2.3.8. Engagement Index (индекс вовлечённости человека)

**Строится из:**

- движений головы
- микромимики
- зрительного контакта с камерой
- активности жестов
- скорости реакции на события

**Фичи:**

- `engagement_score`
- `engagement_variation`
- `engagement_peaks`
- `engagement_consistency`

> ⚡ Один из лучших предикторов востребованности и удержания зрителя.

#### 2.3.9. Confidence / Dominance Index (уверенность / доминантность)

**Строится из:**

- открытой позы
- наклона головы
- жестов
- положения плеч
- тонуса тела

**Фичи:**

- `confidence_score`
- `dominance_score`
- `confidence_variability`
- `confidence_peak_moments`

> ⚡ Важнейшее для блогеров, спикеров, стримеров.

#### 2.3.10. Signs of stress / anxiety (стресс, тревожность)

**Фичи:**

- частое моргание
- self-touch
- микрожесты напряжения (кривление губ)
- закрытая поза
- сглатывания
- fidgeting (ёрзание)
- несинхронные движения

> ⚡ Самая мощная фича для понимания эмоционального состояния автора.

### 🛠️ КАК ЭТО ВСЕ ПОЛУЧИТЬ ИЗ ВИДЕО (практический пайплайн)

Ниже — простая дорожная карта, по которой можно получить все выше перечисленное.

#### ✔ 1. Детекция и трекинг человека

**Использовать:**

- YOLOv8 / RT-DETR / YOLO-NAS → детекция человека
- ByteTrack → трекинг

#### ✔ 2. Pose Estimation

**Для позы:**

- MediaPipe Pose (быстро, стабильно)
- OpenPose (мощно, но медленно)
- MMPose (HRNet, ViTPose) — лучшее качество
- MoveNet — супербыстро

> Получаем ключевые точки тела (33–136 точек).

#### ✔ 3. Hand Pose

- MediaPipe Hands
- MMPose Hand Keypoints

> Фичи → жесты, hand-tracking.

#### ✔ 4. Head Pose Estimation

**Используем:**

- MediaPipe Face Mesh
- SolvePnP по 3D-landmarks
- Или OpenFace "head pose"

#### ✔ 5. Eye tracking

**MediaPipe Face Mesh → даёт:**

- pupils
- iris landmarks

**Далее считаем:**

- направление взгляда
- моргания
- saccades

#### ✔ 6. Behavior classifier (дополнительные модели)

**Есть готовые action recognition модели:**

- SlowFast (Meta AI)
- I3D
- X3D
- VideoMAE

**Можно классифицировать:**

- gestures
- attention
- fidgeting
- walking / running / sitting

#### ✔ 7. Агрегировать всё в feature vectors

**Например:**

- частота жестов → per 10 seconds
- поза тела → дискретизация + энтропия
- направление взгляда → проценты по категориям
- скорость движения → в единицах px/sec
- `stress_score` → модель на AU + жестах
- `confidence_score` → модель на позе + жестах

# 3. Движение, активность, экшн-зоны

## 3.1. Optical Flow Statistics

🔹 3.1.1. Базовые статистики скорости и направления

Оптический поток даёт поля:

- `dx` (горизонтальное движение)
- `dy` (вертикальное движение)

Из этого можно получить:

**Скорости:**

- среднее значение скорости: `mean_flow_magnitude`
- медиана скорости
- максимальная скорость
- стандартное отклонение скорости
- распределение скоростей по кадру

**Направления:**

- среднее направление движения
- доминирующее направление
- энтропия направлений (как хаотично движется кадр)

🔹 3.1.2. Пространственные агрегаты

Поделить кадр на сетку (например 4×4 или 8×8) и получить:

- `flow_mean_grid[row, col]`
- `flow_std_grid[row, col]`
- `flow_max_grid[row, col]`
- `flow_direction_hist_grid[row, col]`

> 💡 Это сильнейшая фича, потому что движение чаще всего локальное (например — лицо, руки, абстрактный объект, текст на экране).

🔹 3.1.3. Temporal Flow Statistics

Динамики потока по времени:

- скорость изменения потока
- "взрывы" активности (motion peaks)
- variance of flow over time
- derivate of flow: acceleration flow

**Фичи:**

- `flow_change_rate`
- `flow_acceleration`
- `flow_turbulence`
- `flow_energy`

🔹 3.1.4. Motion Smoothness / Jerkiness

Даёт очень ценные метрики:

- `smoothness_index`
- `jerkiness_index`
- `flow_temporal_entropy`
- `movement_stability`

> 💡 Например: плохая стабилизация видео → низкое удержание. Слишком резкие движения → болезненная динамика.

🔹 3.1.5. Motion Type Classification

По статистике opt flow можно даже классифицировать:

- статичный кадр
- медленное движение
- панорама
- зум
- трясущаяся камера
- активная сцена
- резкие смены планов

**Фичи:**

- `motion_class` (по модели)
- `motion_intensity_score`
- `pan_probability`
- `zoom_probability`
- `shake_index`

🔹 3.1.6. Motion Energy (старый, но крутой признак)

Ещё в 90-х в CV ввели метрику:

- MEI/MEP (motion energy image / motion history image)

**Суть:**

- агрегируешь движение за T секунд в единое изображение
- считаешь простые статистики

**Фичи:**

- `MEI_total_energy`
- `MHI_contrast`
- `MHI_entropy`

> 💡 Используется до сих пор в индустрии спортсменского анализа, охраны и robotics.

🔹 3.1.7. Motion Clusters

Можно кластеризовать векторы движения:

- группировать пиксели по направлению/скорости
- находить "центры активности"

**Фичи:**

- `num_motion_clusters`
- `cluster_size_distribution`
- `largest_motion_cluster_area`
- `cluster_direction_stats`

> 💡 Это круто, когда есть несколько источников движения, например: человек + экран, несколько людей, движение в фоне.

🔹 3.1.8. Foreground vs Background Motion

Разделение движения:

- `foreground_motion_energy`
- `background_motion_energy`
- `ratio_foreground_background_flow`

> 💡 Эта фича идеально показывает: активность героя, статичность задника, динамичность сцены.

🔥 ИТОГО: из одного видеоролика можно получить 80–150+ фичей только из оптического потока.
🛠 КАК ЭТО ИЗВЛЕЧЬ (полный пайплайн)

Есть 3 способа.

#### ✔ Способ 1: OpenCV Farneback Optical Flow (CPU/GPU)

```python
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, next_gray,
    None,
    0.5, 3, 15, 3, 5, 1.2, 0
)
```

Далее:

```python
dx = flow[..., 0]
dy = flow[..., 1]
magnitude = np.sqrt(dx**2 + dy**2)
direction = np.arctan2(dy, dx)
```

#### ✔ Способ 2: GPU: NVIDIA Optical Flow SDK (самый точный)

Использовать:

```python
cv2.cuda_NvidiaOpticalFlow_1_0
```

> Работает на любом RTX. Это значительно лучше, чем Farneback.

✔ Способ 3: Deep Optical Flow Models (лучший вариант)

Если тебе нужен максимальный результат, используй:

Модели:

RAFT (2020) — state of the art, очень точная

LiteFlowNet2 — лёгкая и быстрая

PWC-Net — быстрый

FlowNet2 — хорошая точность + скорость

Все доступны в PyTorch.

**Пример RAFT:**

```python
from raft import RAFT
model = RAFT().cuda()
flow = model(prev_frame, next_frame)
```

#### ✔ Дальше — считаешь статистики

После получения flow (dx, dy):

**Magnitude:**

```python
mag = np.sqrt(dx**2 + dy**2)
mean_mag = mag.mean()
std_mag = mag.std()
max_mag = mag.max()
```

**Direction histogram:**

```python
hist, _ = np.histogram(np.arctan2(dy, dx), bins=36)
```

**Grid:**

Разбить на 8×8:

```python
grid_stats = []
for i in range(8):
    for j in range(8):
        patch = mag[i*h:(i+1)*h, j*w:(j+1)*w]
        grid_stats.append([patch.mean(), patch.std(), patch.max()])
```

#### ⚙ Автоматизация

Ты можешь вызвать всё это в твоём feature-extractor как модуль:

```
OpticalFlowExtractor
├── compute_flow (model=RAFT)
├── compute_stats
├── compute_grid_stats
├── compute_motion_energy
├── compute_turbulence
└── output_feature_vector()
```


Модуль	Статус	Что сделано	Что дальше
1. Базовые статистики скорости и направления	✅ Готов	Получаем поле потока (векторы dx, dy)	Анализ статистик по кадру/последовательности
2. Пространственные агрегаты	🟡 Частично	Уменьшение разрешения (max_dim)	Кластеризация по регионам, анализ паттернов
3. Динамика потока по времени	🟡 Частично	Обработка последовательностей кадров	Временные ряды, анализ трендов, периодичностей
4. Motion Smoothness / Jerkiness	🔴 Не начат	—	Расчет производных потока (ускорение, рывок)
5. Motion Type Classification	🟡 Задел есть	Визуализация цветом показывает типы	Автоматическая классификация сцен
6. MEI/MHI (Motion Energy/History)	🔴 Не начат	—	Накопление движения, energy images
7. Motion Clusters	🔴 Не начат	—	Кластеризация векторов потока
8. Foreground vs Background Motion	🟡 Задел есть	Единое поле потока	Разделение глобального/локального движения


Фаза 1: Базовый анализ (ТЕКУЩИЙ ЭТАП)
text
✅ Захват видео → RAFT поток → Визуализация
✅ Сохранение сырых данных потока
Фаза 2: Статистический анализ (СЛЕДУЮЩИЙ ШАГ)
text
1. Базовые статистики по каждому кадру
2. Пространственные агрегаты (регионы интереса)
3. Динамика по времени (тренды)
Фаза 3: Продвинутый анализ
text
4. Smoothness/Jerkiness (оценка качества)
5. Классификация типов движения
6. MEI/MHI для компактного представления
Фаза 4: Интеграционный анализ
text
7. Кластеризация движения
8. Разделение переднего/заднего плана