# 🎬 1. Объекты, сцена, окружение

## 1.1. Детектирование объектов         🟩🟩🟥

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

## 1.2. Сценарное распознавание         🟩🟩🟥

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

## 1.3. Контентные темы                 (не входит в Processor) 

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

## 🎯 2.1. Детализация лиц              🟩🟩🟥

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

## 2.2. Эмоции из лица                  🟩🟩🟥

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

## 2.3. Поведение людей (behavioral)    🟩🟩🟥

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

## 3.1. Optical Flow Statistics         🟩🟩🟥

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

### 📌 Что можно извлечь

#### A) Типы движения камеры

Определяются через анализ потока (optical flow) + SLAM/VO алгоритмы:

- `pan_left` / `pan_right` — горизонтальный поворот камеры
- `tilt_up` / `tilt_down` — вертикальный поворот
- `roll` — вращение камеры вокруг оси
- `push_in` / `push_out` — движение вперёд/назад
- `truck_left` / `truck_right` — камера движется вбок
- `pedestal_up` / `pedestal_down` — камера движется вверх/вниз
- `static` — камера недвижима

**→ Фичи:**

- `cam_pan_ratio`, `cam_tilt_ratio`, `cam_zoom_ratio`, `cam_dolly_ratio`

#### B) Стабильность камеры

> 💡 Это ключевая фича.

**Считается так:**

- выделяем фоновый optical flow
- считаем его variance / jerk / jitter

**Фичи:**

- `camera_shakiness_mean`
- `camera_shakiness_std`
- `camera_shakiness_max`
- `camera_shakiness_segments_count`
- `handheld_probability` (0–1)
- `tripod_probability`
- `gimbal_probability`

> Модель: можно обучить простой классификатор на базе стандартов видео.

#### C) Zoom detection

Можно извлечь по изменению масштаба фона:

- `zoom_in_count`
- `zoom_out_count`
- `zoom_speed_mean`
- `zoom_smoothness`

> 💡 Очень полезно, потому что зум влогеров → показатель эмоциональной динамики.

#### D) Camera motion smoothness

Определяется из первого и второго производного потоков:

- `motion_smoothness_mean`
- `motion_smoothness_std`
- `camera_turn_speed_mean` (скорость поворота)
- `acceleration_of_camera_motion`

#### E) Camera trajectory reconstruction

Если нужно advanced:

**Используем:**

- ORB-SLAM / DSO / COLMAP для построения траектории камеры

**Можно получить:**

- 3D длину движения камеры
- число резких разворотов
- длины сегментов стабильности
- амплитуду движения
- «chaos index» — насколько непредсказуемо движется камера

#### F) Camera Motion Styles

Можно классифицировать на стили съёмки:

- Static interview
- Handheld vlog
- Cinematic smooth gimbal
- Action cam (GoPro)
- Drone footage
- Screen capture / fixed UI
- Gaming (FPS/TPP)

**Фичи:**

- `style_static_probability`
- `style_handheld_probability`
- `style_drone_probability`
- `style_gopro_probability`
- `style_cinematic_probability`

### 📌 Как это вычислять (конкретно)

#### 1️⃣ Optical Flow (RAFT или FlowNet2)

Для каждого кадра:

```python
flow = raft_model(frame_t, frame_t1)
```

Оттуда:
- среднее направление фона
- средняя величина движения
- разделение foreground/background через mask R-CNN или SAM

#### 2️⃣ Определение вращения/движений камеры

Берём оптический поток фона → строим модель аффинного преобразования:

```python
affine_matrix = cv2.estimateAffinePartial2D(src_points, dst_points)
```

Потом:
- translation → движения (truck, pedestal)
- rotation → pan/tilt/roll
- scale → zoom

#### 3️⃣ Стабильность

```python
shakiness = np.var(flow_background)
```

#### 4️⃣ Классификация "стиля камеры"

Обучаешь маленькую CNN/ViT:
- вход: 32 кадра optical flow maps
- выход: стиль

### 📌 Какие итоговые фичи стоит сохранять (итоговый список)

#### 📍 Motion magnitude (6 шт)

- `motion_mean`
- `motion_std`
- `motion_max`
- `motion_min`
- `motion_energy`
- `motion_entropy`

#### 📍 Camera shake (5 шт)

- `shake_mean`
- `shake_std`
- `shake_max`
- `shake_jitter_index`
- `handheld_likelihood`

#### 📍 Camera movement types (8 шт)

- `pan_ratio`
- `tilt_ratio`
- `roll_ratio`
- `zoom_ratio`
- `dolly_ratio`
- `truck_ratio`
- `pedestal_ratio`
- `static_ratio`

#### 📍 Camera speed & smoothness (4 шт)

- `rotation_speed`
- `zoom_speed`
- `movement_smoothness`
- `acceleration`

#### 📍 Camera style probabilities (5–7 шт)

- `style_handheld`
- `style_tripod`
- `style_cinematic`
- `style_drone`
- `style_action_cam`
- `style_screencap`
- `style_gaming`

## 3.2. Action Recognition              🟩🟥🟥

🔹 3.2.1. High-level Actions (высокоуровневые действия)

Это то, что современные модели хорошо распознают:

**Люди:**

- walking
- running
- sitting
- standing
- jumping
- dancing
- cooking
- eating
- drinking
- reading
- writing
- talking
- shouting
- laughing
- crying

**Спорт:**

- swimming
- skiing
- boxing
- cycling
- weightlifting
- yoga
- basketball
- football
- tennis

**Ежедневные действия:**

- cleaning
- washing
- cutting
- opening
- closing
- moving object

**Технические действия:**

- working on computer
- adjusting camera
- using phone

🔹 3.2.2. Multi-person actions

Если несколько людей:

- group walking
- fighting
- hugging
- handshakes
- arguing
- teaching
- collaborating
- crowds running
- dancing together

🔹 3.2.3. Fine-grained actions (тонкие действия)

Это очень важно в роликах типа TikTok/YouTube Short:

- face touch
- hair touch
- pointing
- waving
- nodding head
- shrugging
- clapping
- lip-sync
- reacting to screen (reaction videos)
- scrolling phone

🔹 3.2.4. Scene activity type (уровень активности сцены)

На уровне сцены:

- high action intensity
- low action intensity
- chaotic motion
- static

**Фичи:**

- `scene_activity_score`
- `action_entropy`
- `action_change_rate`
- `dominant_action`
- `num_unique_actions`

🔹 3.2.5. Temporal Action Statistics

**Фичи:**

- количество действий
- время каждого действия
- частота смены действий
- разнообразие действий
- равномерность действий
- "action bursts" – всплески активности

🔹 3.2.6. Action Complexity (сложность действия)

**Фичи:**

- `complex_action_score`
- `coordination_level`
- `action_precision`

> 💡 Например, танцы → высокая сложность.

🔹 3.2.7. Action Planning & Intent

Современные модели типа VideoMAE, InternVideo, TimeSformer, VideoGPT могут предсказывать:

- future action prediction
- intent (готовится прыгнуть, готовится бежать)

**Фичи:**

- `predicted_action`
- `action_preparation_time`

🧠 Всего: 200+ фичей из Action Recognition.
🛠 КАК ЭТО ВСЁ ИЗВЛЕЧЬ?

Теперь практический пайплайн — какие модели использовать, как именно.

✔ Шаг 1. Выбор модели Action Recognition

🔥 **ЛУЧШИЕ МОДЕЛИ НА 2025 год:**

1) **VideoMAE / VideoMAEv2 / VideoMAE-XL** — лучший вариант
   - основана на masked autoencoding
   - точность на Kinetics 400/700 — SOTA
   - работает на потребных GPU
   - есть веса на HuggingFace
   - можно fine-tune под свои действия

2) **TimeSformer** (Transformer для видео)
   - первый чистый видеотрансформер
   - медленнее VideoMAE, но хорош

3) **SlowFast** (Meta AI)
   - классика
   - быстрый
   - отлично работает для реальных видео

4) **X3D** (эффективный, лёгкий)
   - быстрый
   - идеально для веб/мобильных приложений

5) **I3D** (Inflated 3D ConvNet)
   - самый стабильный baseline

✔ Шаг 2 — классы действий (400/700/1000 категорий)

**Базы:**

- Kinetics-400
- Kinetics-600
- Kinetics-700
- Something-Something V2
- AViD dataset

**Эти модели выдают:**

- probabilities for each action
- top-1, top-5 actions

✔ Шаг 3 — периодичность

**Обрабатывать:**

- каждые 1–2 секунды
- или каждые 32–64 кадра

**Получаем:**

- временной ряд действий → `action(t)`

✔ Шаг 4 — Построение фичей

**Из raw output (probabilities):**

- mean action probability
- max action probability
- entropy of actions
- variation of action probabilities

**Из action sequence:**

- dominante action
- action transitions
- duration of each action
- number of unique actions
- action cluster count
- average action complexity

✔ Шаг 5 — Агрегация в финальные фичи

**Например:**

- `action_diversity = number_of_unique_actions / duration`
- `action_intensity = average_motion_related_actions_score`
- `action_change_rate = transitions / time`
- `action_stability = longest_duration_of_one_action`

✔ Шаг 6 — Multi-person action recognition

Если в кадре несколько людей:

- тречим людей (ByteTrack)
- для каждого человека запускаем VideoMAE
- объединяем статистики

### 🧩 ГОТОВАЯ АРХИТЕКТУРА МОДУЛЯ

```
ActionRecognitionModule:
├── FrameSampler
├── PersonTracker
├── ActionModel (VideoMAE)
├── ActionPostProcessor
├── FeatureExtractor
└── Output: 200+ action features
```

# 🌈 4. Стиль и кинематографические признаки

## 4.1. Цвет и свет                     🟩🟥🟥

🌈 A) Цветовые статистики (базовые, но расширенные)

Из каждого кадра → усредняем по видео:

- `color_mean_r`, `color_mean_g`, `color_mean_b`
- `color_std_r`, `color_std_g`, `color_std_b`
- `color_min_r/g/b`, `color_max_r/g/b`
- `color_skewness_r/g/b` (асимметрия)
- `color_kurtosis_r/g/b`

> ➡️ Позволяет анализировать баланс белого, экспозицию, общее освещение.

🎨 B) Цветовые пространства (LAB, HSV, YUV, HSL)

В LAB можно оценить:

- L-channel → яркость
- a/b → тепло/холод, зелёный–красный, синий–жёлтый

**Фичи:**

- `hue_mean`, `hue_std`, `hue_entropy`
- `saturation_mean`, `saturation_std`
- `value_mean`, `value_std`
- `lab_L_mean`, `lab_L_contrast`, `lab_L_dynamic_range`

🌅 C) Цветовые гармонии и composition features

Из Hue histogram можно выявить:

- `dominant_color` (основной оттенок)
- `secondary_color`, `tertiary_color`
- `palette_size` (разнообразие палитры)
- `colorfulness_index`
- `warm_vs_cold_ratio` (отношение тёплых и холодных тонов)
- `skin_tone_ratio` (важно для блогов/интервью)
- `color_palette_entropy`

Можно даже определить:

- цветовые гармонии: complementary, analogous, triadic, split-complementary
- фичи: `color_harmony_type` (one-hot или probability)

### 💡 D) Освещение (Lighting features)

> 💡 Это один из самых важных блоков.

#### 1) Глобальная яркость

- `brightness_mean`
- `brightness_std`
- `brightness_entropy`
- `overexposed_pixels_ratio`
- `underexposed_pixels_ratio`

#### 2) Контраст

- `global_contrast`
- `local_contrast_mean`
- `local_contrast_std`
- `contrast_entropy`

#### 3) Dynamic Range (HDR score)

Используем log-scale luminance:

- `dynamic_range_db`
- `highlight_clipping_ratio`
- `shadow_clipping_ratio`

#### 4) Lighting Uniformity

Изучаем распределение яркости по сетке:

- `lighting_uniformity_index`
- `center_brightness`
- `corner_brightness`
- `vignetting_score`

#### 5) Directional Lighting

Определяем направление света по теням:

- `light_direction_angle`
- `light_source_count_estimate`
- `soft_light_probability`
- `hard_light_probability`

### 🎥 E) Кинематографические цветовые стили

Можем определить стиль цветокоррекции:

- Teal & Orange
- Natural
- High saturation
- Film look (Kodak, Fuji)
- Desaturated / moody
- Vintage / warm retro
- TikTok filters style

**Фичи:**

- `style_teal_orange_prob`
- `style_film_prob`
- `style_desaturated_prob`
- `style_hyper_saturated_prob`
- `style_vintage_prob`
- `style_tiktok_prob`

> Классифицировать можно через ResNet/ViT обученный на 50–100k примеров LUT-стилей.

### 🔥 F) Motion + Lighting (важно!)

Многие вирусные ролики используют эффект:

- быстрых изменений яркости
- ритмические вспышки
- стробоскопические переходы

**Фичи:**

- `brightness_change_speed`
- `color_change_speed`
- `scene_flicker_intensity`
- `flash_events_count`
- `color_transition_variance`

🎯 G) Aesthetic & Cinematic Scores

**Есть готовые модели:**

- NIMA (Neural Image Assessment)
- LAION aesthetic predictor

**Можно добавить:**

- `aesthetic_score_mean`
- `aesthetic_score_std`
- `cinematic_lighting_score`
- `professional_look_score`

> 👉 Эти признаки особенно любят профессиональные аналитики.

🧠 H) Temporal Color Patterns (динамика)

Можно использовать 1D temporal CNN для последовательности значений Hue/Saturation:

- `color_stability`
- `color_temporal_entropy`
- `color_pattern_periodicity`
- `scene_color_shift_speed`

### 📌 Итоговые группы фичей по 4.1

| Группа | Пример фичей |
|--------|--------------|
| Цветовые статистики | mean/std/skew/kurt |
| HSV/LAB/YUV показатели | hue_mean, saturation_mean |
| Палитра и гармонии | dominant_color, color_entropy |
| Освещение | brightness_mean, dynamic_range |
| Контраст | global/local contrast |
| Направление света | light_direction_angle |
| Стили цветокоррекции | teal_orange_prob |
| Эстетичность | aesthetic_score |
| Временные изменения | color_transition_variance |

> 💡 Всего можно получить 100–200 цветовых/осветительных фичей.

## 4.2. Композиция кадра                🟩🟥🟥

### 🧱 A) Правило третей (Rule of Thirds)

#### 1) Расположение главного объекта

Выделяешь главный объект (или лицо) через object detection + tracking.

**Фичи:**

- `main_subject_x_pos` (0–1 относительно сетки 3×3)
- `main_subject_y_pos`
- `rule_of_thirds_alignment` (0–1 расстояние от пересечения линий)
- `subject_offcenter_distance`
- `subject_offcenter_angle`

#### 2) Secondary subjects

- `secondary_subjects_count`
- `secondary_subjects_alignment_score`
- `subject_balance_index`

⚖️ B) Баланс композиции

> 💡 Мера того, насколько кадр "перевешивает" в сторону — супер важная фича.

**Считается по:**

- heatmap attention (Grad-CAM)
- brightness distribution
- object sizes

**Фичи:**

- `balance_left_right_ratio`
- `balance_top_bottom_ratio`
- `mass_center_x`, `mass_center_y`
- `visual_weight_asymmetry`

🔍 C) Leading Lines (направляющие линии)

> 💡 Эти линии указывают взгляд зрителя и определяют профессионализм видео.

**Находятся через детекцию линий:**

- Hough transform
- LSD (Line Segment Detector)
- Deep-LSD / HAWP (нейросетевые детекторы линий)

**Фичи:**

- `leading_lines_count`
- `leading_lines_strength` (длина + контраст)
- `leading_lines_direction_mean`
- `leading_lines_to_subject_alignment` (насколько они указывают на объект)

### 🔶 D) Depth & Foreground/Background (глубина пространства)

Можно вынести из:

- MiDaS (depth estimation)
- DPT (state-of-the-art)

**Фичи:**

#### 1) Глубина сцены

- `depth_mean`
- `depth_std`
- `depth_dynamic_range`
- `depth_entropy`
- `foreground_size_ratio`
- `background_clutter_index`

#### 2) Depth layers

- `num_depth_layers` (2D → 3D планов)
- `foreground_depth_distance`
- `midground_presence_ratio`
- `background_depth_balance`

#### 3) Cinematic depth indicators

- `bokeh_probability`
- `shallow_depth_of_field_prob`
- `focus_plane_variation`

🎯 E) Симметрия

#### 1) Reflection-based symmetry

- `horizontal_symmetry_score`
- `vertical_symmetry_score`
- `radial_symmetry_score`

#### 2) Object symmetry

- `face_symmetry_score` (если лицо в кадре — важная эстетическая метрика)

**scene_symmetry_type:**

- none
- horizontal-align
- vertical-align
- central symmetrical layout

📐 F) Negative Space (негативное пространство)

> 💡 Это огромный фактор в рекламе, роликах с продуктами, интервью, влогах.

**Находим:**

- сегментация кадра (SAM, DeepLab)
- выделяем пустые зоны

**Фичи:**

- `negative_space_ratio`
- `negative_space_left/right/top/bottom`
- `empty_background_entropy`
- `object_to_background_ratio`

🖼 G) Framing (обрамление кадра)

Как элементы кадра «обрамляют» объект:

- doorway framing
- screen within screen
- frame-inside-frame
- natural framing (деревья, окна, коридоры)

**Фичи:**

- `framing_present`
- `framing_strength`
- `framing_type`

> → Это часто используется в художественной съёмке.

🧩 H) Complexity & Clutter (захламлённость сцены)

Используется в рекламе и эстетическом анализе.

**Методы:**

- edge density
- region segmentation entropy
- object-count-based clutter

**Фичи:**

- `scene_complexity_score`
- `edge_density`
- `object_clutter_index`
- `background_texture_complexity`

🎞 I) Composition Style Classification

Можем классифицировать стиль композиции:

- Minimalist
- Documentary / Real-life
- Vlog
- Cinematic
- Product-centered
- Interview
- TikTok aesthetic
- Gaming UI
- Artistic / abstract

**Фичи:**

- `style_minimalist_prob`
- `style_cinematic_prob`
- `style_interview_prob`
- `style_product_prob`
- `style_gaming_prob`
- `style_tiktok_prob`

### 🧲 J) Attention Map / Saliency Analytics

**С помощью:**

- DeepGaze II,
- Itti-Koch visual saliency,
- SAM-based attention

Можно анализировать:

куда смотрит человеческий глаз

равномерность внимания

концентрацию на объекте

Фичи:

saliency_center_bias

saliency_focus_spread

saliency_entropy

attention_to_face_ratio

attention_to_product_ratio

🔥 Итог: Сколько фичей можно извлечь?

По 4.2. Композиции можно спокойно достать 80–150 фичей, которые будут иметь значение для:

Предсказания качества/вирусности

Построения отчётов для проф. аналитиков

Оценки уровня продакшена

Разбора стиля создания контента

## 4.3. Shot Quality Metrics            🟩🟥🟥

🧼 A) Sharpness & Clarity (резкость)

> 💡 Резкость — главный показатель технического качества.

**Извлекается через:**

- Laplacian variance
- Tenengrad gradient
- SMD2 (Sum of Modified Differences)
- Deep blur detector (CNN)

**Фичи:**

- `sharpness_laplacian`
- `sharpness_tenengrad`
- `blur_score`
- `motion_blur_probability`
- `focus_accuracy_score`
- `edge_clarity_index`
- `spatial_frequency_mean`

📸 B) Noise Levels (шум)

**Определяется через:**

- Wavelet noise estimation
- BM3D noise prior
- CNN noise prediction models

**Фичи:**

- `noise_level_luma`
- `noise_level_chroma`
- `iso_estimated_value`
- `grain_strength`
- `noise_spatial_entropy`

🌑 C) Exposure (экспозиция)

> 💡 Качество экспозиции = насколько правильно снят кадр.

**Фичи:**

- `underexposure_ratio`
- `overexposure_ratio`
- `midtones_balance`
- `exposure_histogram_skewness`
- `highlight_recovery_potential`
- `shadow_recovery_potential`
- `exposure_consistency_over_time`

🔅 D) Contrast Quality

- `contrast_global`
- `contrast_local`
- `contrast_dynamic_range`
- `contrast_clarity_score`
- `microcontrast` (очень сильный показатель качества линзы)

🎨 E) Color Accuracy & Consistency

**Технические признаки:**

- `white_balance_shift`
- `color_cast_type` (green/magenta shift)
- `skin_tone_accuracy_score`
- `color_fidelity_index`
- `color_noise_level`
- `color_uniformity_score`

🎚 F) Compression Artifacts (битрейт и качество кодека)

**Можно определить визуально, без знания оригинального файла:**

- Blockiness
- Ringing
- Banding
- Mosquito noise

**Фичи:**

- `compression_blockiness_score`
- `banding_intensity`
- `ringing_artifacts_level`
- `bitrate_estimation_score`
- `codec_artifact_entropy`

📐 G) Lens Quality Simulation

Изображение можно проанализировать на предмет качества оптики:

- `chromatic_aberration_level` (фиолетовые/зелёные ореолы)
- `vignetting_level`
- `distortion_type` (barrel/pincushion)
- `lens_sharpness_drop_off` (снижение резкости к краям)

🌫 H) Cleanliness / Obstructions Detection

**Анализ помех:**

- водяные капли на объективе
- грязь
- запотевание
- затемнение углов (hood)

**Фичи:**

- `lens_obstruction_probability`
- `lens_dirt_probability`
- `fog_haziness_score`
- `veiling_glare_score`

🧪 I) Temporal Quality Metrics (по времени)

> 💡 Очень важны для динамичных сцен:

- `temporal_sharpness_stability`
- `temporal_exposure_stability`
- `temporal_noise_variation`
- `temporal_flicker_score`
- `rolling_shutter_artifacts_score`

🎞 J) Shot Technical Classifier (quality levels)

**Можно создать классификатор, который определяет уровень кино/проф. качества:**

- cinematic_pro
- cinematic_lowlight
- smartphone_good
- smartphone_poor
- webcam
- screen_recording
- surveillance

**Фичи:**

- `quality_cinematic_prob`
- `quality_smartphone_prob`
- `quality_webcam_prob`
- `quality_screenrecord_prob`
- `quality_surveillance_prob`

### 🔥 ИТОГ: Какие фичи можно собрать из "Shot Quality Metrics"?

Блок 4.3 может дать 70–120 высокополезных фичей, среди них:

| Группа | Примеры |
|--------|---------|
| Резкость | Laplacian, Tenengrad, blur score |
| Шум | luma/chroma noise, grain |
| Экспозиция | under/over exposure, histogram skew |
| Контраст | microcontrast, global/local |
| Цвет | white balance shift, skin tone accuracy |
| Компрессия | blockiness, banding, ringing |
| Оптика | aberrations, vignetting |
| Препятствия | dirt/fog/glare |
| Временные качества | flicker, rolling shutter |
| Классификатор качества | cinematic/smartphone/webcam |

# ✂️ 5. Монтаж, ритм, структура

## 5.1. Cut detection                   🟩🟥🟥

🔧 A) Типы склеек (детектируемые)
✔ Hard Cuts (моментальное переключение)

Определяются через:

разницу гистограмм

разницу feature maps ViT/ResNet

SSIM drop

optical flow jump

Фичи:

hard_cuts_count

hard_cuts_per_minute

hard_cut_strength_mean

✔ Soft Cuts (плавные переходы)

fade-in/out, dissolve

Определяются через:

временную деконсистентность яркости

медленное изменение гистограммы

низкий optical flow, но плавный SSIM drop

Фичи:

fade_in_count

fade_out_count

dissolve_count

avg_fade_duration

✔ Motion-based Cuts (динамические переходы)

Это “whip pan”, “zoom transition”, "speed ramp cuts".

Определяются по всплескам optical flow:

whip_pan_transitions_count

zoom_transition_count

motion_cut_intensity_score

flow_spike_ratio

✔ Stylized / Editing Transitions

Можно классифицировать по CNN:

transition_wipe_count

transition_slide_count

transition_glitch_count

transition_flash_count

transition_luma_wipe_count

🎞 B) Cut Timing Statistics

(очень важны для анализа ритма!)

Фичи:

🚀 Частота склеек

cuts_per_second

cuts_per_minute

median_cut_interval

min_cut_interval

max_cut_interval

⚖ Ритм

cut_interval_std

cut_interval_entropy

cut_rhythm_uniformity_score

Малое стандартное отклонение → ровный монтаж (интервью/новости).
Высокое → хаотичный монтаж (TikTok, экшн, мемы).

🔥 C) Shot Duration Analysis

После определения склеек мы можем измерить:

avg_shot_length

median_shot_length

short_shots_ratio (<1 сек)

long_shots_ratio (>4 сек)

very_long_shots_count (>10 сек)

extremely_short_shots_count (<0.25 сек)

Вирусные ролики часто используют:

много коротких планов

паттерны: 0.8–1.5 сек/кадр

🧠 D) Детекция Jump Cuts (важно для блогов!)

Jump-cuts = резкая смена позиции объекта на одном плане.
Почти все TikTok/влоги используют это.

Фичи:

jump_cuts_count

jump_cut_ratio_per_minute

jump_cut_intensity

Метод:

сравнение позы/лица/главного объекта между adjacent shot

если фон одинаковый → это jump cut

🌀 E) Scene Boundary Detection (High-level сегментация)

Можно объединить шоты в сцены:

scene_count

avg_scene_length

scene_to_shot_ratio

scene_transition_types

Это уже части storytelling analytics.

🎧 F) Audio-assisted Cut Detection (повышение качества)

При резких сменах звука (VAD/energy spike):

audio_cut_alignment_score

audio_spike_cut_ratio

scene_whoosh_transition_prob

Многие монтажи синхронизируются с битами музыки.

🎯 G) Shot & Cut Style Classification

Классификатор типа монтажа:

fast-cut montage

slow-paced editorial

social media style

documentary style

cinematic editing

“meme-style editing”

high-action-edit (GoPro)

Фичи:

edit_style_fast_prob

edit_style_cinematic_prob

edit_style_meme_prob

edit_style_social_prob

edit_style_slow_prob

### 📌 Итог: Какие фичи можно собрать по 5.1?

60–120 монтажных признаков, включая:

| Категория | Примеры |
|-----------|---------|
| Типы склеек | hard/fade/dissolve/whip pan |
| Частота | cuts per minute, intervals |
| Ритм | entropy, std, uniformity |
| Длительность планов | avg/median/short/long |
| Jump cuts | count, intensity |
| Scene boundaries | scene_count |
| Аудио-связь | audio-cut sync score |
| Стиль монтажа | fast/cinematic/meme |

## 5.2. Video pacing                    🟩🟥🟥

🔥 A. Метрические фичи темпа монтажа (чистый pace)
1. Cut rate per second

cuts_per_10s (кол-во склеек за каждые 10 секунд)

cuts_variance (разброс — важный показатель динамических сцен)

longest_shot_duration

shortest_shot_duration

median_shot_duration

shot_duration_entropy (насколько разные длительности шотов)

Модель для шотов:

PySceneDetect

или своя CNN/Transformers модель для cut detection

2. Shot rhythm curve

Строится временной ряд изменения длительности шотов:

pace_curve_mean

pace_curve_slope (ускоряется ли темп монтажа)

pace_curve_peaks (всплески смен кадров)

pace_curve_periodicity (есть ли ритмика — клипы, монтаж как у MrBeast)

3. Scene pacing

Считаем:

scene_changes_per_minute

average_scene_duration

scene_duration_variance

🎭 B. Темп визуального контента внутри шотов

> 💡 Даже без склеек темп может быть высоким, если камера движется или объекты активны.

#### 4. Motion pacing

На базе optical flow:

- `mean_motion_speed_per_shot`
- `motion_speed_variance`
- `share_of_high_motion_frames`
- `optical_flow_direction_changes_per_second` (хаос в движении камеры)

#### 5. Content change rate

Каждый кадр делаем embedding (например, CLIP, SigLIP).

**Дальше считаем:**

- `frame_embedding_diff_mean`
- `frame_embedding_diff_std`
- `high_change_frames_ratio`
- `scene_embedding_jumps`

**Это фактически измеряет «визуальную новизну»:**

- появление новых объектов, людей
- резкое изменение кадра
- смена цветовой палитры

> 💡 YouTube и TikTok обожают high novelty pace.

🎤 C. Темп аудио

> 💡 Не забывай: динамика видео — это еще и audio pacing.

#### 6. Speech pacing

На основе ASR (whisper):

- `words_per_minute`
- `speech_rate_variance`
- `pause_count_per_minute`
- `average_pause_duration`
- `talk_bursts_per_minute` (интенсивные речевые отрывки)

#### 7. Music pacing

**Определяется:**

- `bpm`
- `bpm_fluctuation`
- `spectral_flux` (насколько меняется спектр)

#### 8. Audio-visual pacing sync

> 💡 Это advanced фича:

**Считаем корреляцию:**

- визуальной динамики (optical flow)
- аудио динамики (energy of audio signal)

**Получаем:**

- `av_sync_score` (0 — несоответствие, 1 — хореография)
- `av_energy_alignment`
- `beats_per_cut_ratio`

🤖 D. Поведенческий темп (action pacing)

Если есть люди и объекты, можно мерить темп их активности.

#### 9. Per-person motion pace

По keypoints (Mediapipe, RTMPose):

- `avg_pose_speed`
- `pose_speed_variance`
- `bursts_of_activity_per_minute`
- `freeze_moments_count`

#### 10. Object change pacing

По трекингу объектов:

- `new_objects_per_10s`
- `object_entry_exit_rate`
- `main_object_switching_rate` (например, смена спикеров)

🎨 E. Темп визуального стиля

#### 11. Color pacing

Считается на уровне шотов или 1fps:

- `color_histogram_diff_mean`
- `saturation_change_rate`
- `brightness_change_rate`

> 💡 Видео с частой сменой цветовых тональностей — обычно динамичные (влог travel, music videos).

#### 12. Lighting pacing

С резкими перепадами яркости (магия, трюки, прыжки, переходы):

- `luminance_spikes_per_minute`
- `high_frequency_flash_ratio`

🧩 F. Структурные фичи темпа

#### 13. Segment pacing (story sections)

После scene segmentation можно:

- `intro_pace`
- `main_pace`
- `climax_pace`
- `outro_pace`

> 💡 Например: Если intro очень медленный → падение удержания.

#### 14. Pacing symmetry

**Смотрим:**

- Ascending pace (ускорение к концу видео)
- Descending pace
- Chaotic pace (хаотичная смена)

> Определяется по derivative pace_curve.

📈 Итог: что ты реально можешь получить

Вот компактный список фичей:

#### Cut / shot features

- `shot_durations`: mean, median, std, min, max
- `cut_rate`
- `tempo_curve_slope`
- `shot_duration_entropy`

#### Motion features

- `optical_flow_mean`
- `motion_spikes`
- `camera_movement_intensity`

#### Semantic pacing

- `embedding_change_rate`
- `new_object_rate`
- `new_face_rate`

#### Audio pacing

- `words_per_minute`
- `speech_pause_ratio`
- `audio_energy_variance`
- `bpm`
- `av_sync_score`

#### Stylistic pacing

- `color_change_rate`
- `brightness_change_rate`
- `saturation_change_velocity`

#### Structural pacing

- `intro_speed`
- `main_speed`
- `climax_speed`
- `pacing_symmetry`

## 5.3. Story Structure                 🟩🟥🟥

🧩 A. Сегментация истории (Story Segmentation)

**Ты используешь:**

- Scene detection (визуально)
- Topic segmentation (по тексту — через embeddings)
- Audio energy segmentation (по всплескам звука)
- Face/people presence segmentation
- Visual pattern segmentation (CLIP embeddings)

> На выходе — временная разметка story segments.

**Фичи:**

- `number_of_story_segments`
- `avg_story_segment_duration`
- `narrative_continuity_score` (похожесть соседних сегментов)
- `abrupt_story_transition_count` (jump cuts по смыслу)

🎣 B. Hook Quality Features (первые 5–10 секунд)

> 💡 YouTube и TikTok ориентируются на первые секунды.

**Извлекаемые фичи:**

- `hook_motion_intensity` (оптический поток)
- `hook_cut_rate`
- `hook_topic_relevance` (как быстро появляется тема видео)
- `hook_face_presence` (часто лица в первом кадре → выше удержание)
- `hook_brightness & saturation spike`
- `hook_audio_energy`
- `hook_visual_surprise_score` (насколько резко отличается от последующих кадров)

**hook_surprise_score можно получить через:**

- CLIP embedding jumps
- резкий рост optical flow
- появление лица/крупного объекта
- скачок аудио энергии

🔺 C. Climax / Peak Moment Detection

Ключевые события в ролике сильно коррелируют с:

- Retention > 40%
- Watch time
- Подписками

**Можно автоматом извлечь пик:**

**Используем объединённый сигнал:**

- motion peaks
- audio energy peaks
- face expression peaks (surprise, excitement)
- semantic embedding jumps
- speech intensity
- topic shift

**Фичи:**

- `climax_timestamp`
- `climax_strength` (нормализованный энерго-пик)
- `number_of_peaks` (иногда несколько мини-кульминаций)
- `climax_duration`

🎬 D. Character-level Story Structure (если есть люди)

**Используется детекция:**

- трекинг лиц
- трекинг объектов
- кластеризация CLIP embedding сюжетов

**Фичи:**

- `number_of_speakers`
- `speaker_switch_rate`
- `main_character_screen_time`
- `main_character_presence_curve`
- `new_character_entry_points`

🔤 E. Topic Story Structure (по тексту, субтитрам, речи)

**Используем:**

- sentence embeddings (e5, mpnet, instructor-xl)
- LLM-based topic segmentation
- keyword detection

**Фичи:**

- `number_of_topics`
- `avg_topic_duration`
- `topic_shift_times`
- `topic_diversity`
- `semantic_coherence_score` (продолжительность логически связанных сегментов)

😱 F. Emotion-driven Story Arc

Видео имеют эмоциональную кривую (emotion curve):

- excitement
- happiness
- tension
- curiosity
- surprise

**Можешь получить это через:**

- facial emotions
- voice emotion (prosody, pitch)
- music energy
- semantic emotions из текста

**Строишь кривую и извлекаешь фичи:**

- `emotional_arc_slope`
- `emotional_peaks`
- `emotional_valence_sd`
- `emotional_narrative_shape` (например, 3-актная структура)

🎼 G. Audio Story Structure

**Здесь анализируются:**

- музыка: смена треков, тональность, BPM
- речь: интенсивность, пики, монологи
- шумы: crowd → music → silence → applause

**Фичи:**

- `audio_scene_count`
- `audio_scene_duration`
- `audio_transition_intensity`
- `voice_music_ratio`
- `soundtrack_change_count`

🎨 H. Visual Style Story Structure

**Смотрим, как меняется визуальный стиль:**

- цветовая палитра
- освещение
- композиция
- тип кадров (крупный/средний/общий)

**Фичи:**

- `color_arc_complexity`
- `lighting_curve`
- `framing_style_transition_count`
- `shot_type_distribution`

📚 I. Narrative Shape Classification

Современные модели способны автоматически классифицировать историю по форме:

- 3-act structure
- Hero's journey
- Anecdote
- Tutorial
- Problem → Solution
- Product review
- Challenge format
- Reaction format
- Documentary
- Commentary video
- News

**Можно хранить:**

- `predicted_narrative_type`
- `narrative_confidence`

> 💡 Этим пользуются специалисты и редакторы YouTube-агентств.

💡 Итого — какие фичи у тебя будут

Полный список мл-модельных фичей:

Structural Timing

intro_duration

hook_duration

main_section_duration

climax_duration

outro_duration

cta_presence

Curve-based Features

story_energy_curve

semantic_coherence_curve

emotion_curve

pacing_curve

Story Landmarks

hook_timestamp

climax_timestamp

cta_timestamp

number_of_topics

number_of_speakers

main_character_screen_time

Complex Story Metrics

narrative_type

narrative_complexity

story_flow_score

story_coherence_score

abrupt_semantic_changes

# 🔤 6. Текст в кадре text scoring      🟩🟥🟥

📚 A. Основные извлекаемые данные
1. Raw OCR text dataset

Для каждого кадра/сегмента ты сохраняешь:

сам текст

координаты текста (bbox)

confidence score

стиль текста (детектируется отдельно)

🧩 B. Фичи уровня количества текста
1. Text density

text_area_ratio (процент кадра, занятый текстом)

text_boxes_per_frame

avg_text_length_per_frame

text_duration_on_screen

2. Text frequency

text_appear_rate (новые тексты в секунду)

repeated_text_ratio (один текст повторяется несколько кадров)

🎨 C. Фичи уровня оформления текста
3. Размер и положение

avg_font_size

max_font_size

text_position_distribution:

top

center

bottom
(85% YouTube creators ставят текст в верхнюю треть!)

4. Цвет и стиль

Можно извлечь из bbox:

text_color

outline_color

contrast_with_background

readability_score (формула через WCAG)

5. Тип текста (classification)

subtitles

titles / headers

stickers / memes

captions

emojis

🧠 D. Семантические фичи текста (Text Scoring)

OCR → language model (LLaMA, Qwen, T5, e5 embeddings).

6. Complexity & readability

FKGL (Flesch-Kincaid Grade Level)

Gunning fog index

lexical_density

avg_word_length

keyword_density

unique_words_ratio

7. Attention / virality score (ВАЖНО)

Много агентств YouTube уже используют это:

presence_of_numbers (например «10 вещей…»)

power_words_score («crazy», «insane», «secret», «reveal»)

emotional_tone_score

curiosity_score

urgency_score («now», «fast», «don’t skip»)

Это прямо влияет на CTR / удержание.

8. Topical relevance (соответствие теме видео)

CLIP/Text embeddings сравниваем:

ocr_text_embedding

video_topic_embedding

Фича:
text_topic_alignment_score

🧨 E. Временные фичи текста (как он появляется)
9. Text timing

time_to_first_text (первые секунды = сильный сигнал)

avg_text_display_time

text_on_screen_continuity

text_switch_rate (как быстро меняется текст)

10. Emphasis detection (important!)

Выделяем моменты «пиков»:

text_flash_events

bold/zoomed text moments

callout_text

meme_moment flag

🧲 F. Фичи взаимодействия текста и видео
11. Text → Action correlation

Когда текст появляется синхронно с action peak (jump, transition):

text_action_sync_score

text_motion_alignment

12. Text & speech consistency

Проверяем:
Сказано ли то же самое, что написано?

Фичи:

speech_text_similarity

speech_text_redundancy_score

speech_text_gap (разница между услышанным и увиденным)

13. Multimodal emphasis score

Комбинирует:

брендированные цвета текста

эмоции лица

жесты рук

пик аудио энергии

вспышку кадра

Получается фича:
multimodal_attention_boost_score

Это фактически "энергия момента", которую оценивают продюсеры.

📝 G. Специальные OCR-фичи для YouTube Shorts / TikTok
14. Big text moments

Когда текст занимает >25% кадра:

big_text_rate

big_text_duration

big_text_positions

15. Meme format detection

Если текст → мем (например:
«Когда делаешь…» → верх текста, нижний комментарий)

Фичи:

meme_format_flag

meme_transformation_score

16. Call-to-action (CTA) detection

ищем:

“Subscribe”

“Follow”

“Like”

“Link in bio”

Фичи:

cta_presence

cta_timestamp

cta_strength (эмоционально + визуально)

🧮 Итоговый список фичей (машинно читаемые)

#### 📌 Visual

- `text_area_ratio`
- `avg_font_size`
- `text_color_entropy`
- `readability_score`
- `text_location_histogram`

#### 📌 Semantic

- `ocr_text_embedding`
- `lexical_density`
- `curiosity_score`
- `emotional_tone`
- `topic_alignment_score`

#### 📌 Temporal

- `text_appear_rate`
- `avg_text_duration`
- `text_switch_rate`
- `time_to_first_text`

#### 📌 Multimodal

- `speech_text_similarity`
- `text_action_sync`
- `emphasis_score`
- CTA timestamps

# 🧠 7. Высокоуровневая семантика видео 🟩🟥🟥

🔹 A. Scene-level semantic embeddings

Для каждой сцены (или shot) можно получить:

- CLIP / BLIP / EVA-CLIP / OpenCLIP embeddings
- Embeddings текста + изображения (мультимодальные)

**Используем для:**

- сравнения с другими видео
- кластеризации контента
- поиска тем

**Фичи:**

- `scene_embedding` (1024–4096 dims)
- `scene_similarity_to_video_embedding`
- `scene_topic_label` (через кластеризацию или zero-shot классификацию)

🔹 B. Video-level semantic embeddings

**Объединяем все сцены:**

- `mean_embedding`
- `weighted_mean_embedding` (вес по attention/удержанию/динамике)
- `max_embedding`
- `variance_embedding`

**Позволяет:**

- сравнивать видео между собой
- строить поисковые системы по смыслу
- рекомендовать похожие видео

🔹 C. High-level topic / concept detection

**Используем мультимодальные модели:**

- CLIP zero-shot classification
- BLIP captioning → LLM → topic extraction
- LLM-based video summary → concept extraction

**Фичи:**

- `topic_0 / topic_1 / topic_2` (one-hot / probability)
- `topic_diversity_score`
- `topic_transition_rate` (как часто меняется тема)
- `main_concept_embedding`
- `secondary_concept_embedding`

> 💡 Примеры тем: «спорт», «музыка», «обучение», «приготовление еды», «геймерский контент», «DIY», «мода» и др.

🔹 D. Event detection / key moments

**Используем:**

- мультифреймовые embeddings (CLIP / VideoCLIP / VideoMAE / TimeSformer)
- OCR + Audio + Visual + Text embeddings

**Фичи:**

- `number_of_events_per_video`
- `event_duration_mean`
- `event_importance_score`
- `event_type` (humor, tutorial_step, action, reaction, music drop, etc.)

🔹 E. Emotion / sentiment at video level

**Собираем из:**

- лица → эмоции
- аудио → prosody / speech emotion
- текст → sentiment analysis (OCR, captions, ASR)

**Фичи:**

- `avg_emotion_valence` (средняя положительная/отрицательная окраска)
- `emotion_variance`
- `peak_emotion_intensity`
- `emotion_arcs` (start/mid/end)
- `sentiment_distribution` (positive/neutral/negative)

🔹 F. Story / narrative-level embeddings

**Можно получить через:**

- CLIP/BLIP captions по сценам → LLM → summary embedding
- Topic modeling (BERTopic, Top2Vec)
- Narrative arc detection (Hook → Body → Climax → Resolution)

**Фичи:**

- `narrative_embedding` (vector)
- `story_flow_score` (coherence)
- `narrative_complexity_score`
- `hook_quality_score`
- `climax_strength_score`
- `resolution_presence_flag`

🔹 G. Multimodal semantic features

**Объединяем:**

- Visual embeddings
- Audio embeddings
- Text embeddings (OCR + ASR + subtitles)

**Фичи:**

- `multimodal_attention_score` (что сильнее влияет на удержание)
- `multimodal_event_alignment_score`
- `cross-modal novelty_score` (как часто появляются новые объекты/темы/музыка)

🔹 H. High-level tagging / classification

**Можно предсказывать:**

- жанр видео (educational, entertainment, gaming, vlog, music, sports, etc.)
- тип контента (tutorial, reaction, review, challenge, news, short-form)
- style labels (cinematic, vlog, tutorial, cinematic humor, etc.)

**Фичи:**

- `genre_probabilities` (one-hot / soft)
- `content_type_probabilities`
- `style_probability_distribution`

🔹 I. Summary / Analytics features для проф. аналитиков

- Top concepts / keywords
- Event timeline (timestamps of key events)
- Emotion timeline
- Visual style transitions
- Text & speech highlights
- Semantic novelty (видео уникально по смыслу?)
- Content similarity to top-performing videos

> 💡 Эти фичи позволяют аналитику понять, почему видео работает или нет.

### 📌 Итоговые High-level Semantic фичи

| Группа | Примеры фичей |
|--------|---------------|
| Scene embeddings | scene_embedding, scene_similarity |
| Video embeddings | mean_embedding, weighted_embedding |
| Topics / concepts | topic_0/1/2, topic_diversity, topic_transition_rate |
| Events | number_of_events, event_importance, event_type |
| Emotions / sentiment | avg_valence, peak_emotion, emotion_arcs |
| Narrative / story | narrative_embedding, story_flow_score, hook_score |
| Multimodal | multimodal_attention, event_alignment, cross_modal_novelty |
| High-level labels | genre_probabilities, style_probability, content_type_probabilities |

# 📊 8. Метрики сравнений               🟩🟥🟥

🔹 A. Semantic similarity

**Используем high-level embeddings (блок 7):**

- Cosine similarity между видео embeddings
- CLIP / BLIP / VideoCLIP embeddings

**Фичи:**

- `semantic_similarity_mean` (среднее по топ-N видео)
- `semantic_similarity_max` (максимальная похожесть)
- `semantic_similarity_min`
- `semantic_novelty_score = 1 - max_similarity`

> 💡 Применение: определяет насколько видео уникально или похоже на другие популярные видео.

🔹 B. Topic / Concept Overlap

**Сравниваем темы и концепты видео:**

- topic embeddings (от блока 7)
- Jaccard similarity по извлечённым ключевым словам

**Фичи:**

- `topic_overlap_score` (доля совпадающих тем)
- `topic_diversity_comparison` (разница в тематическом разнообразии)
- `key_concept_match_ratio`

🔹 C. Style & Composition Similarity

**Сравниваем визуальные характеристики:**

- Цвет, свет, композиция (блок 4)
- Типы кадров и монтаж (блок 4.4 + 5)
- Оптический поток и движение (блок 3)

**Фичи:**

- `color_histogram_similarity`
- `lighting_pattern_similarity`
- `shot_type_distribution_similarity`
- `cut_rate_similarity`
- `motion_pattern_similarity`

🔹 D. Text & OCR Similarity

**Сравниваем текст в кадре:**

- OCR embeddings + semantic embeddings
- Распределение текста по кадру и времени

**Фичи:**

- `ocr_text_semantic_similarity`
- `text_layout_similarity` (позиции текста, длина текста, font size)
- `text_timing_similarity` (как быстро появляются тексты в видео)

🔹 E. Audio / Speech Similarity

**Используем:**

- Audio embeddings (yamnet, VGGish, OpenL3)
- Speech / music / prosody
- BPM / tempo / energy

**Фичи:**

- `audio_embedding_similarity`
- `speech_content_similarity` (по ASR)
- `music_tempo_similarity`
- `audio_energy_pattern_similarity`

🔹 F. Emotion & Behavior Similarity

Эмоции лиц, тон голоса, движение объектов

**Фичи:**

- `emotion_curve_similarity` (по кривой эмоций)
- `pose_motion_similarity` (действия людей)
- `behavior_pattern_similarity` (активность объектов)

🔹 G. Temporal / Pacing Similarity

Cut rate, shot duration, scene length. Плавность повествования, ритм.

**Фичи:**

- `pacing_curve_similarity`
- `shot_duration_distribution_similarity`
- `scene_length_similarity`
- `temporal_pattern_novelty` (чем отличается от других видео)

🔹 H. High-level Comparative Scores

**Для аналитиков и моделей:**

- `overall_similarity_score = weighted sum всех выше`
- `uniqueness_score = 1 – overall_similarity`
- `trend_alignment_score = насколько видео похоже на топ видео в своей нише`
- `viral_pattern_score = схожесть с успешными видео (опираемся на historical data)`

🔹 I. Group / Batch Metrics

Если анализируем набор видео:

- Cluster similarity metrics
- Inter-video variance (по темам, эмоциям, монтажу, аудио)
- Top-K nearest neighbors
- Outlier detection (анализ уникальных/нестандартных видео)

### 📌 Итоговые фичи блока 8

| Категория | Примеры фичей |
|-----------|---------------|
| Semantic | semantic_similarity_mean/max/min, semantic_novelty_score |
| Topics | topic_overlap_score, key_concept_match_ratio |
| Visual / Composition | color_histogram_similarity, shot_type_distribution_similarity |
| Text / OCR | ocr_text_semantic_similarity, text_timing_similarity |
| Audio | audio_embedding_similarity, speech_content_similarity |
| Emotion / Behavior | emotion_curve_similarity, pose_motion_similarity |
| Temporal / Pacing | pacing_curve_similarity, shot_duration_distribution_similarity |
| High-level | overall_similarity_score, uniqueness_score, trend_alignment_score |

# ⭐ 9. Степень уникальности видео      🟥🟥🟥

🔹 A. Semantic / Content Novelty

**Используем embeddings (блок 7 + 8):**

- Video-level embeddings (CLIP, BLIP, VideoCLIP)
- Сравниваем с топ-100/топ-1000 видео в данной категории

**Фичи:**

- `semantic_novelty_score = 1 – max(similarity_with_top_videos)`
- `topic_novelty_score = доля новых концептов, которых нет в популярных видео`
- `concept_diversity_score = количество уникальных концептов / общее количество`

🔹 B. Visual / Style Novelty

**Сравниваем визуальный стиль (цвет, свет, композицию, тип кадров):**

- `color_palette_novelty = различие палитры с топ-видео`
- `lighting_style_novelty = уникальные схемы освещения`
- `shot_type_novelty = редкие кадры или нестандартные планы`
- `camera_motion_novelty = уникальные движения камеры (dolly, gimbal, drone)`

🔹 C. Editing & Pacing Novelty

Cut rate, rhythm, shot duration distribution, scene length.

**Используем сравнение с топ видео в нише**

**Фичи:**

- `cut_rate_novelty`
- `shot_duration_distribution_novelty`
- `scene_length_novelty`
- `pacing_pattern_novelty`

🔹 D. Audio Novelty

Музыка, эффекты, голос, темп речи

**Фичи:**

- `music_track_novelty` (новый трек, BPM, стиль)
- `voice_style_novelty`
- `sound_effects_novelty`
- `audio_energy_pattern_novelty`

🔹 E. Text / OCR Novelty

Заголовки, надписи в кадре, мемы

**Фичи:**

- `ocr_text_novelty = новые ключевые слова / фразы`
- `text_layout_novelty = необычное расположение текста`
- `text_style_novelty = уникальные шрифты, цвета, эффекты`

🔹 F. Behavioral & Motion Novelty

Движения людей, объектов, экшн зоны

**Фичи:**

- `pose_motion_novelty = необычные движения / редкие паттерны`
- `object_interaction_novelty = уникальные взаимодействия объектов`
- `action_sequence_novelty = уникальные комбинации действий`

🔹 G. Multimodal Novelty

**Комбинируем визуальный, аудио, текстовый и семантический сигнал**

**Фичи:**

- `multimodal_novelty_score`
- `novel_event_alignment_score` (новые события в видео, которых нет в топ видео)
- `overall_novelty_index` (средневзвешенный по всем модальностям)

🔹 H. Temporal / Trend Novelty

**Оцениваем новизну по времени и трендам:**

**Фичи:**

- `trend_alignment_score = насколько видео соответствует текущему тренду (низкий → уникально)`
- `historical_similarity_score = сравнение с прошлым контентом`
- `early_adopter_score = насколько видео новаторское в своей нише`

### 📌 Итоговые фичи блока 9

| Категория | Примеры фичей |
|-----------|---------------|
| Semantic | semantic_novelty_score, topic_novelty_score, concept_diversity_score |
| Visual / Style | color_palette_novelty, lighting_style_novelty, shot_type_novelty |
| Editing / Pacing | cut_rate_novelty, shot_duration_distribution_novelty, pacing_pattern_novelty |
| Audio | music_track_novelty, voice_style_novelty, sound_effects_novelty |
| Text / OCR | ocr_text_novelty, text_layout_novelty, text_style_novelty |
| Behavioral / Motion | pose_motion_novelty, object_interaction_novelty, action_sequence_novelty |
| Multimodal | multimodal_novelty_score, novel_event_alignment_score, overall_novelty_index |
| Temporal / Trend | trend_alignment_score, historical_similarity_score, early_adopter_score |