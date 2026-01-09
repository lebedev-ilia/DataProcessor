# Описание фичей модуля micro_emotion

Модуль извлекает микроэмоции и Action Units (AU) с использованием OpenFace через Docker. Оптимизированная версия использует ключевые AU (10-14), baseline subtraction, PCA для остальных AU, компактные метрики pose/gaze/landmarks, детекцию micro-expressions и per-frame векторы для VisualTransformer.

## Общие принципы оптимизации

### Ключевые AU вместо всех 45
Вместо хранения всех 45 AU в непроцессированном виде, выделен ключевой поднабор (10-14 AU), которые наиболее информативны для UGC/вовлечённости. Остальные AU агрегируются через PCA.

### Baseline subtraction
Вычисляется средняя интенсивность AU для данного лица в «нейтральных» кадрах (нижние 20% по общей активности) и хранится `au_intensity_delta = intensity - baseline`. Это существенно уменьшает межсубъектный сдвиг.

### PCA для остальных AU
AU, не входящие в ключевой набор, проецируются через PCA (3-5 компонент), что даёт компактное представление без потери важной информации.

### Компактные геометрические признаки
Вместо хранения всех 68 landmarks в явном виде, вычисляются компактные геометрические признаки (mouth_opening, smile_width, face_asymmetry) и PCA проекции.

## 1. Ключевые Action Units (AU)

### Выбранные AU (10-14)
- **AU06** (cheek raiser) + **AU12** (lip corner puller) — улыбка/счастье
- **AU04** (brow lowerer), **AU01/AU02** (brow raisers) — удивление/грусть/фокус
- **AU25/26** (lips part / jaw drop) — говорение/удивление
- **AU7/23** (lid tightener / lip tightener) — напряжение/негатив
- **AU45/AU43** (blink/eyes closed) — мигание / сонливость
- **AU15** (lip corner depressor) — печаль/негатив
- **AU20** (lip stretcher), **AU10** (upper lip raiser) — другие эмоции

### Фичи для каждого ключевого AU
- **{au}_intensity_mean**: Средняя интенсивность AU (0.0-5.0)
- **{au}_intensity_std**: Стандартное отклонение интенсивности
- **{au}_intensity_delta_mean**: Средняя интенсивность относительно baseline (корректированная)
- **{au}_presence_rate**: Доля кадров, где presence==1 (0.0-1.0)
- **{au}_peak_count**: Количество пиков интенсивности (детектированных вспышек)

### Baseline для AU
- **au_baseline**: Словарь baseline значений для каждого AU (вычисляется из нейтральных кадров)

## 2. PCA для остальных AU

### au_pca_1, au_pca_2, au_pca_3
Первые 3 PCA компоненты для AU, не входящих в ключевой набор. Представляют основные паттерны активности остальных AU.

### au_pca_var_explained_1..k
Доля объяснённой дисперсии для каждой PCA компоненты (показывает информативность компонент).

## 3. Head Pose (Поза головы) — оптимизировано

### pose_Ry_mean, pose_Rx_mean
Средние значения горизонтального поворота (Ry) и наклона/кивка (Rx). Основные метрики для трансформера.

### pose_Ry_std, pose_Rx_std
Стандартные отклонения поворотов. Показывают нестабильность позы.

### pose_stability_score
Оценка стабильности позы (1 - normalized std of Rx,Ry,Rz), где 1.0 = очень стабильная поза, 0.0 = нестабильная.

### pose_Tz_mean
Среднее значение приближения/удаления от камеры («zoom toward camera»). Полезно для оценки вовлечённости.

### pose_Rx/Ry/Rz min/max
Экстремальные значения поворотов (для видео-уровневых агрегатов).

## 4. Gaze Direction (Направление взгляда) — улучшено

### gaze_x_mean, gaze_y_mean
Средние углы взгляда по горизонтали и вертикали (в градусах).

### gaze_x_std, gaze_y_std
Стандартные отклонения углов взгляда.

### gaze_centered_ratio
Доля кадров с взглядом в область камеры (±10° по горизонтали/вертикали). Показывает зрительный контакт.

### blink_rate_per_min
Частота миганий в минуту (детектируется через AU45 presence с длительностью < 0.25s).

### eye_contact_score
Комбинированная оценка зрительного контакта: `gaze_centered_ratio * 0.7 + normalized_blink_rate * 0.3` (0.0-1.0).

## 5. Facial Landmarks (Точки лица) — оптимизировано

### Компактные геометрические признаки

#### mouth_opening_mean, mouth_opening_std
Среднее и стандартное отклонение открытия рта. Вычисляется как расстояние между верхней и нижней губой, нормализованное по межглазному расстоянию.

#### smile_width_mean, smile_width_std
Среднее и стандартное отклонение ширины улыбки (расстояние между уголками губ).

#### face_asymmetry_score
Оценка асимметрии лица (0.0-1.0). Вычисляется через корреляцию L-R landmark distances. Выше значение = больше асимметрия.

#### landmarks_pca_1..5
Первые 5 PCA компонент для всех 68 landmarks (2D). Компактное представление пространственной структуры лица.

### 3D Landmarks

#### head_depth_variation
Вариация глубины головы (std of Z координаты носа). Показывает движение головы вперёд/назад.

## 6. Micro-expressions Detection (новый функционал)

Micro-expressions — быстрые, короткие вспышки AU интенсивности (0.03–0.5s). Часто дают сигнал искренности/эмоционального всплеска.

### Алгоритм детекции
1. Сглаживание AU интенсивности гауссом (σ = 0.03–0.1s)
2. Поиск локальных пиков с амплитудой > baseline + 1.5*std
3. Фильтрация по длительности: peak ≤ 0.5s (для microexpr обычно ≤0.2s)
4. Минимальное расстояние между пиками: 0.2s

### Типы micro-expressions
- **smile**: AU06 + AU12 (улыбка)
- **surprise**: AU01 + AU02 + AU25 + AU26 (удивление)
- **frown**: AU04 + AU15 (хмурость)
- **disgust**: AU09 + AU10 (отвращение)

### Фичи micro-expressions
- **microexpr_count**: Количество обнаруженных micro-expressions
- **microexpr_rate_per_min**: Частота micro-expressions в минуту
- **microexpr_max_intensity**: Максимальная интенсивность среди всех micro-expressions
- **microexpr_types_distribution**: Распределение по типам (словарь: тип → количество)
- **microexpr_timestamps**: Список временных меток micro-expressions (в секундах)
- **microexpr_types**: Список типов для каждого micro-expression

## 7. Per-Frame Vectors для VisualTransformer

Компактный per-frame вектор (~18-22 числа) для VisualTransformer. Компоненты:

1. **time_norm** (1) — нормализованное время кадра (0..1)
2. **face_presence_flag** (1) — флаг наличия лица
3. **au12_intensity_delta** (1) — AU12 интенсивность относительно baseline
4. **au6_intensity_delta** (1) — AU06 интенсивность относительно baseline
5. **au4_intensity_delta** (1) — AU04 интенсивность относительно baseline
6. **au25_presence_rate_short** (1) — AU25 presence в коротком окне (±0.5s)
7. **blink_flag** (1) — флаг мигания (AU45 presence в текущем кадре)
8. **pose_Ry_norm** (1) — нормализованный горизонтальный поворот
9. **pose_Rx_norm** (1) — нормализованный наклон/кивок
10. **gaze_centered_flag** (1) — флаг взгляда в камеру
11. **gaze_x** (1) — нормализованный горизонтальный угол взгляда
12. **gaze_y** (1) — нормализованный вертикальный угол взгляда
13. **mouth_opening_norm** (1) — нормализованное открытие рта
14. **face_asymmetry_score** (1) — оценка асимметрии лица
15. **microexpr_recent_count_window** (1) — число микровспышек в последние 1-2s
16. **au_pca_1..3** (3) — первые 3 PCA компоненты для остальных AU
17. **au_quality_flag** (1) — флаг качества AU (confidence > threshold)

Итого: ~18-22 числа. Все значения z-normalize по train set. Missing values → zeros + соответствующие флаги качества.

## 8. Видео-уровневые агрегаты

### Key AU Aggregates
Для ключевых AU (AU06, AU12, AU04, AU25, AU26, AU07, AU15):
- **{au}_mean**: Средняя интенсивность
- **{au}_std**: Стандартное отклонение
- **{au}_min**: Минимальная интенсивность
- **{au}_max**: Максимальная интенсивность
- **{au}_median**: Медианная интенсивность
- **{au}_peak_count**: Количество пиков

### Micro-expressions Aggregates
- **microexpr_count**: Общее количество micro-expressions
- **microexpr_rate_per_min**: Частота в минуту
- **microexpr_max_intensity**: Максимальная интенсивность
- **microexpr_types_distribution**: Распределение по типам
- **microexpr_burstiness**: Вариативность частоты micro-expressions (variance over per-minute counts)

### Другие агрегаты
- **smile_ratio**: Доля кадров с улыбкой (AU12+AU06 > threshold)
- **eye_contact_ratio**: Доля кадров с взглядом в камеру (gaze_centered_flag==1)
- **blink_rate_per_min**: Частота миганий в минуту
- **pose_stability_score**: Оценка стабильности позы (1 - normalized std)
- **face_presence_ratio**: Доля кадров с обнаруженным лицом
- **avg_mouth_opening**: Среднее открытие рта
- **speaking_ratio**: Доля кадров с говорением (если синхронизировано с аудио)
- **face_turn_away_ratio**: Доля кадров с отворотом головы (|Ry| > threshold)

## 9. Reliability Flags и Quality Scores

### au_quality_overall
Средняя уверенность AU по всем кадрам (0.0-1.0). Вычисляется как среднее значение AU confidence.

### au_quality_reliable
Флаг надёжности AU данных (bool). True если au_quality_overall > threshold (0.5).

### landmark_visibility_mean
Средняя доля видимых landmarks (0.0-1.0). Показывает качество детекции landmarks.

### landmark_visibility_reliable
Флаг надёжности landmarks (bool). True если landmark_visibility_mean > 0.8.

### occlusion_flag
Флаг окклюзии лица (bool). True если landmark_visibility_mean < 0.7.

### lighting_flag
Флаг качества освещения (bool). True если освещение низкое → AU менее надёжны.

## 10. Summary Statistics

### success
Бинарный флаг успешности анализа (True/False). True если хотя бы один кадр содержит лицо.

### face_count_frames
Количество кадров с обнаруженными лицами.

### success_rate
Доля кадров с успешным обнаружением лица (0.0-1.0).

### frames_processed
Количество обработанных кадров.

### au_count_detected
Количество обнаруженных Action Units.

### landmarks_2d_count, landmarks_3d_count
Количество 2D/3D landmarks (обычно 68).

### openface_version, docker_image_tag
Версия OpenFace и Docker образа для воспроизводимости.

## Методы вычисления

1. **Baseline Subtraction**: Вычисляется средняя интенсивность AU для нижних 20% кадров по общей активности (нейтральные кадры).

2. **PCA для AU**: Применяется PCA к интенсивностям всех AU, кроме ключевых. Сохраняются первые 3-5 компонент, объясняющих 90% дисперсии.

3. **Micro-expressions Detection**: 
   - Сглаживание AU интенсивности (σ = 0.03-0.1s)
   - Поиск пиков с порогом > baseline + 1.5*std
   - Фильтрация по длительности (≤ 0.5s) и расстоянию между пиками (≥ 0.2s)
   - Комбинации AU для определения типа выражения

4. **Gaze Centered Detection**: Взгляд считается направленным в камеру, если |gaze_x| < 10° и |gaze_y| < 10°.

5. **Blink Detection**: Мигание детектируется как AU45 presence с длительностью < 0.25s.

6. **Landmark PCA**: Применяется PCA к координатам всех 68 landmarks (2D), сохраняются первые 5 компонент.

7. **Geometric Features**: Вычисляются компактные геометрические признаки (mouth_opening, smile_width, face_asymmetry) вместо хранения всех координат.

## Зависимости

- **Docker**: Для запуска OpenFace контейнера
- **openface/openface:latest**: Docker образ OpenFace
- **pandas**: Для работы с CSV результатами OpenFace
- **numpy**: Для численных вычислений
- **scipy**: Для фильтрации сигналов и поиска пиков
- **scikit-learn**: Для PCA
- **opencv-python**: Для работы с изображениями

## Использование

Модуль требует:
- **Docker**: Установленный и запущенный Docker
- **OpenFace Image**: Загруженный образ `docker pull openface/openface:latest`
- **Frames**: Кадры через FrameManager

Модуль может использовать данные из других модулей через флаги:
- `--use-face-detection`: (legacy flag) фильтровать кадры по `core_face_landmarks.face_present` (face_detection удалён)

## Производительность

- **Скорость**: ~10-30 FPS (зависит от разрешения и сложности сцены)
- **Точность**: Высокая точность благодаря OpenFace
- **Ресурсы**: Требует значительных вычислительных ресурсов и Docker
- **Оптимизация**: Использование ключевых AU и PCA значительно уменьшает размер выходных данных

## Применение

Результаты micro_emotion могут быть использованы для:
- Анализа микроэмоций (резкие эмоции длительностью 0.03-0.5 секунды)
- Оценки искренности и естественности через анализ AU с baseline subtraction
- Анализа физиологических сигналов (стресс, напряжение) через комбинацию AU
- Детекции асимметрии лица через сравнение левой и правой сторон
- Анализа вовлеченности через направление взгляда, зрительный контакт и мимику
- Per-frame векторы для VisualTransformer (компактное представление для downstream моделей)
