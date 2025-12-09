videoprocessor/
│
├── core/                          # Основная инфраструктура (не меняется)
│   ├── __init__.py
│   ├── orchestrator.py            # Главный управляющий pipeline                   +
│   ├── module_executor.py         # DAG модулей, планирование, execution           +
│   ├── base_module.py             # Единый интерфейс модулей                       +
│   ├── frame_reader.py            # Универсальное чтение видео                     +
│   ├── frame_cache.py             # Shared memory / mmap кэш видеокадров           +
│   ├── face_timeline.py           # Быстрая детекция лиц по таймлайну              +
│   ├── frame_sampling.py          # Стратегии выборки кадров                       +
│   ├── scheduler.py               # Планировщик, ресурсы, параллелизм              +
│   ├── resource_manager.py        # GPU/CPU/Memory менеджер                        +
│   ├── result_store.py            # Единое хранилище результатов (in-memory)       +
│   ├── output_store.py            # Запись parquet/json/ndjson                     +
│   ├── model_registry.py          # Lazy-loading и повторное использование моделей +
│   └── config.py                  # Глобальная конфигурация системы                +
│
├──modules/
│   ├── objects_scene/                      # (1) Объекты, сцена, окружение
│   │   ├── object_detection.py             # OWL-ViT / YOLO / RT-DETR
│   │   ├── scene_classification.py         # Places365
│   │   ├── segmentation.py                 # Semantic segmentation (если нужна)
│   │   └── features_object_stats.py        # Aggregation → object stats, entropy
│   │
│   ├── faces/                              # (2) Лица, эмоции, поведение
│   │   ├── face_detection.py               # MediaPipe / YOLO-face
│   │   ├── face_landmarks.py               # 468 landmarks
│   │   ├── face_pose.py                    # Head pose estimation (PnP / Hopenet)
│   │   ├── face_quality.py                 # NIMA, blur, noise
│   │   ├── face_color_lighting.py          # Skin mask → color/light features
│   │   ├── face_attributes.py              # gender, makeup, glasses
│   │   ├── face_iris.py                    # Eye gaze
│   │   ├── face_motion.py                  # Optical flow on facial ROI
│   │   ├── face_3dmm.py                    # DECA/EMOCA 3d modelling
│   │   └── face_rare_features.py           # attractiveness, fatigue, engagement
│   │
│   ├── emotions/                           # (2.2) Эмоции
│   │   ├── basic_emotions.py               # Ekman (DeepFace)
│   │   ├── affectnet_arousal_valence.py    # Arousal-Valence
│   │   ├── micro_expressions.py
│   │   ├── emotion_dynamics.py             # temporal smoothing, changes
│   │   ├── emotion_asymmetry.py            # left/right face
│   │   └── emotion_physiology.py           # action units → physiology
│   │
│   ├── human_behavior/                     # (2.3) Body language, gestures
│   │   ├── hands.py                        # Mediapipe Hands
│   │   ├── pose.py                         # Body pose + keypoints
│   │   ├── tracking.py                     # ByteTrack / OC-SORT
│   │   ├── behavior_gestures.py            # gesture classification
│   │   ├── behavior_movement_patterns.py   # speed/accel of persons
│   │   ├── behavior_interaction.py         # person-object interactions
│   │   ├── behavior_engagement.py
│   │   └── behavior_stress.py
│   │
│   ├── motion_activity/                    # (3) Motion & Action Recognition
│   │   ├── optical_flow.py                 # Farneback / NVIDIA OF
│   │   ├── motion_stats.py                 # jerkiness, energy, smoothness
│   │   ├── camera_motion.py                # shake, zoom, pan/tilt
│   │   ├── action_recognition.py           # VideoMAE, X3D, I3D
│   │   ├── action_temporal.py              # clusters, segments, tempo
│   │   └── motion_foreground_background.py # fg/bg motion
│   │
│   ├── style_cinema/                       # (4) Стиль, цвет, композиция
│   │   ├── color_basic.py
│   │   ├── color_advanced.py               # LAB, HSV, harmonies
│   │   ├── lighting.py
│   │   ├── composition_basic.py            # rule of thirds
│   │   ├── composition_advanced.py         # symmetry, depth, saliency
│   │   ├── shot_quality.py
│   │   ├── aesthetic_scores.py             # NIMA aesthetics, cinematic score
│   │   └── temporal_color.py               # color flow
│   │
│   ├── editing_pacing/                     # (5) Монтаж, Cuts, Pacing
│   │   ├── cut_detection.py
│   │   ├── shot_segmentation.py
│   │   ├── pace_basic.py                   # cut rate, rhythm
│   │   ├── pace_visual.py                  # optical flow -> pace
│   │   ├── pace_audio_visual.py
│   │   ├── pacing_segments.py              # story pacing
│   │   └── cut_style_classification.py
│   │
│   ├── ocr_text/                           # (6) Текст в кадре
│   │   ├── ocr.py
│   │   ├── text_dynamic.py                 # motion of text
│   │   ├── text_semantics.py               # text-topic matching
│   │   ├── text_action_correlation.py
│   │   └── text_meme_format.py
│   │
│   ├── semantics/                          # (7) Семантика видео high-level
│   │   ├── scene_embeddings.py             # CLIP/SigLIP on frames
│   │   ├── video_embeddings.py             # VideoMAE or VideoCLIP
│   │   ├── semantic_topics.py              # topic models
│   │   ├── event_detection.py
│   │   ├── narrative_structure.py
│   │   ├── sentiment_video_level.py
│   │   └── multimodal_embeddings.py
│   │
│   ├── comparisons/                        # (8) Метрики сравнений
│   │   ├── similarity_visual.py
│   │   ├── similarity_audio.py
│   │   ├── similarity_text.py
│   │   ├── similarity_emotions.py
│   │   ├── similarity_pacing.py
│   │   ├── similarity_multimodal.py
│   │   └── batch_video_metrics.py
│   │
│   └── novelty/                            # (9) Новизна видео
│       ├── novelty_visual.py
│       ├── novelty_style.py
│       ├── novelty_pacing.py
│       ├── novelty_audio.py
│       ├── novelty_text.py
│       ├── novelty_multimodal.py
│       └── novelty_trend.py
│
├── preprocess/                    # Предобработка видео/лица/аудио
│   ├── __init__.py
│   ├── face_preprocessing.py
│   ├── landmarks_utils.py
│   ├── bbox_utils.py
│   ├── smoothing.py
│   ├── audio_preprocessor.py
│   └── text_cleaner.py
│
├── runtime/                       # Изоляция зависимостей
│   ├── __init__.py
│   ├── docker/                    # Docker images per module
│   │   ├── deca/
│   │   ├── mediapipe/
│   │   ├── openface/
│   │   ├── tensorflow/
│   │   └── yolov8/
│   ├── environments/              # per-module virtualenvs
│   │   ├── deca_venv/
│   │   ├── openface_venv/
│   │   └── tf_venv/
│   └── runners/                   # запуск модулей как отдельные процессы
│       ├── process_runner.py
│       ├── grpc_runner.py
│       └── local_runner.py
│
├── cache/                         # Everything cache-friendly
│   ├── frames/                    # cached frame images/mmap
│   ├── timelines/                 # face timeline json
│   ├── sampling/                  # sampled windows
│   ├── module_outputs/            # feature cache
│   └── models/                    # downloaded pretrained models
│
├── configs/                       # Настройки
│   ├── modules.yaml               # конфиг модулей
│   ├── sampler.yaml               # sampling policy
│   ├── orchestrator.yaml
│   ├── logging.yaml
│   └── resources.yaml             # GPU/CPU limits
│
├── examples/                      # Примеры использования
│   ├── process_video.py
│   ├── extract_face_features.py
│   ├── sample_video_demo.py
│   └── build_pipeline.py
│
├── tests/                         # Тесты
│   ├── unit/
│   ├── integration/
│   └── performance/
│
├── utils/
│   ├── logging.py
│   ├── timers.py
│   ├── visualizer.py
│   ├── file_utils.py
│   └── gpu_utils.py
│
└── cli/
    ├── videoprocess.py            # CLI for pipeline
    └── inspect_video.py

