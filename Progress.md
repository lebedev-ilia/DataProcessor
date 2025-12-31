Итог на сейчас (handoff для другой нейросети)
Контекст / цель
Мы ведём проект TrendFlow/DataProcessor к baseline по плану из docs/BASELINE_IMPLEMENTATION_PLAN.md. Сейчас были две ключевые задачи:
Финальная проверка/фикс Segmenter (union sampling + корректный metadata.json).
Начать динамический батчинг (на уровне модулей и оркестраторов), но мы успели только начать инфраструктуру.
Что сделано сегодня (фактические изменения в коде)
1) Segmenter — доведён до “строгого” контракта union-sampling
Файл: Segmenter/segmenter.py
Сделаны важные правки/гарантии:
Папка вывода теперь следует video_id, если он передан (а не basename файла).
Раньше Segmenter писал в папку по имени файла (что ломало ожидания оркестратора, если файл назывался иначе).
union_frame_indices_source и union_timestamps_sec теперь отражают только реально извлечённые кадры, а не “запрошенные”.
Это критично: раньше могла быть ситуация, когда union содержит source-индексы, которые фактически не считались (короткое/битое видео) → downstream мог получить frame_indices, которые указывают на несуществующие union-позиции.
Добавлены поля совместимости в metadata:
batch_size (дублирует chunk_size), чтобы FrameManager уверенно читал и старый и новый формат.
Добавлен self-check в Segmenter.run():
для каждого frames_meta[component].frame_indices проверяется: list, int, sorted, unique, in-range относительно total_frames (union-domain);
если что-то не так — raise (контракт “no-fallback”/строгая валидация).
Текущее поведение: Segmenter формирует per-component frame_indices в union-domain, плюс хранит debug source_frame_indices.
2) Начата инфраструктура для динамического батчинга (пока только “probe”)
Файл: VisualProcessor/utils/resource_probe.py (добавлен)
Содержит:
get_cuda_mem_info() — best-effort получение (free_bytes, total_bytes) через torch.cuda.mem_get_info(), возвращает None если CUDA недоступна/torch нет.
pick_device(preferred="auto") — безопасный выбор cpu/cuda.
Важно: файл VisualProcessor/utils/batching.py пытались добавить, но он не был создан и сейчас отмечен как удалённый/отсутствующий. Реализация auto_batch_size() ещё не сделана.
Что было обнаружено (важные наблюдения по коду)
В core провайдерах уже есть ручной --batch-size:
VisualProcessor/core/model_process/core_clip/main.py
VisualProcessor/core/model_process/depth_midas/main.py
VisualProcessor/core/model_process/object_detections/main.py (YOLO batching)
Но это статическая константа, без автоподбора по памяти/размеру кадра.
В object_detections/main.py есть путаница ожиданий по цвету: местами документация/комменты ожидают BGR, но FrameManager.get() по контракту возвращает RGB. Это нужно привести к единому: считать вход RGB и НЕ делать лишних конверсий.
Текущее состояние задач (TODO)
✅ Segmenter финально закрыт (union mapping корректный, folder naming по video_id, self-check).
⏳ Dynamic batching framework — только старт: есть resource_probe, но нет auto_batch_size, нет интеграции в модули.
⏳ Dynamic parallelism на уровне оркестраторов — ещё не начинали.
Что следующей нейросети делать дальше (конкретный план “с места”)
A) Доделать VisualProcessor/utils/batching.py
Создать новый файл (его сейчас нет) и реализовать, минимум:
estimate_image_bytes(h, w, channels=3, dtype_bytes=4, safety_mult=...)
auto_batch_size(device, frame_shape, model_hint, max_batch_cap, reserve_ratio)
если device="cpu" → возвращать небольшой фикс (например 1–4) или конфиг.
если cuda → использовать get_cuda_mem_info() и подобрать batch так, чтобы укладываться в free_bytes * (1 - reserve_ratio).
B) Интегрировать auto-batching в core провайдеры (первый проход)
Начать с:
core_clip/main.py: если --batch-size 0 или --batch-size auto → подбирать автоматически.
depth_midas/main.py: аналогично, но с более консервативным safety factor (MiDaS прожорливее).
object_detections/main.py (YOLO path): автоподбор batch_size + привести вход к RGB контракту (убрать BGR ожидания/конверсии где не нужно).
C) Динамический батчинг на уровне модулей/оркестратора
Протащить “policy” через config/CLI (например batch_size=auto, device=auto) в VisualProcessor/main.py и/или общий orchestrator.
Добавить верхнеуровневые лимиты параллелизма (например max_parallel_videos) на основе GPU RAM/CPU RAM (хотя бы stub + безопасный default=1).
Полезные точки входа в код (куда смотреть)
Segmenter: Segmenter/segmenter.py
Frame IO: VisualProcessor/utils/frame_manager.py (читает batch_size или chunk_size)
Core providers:
VisualProcessor/core/model_process/core_clip/main.py
VisualProcessor/core/model_process/depth_midas/main.py
VisualProcessor/core/model_process/object_detections/main.py
(опционально) core_optical_flow/main.py, core_face_landmarks/main.py
Baseline plan: docs/BASELINE_IMPLEMENTATION_PLAN.md
Критичные контракты, которые нельзя сломать
Segmenter владеет sampling: модули/коры не придумывают frame_indices, только читают из metadata и не фолбэчат.
FrameManager.get() возвращает RGB.
Union-domain: metadata[component].frame_indices — индексы в union-таймлайне (0..N-1), frames_dir хранит только union кадры.