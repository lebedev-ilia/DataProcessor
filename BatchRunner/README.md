# BatchRunner (HF videos → DataProcessor → HF artifacts)

Цель: массово прогонять видео (из HF datasets `Ilialebedev/videos1..11`) через **Segmenter → VisualProcessor → AudioProcessor** и публиковать per‑video артефакты (NPZ/manifest) в отдельный HF dataset (files-based).

## Основные принципы

- **Idempotency**: если видео уже обработано и загружено — пропускаем.
- **Batching**: скачиваем батч (8–16), обрабатываем **по одному** (RTX2060 6GB), выгружаем, чистим локальный кеш.
- **Progress**:
  - `Interpret/excluded_videos.json` — список video_id, которые нужно исключить (ошибка скачивания и т.п.)
  - `BatchRunner/state.jsonl` — события прогонов (ok/error/empty), run_id, пути, тайминги.
  - (опционально) `BatchRunner/hf_out_existing.json` — локальный кеш “что уже есть в hf-out-repo”, чтобы не листать HF на каждом запуске.

## Шаги

1) Построить индекс где лежит каждый `video_id`:

```bash
source Interpret/.venv/bin/activate
python BatchRunner/build_hf_video_index.py \
  --owner Ilialebedev \
  --datasets videos1,videos2,videos3,videos4,videos5,videos6,videos7,videos8,videos9,videos10,videos11 \
  --out BatchRunner/hf_video_index.json
```

2) Запуск runner:

```bash
source Interpret/.venv/bin/activate
python BatchRunner/run_batch.py \
  --main-ready-dir /Users/user/Desktop/DataProcessor/Interpret/main_ready \
  --hf-index BatchRunner/hf_video_index.json \
  --result-store-base /Users/user/Desktop/DataProcessor/result_store_runs \
  --hf-out-repo Ilialebedev/vp_runs_npz \
  --hf-videos11-repo Ilialebedev/videos11 \
  --batch-size 12 \
  --skip-remote-existing
```

## Fallback download (если видео нет в videos1..11)

Если `video_id` отсутствует в `hf_video_index.json`, runner:
- скачает видео через `yt-dlp` (с ограничением `--max-duration-sec`, по умолчанию 1200)
- загрузит его в `--hf-videos11-repo` как `<video_id>.mp4`
- затем продолжит обработку (Segmenter → Visual → Audio)

Опционально можно указать cookies для yt-dlp:

```bash
python BatchRunner/run_batch.py \
  ... \
  --yt-cookies /abs/path/to/cookies.txt
```

## Токен HF

Runner ожидает токен в:
- `--hf-token ...` или
- env `HF_TOKEN` / `HUGGINGFACE_TOKEN`

## Resume / idempotency

- По умолчанию runner делает **resume** по `BatchRunner/state.jsonl` (пропускает `process: ok` и `skip: ok`).
- Чтобы отключить: `--no-resume`
- Чтобы не переобрабатывать уже загруженное в HF: `--skip-remote-existing`
- Чтобы обновить кеш удалённого списка: `--refresh-remote-index`


