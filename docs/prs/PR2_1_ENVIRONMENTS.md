# PR‑2.1 — Virtual environments & dependency hygiene (baseline)

Цель PR‑2.1: убрать “магические” падения из‑за того, что разные части пайплайна запускаются в разных виртуальных окружениях с разным набором пакетов.

## 1) Каноничные окружения (решение)

- **Root orchestrator / Segmenter (baseline smoke)**: `./.data_venv`
  - запускаем root `main.py` именно этим интерпретатором
  - минимальные зависимости зафиксированы в `requirements/dataprocessor_smoke.txt`
- **VisualProcessor (orchestrator + modules по умолчанию)**: `./VisualProcessor/.vp_venv`
  - VisualProcessor сам запускает модули через этот venv (см. `VisualProcessor/main.py`)
- **Isolated core venv (пример конфликтов)**:
  - `core_face_landmarks`: `VisualProcessor/core/model_process/core_face_landmarks/.core_face_landmarks_venv`
  - requirements рядом: `VisualProcessor/core/model_process/core_face_landmarks/requirements.txt`

## 2) Быстрый “doctor” для окружений

Скрипт проверяет наличие venv и базовые imports, плюс `ffmpeg/ffprobe`:

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML/DataProcessor"
python3 scripts/venv_doctor.py
```

## 3) Создание `.data_venv` (smoke)

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML/DataProcessor"
python3 -m venv .data_venv
./.data_venv/bin/python -m pip install -U pip
./.data_venv/bin/pip install -r requirements/dataprocessor_smoke.txt
```

## 4) Почему так (минимально)

- PR‑2/PR‑2.1 фокусируются на **контрактах** (run identity / manifest / meta).
- Полные pinned requirements для всех ML зависимостей будут оформляться позже (когда стабилизируем состав baseline‑модулей и Triton/оптимизации).


