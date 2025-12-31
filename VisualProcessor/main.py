import os
import sys

_path = os.path.dirname(__file__)

if _path not in sys.path:
    sys.path.append(_path)

import yaml
import argparse
import logging
import subprocess
import json
import hashlib
import uuid
import time

from utils.logger import get_logger
from utils.results_store import ResultsStore
from utils.manifest import RunManifest, ManifestComponent
from utils.artifact_validator import validate_npz

logger = get_logger("VisualProcessor")


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_latest_artifact(component_dir: str, exts=(".npz", ".json")) -> list:
    if not os.path.isdir(component_dir):
        return []
    files = []
    for name in os.listdir(component_dir):
        p = os.path.join(component_dir, name)
        if os.path.isfile(p) and any(name.lower().endswith(e) for e in exts):
            files.append(p)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def _derive_run_context(cfg: dict) -> dict:
    g = cfg.get("global") or {}
    frames_dir = g.get("frames_dir")
    frames_meta = {}
    if frames_dir:
        meta_path = os.path.join(frames_dir, "metadata.json")
        if os.path.exists(meta_path):
            try:
                frames_meta = _safe_load_json(meta_path)
            except Exception:
                frames_meta = {}

    platform_id = g.get("platform_id") or frames_meta.get("platform_id") or "youtube"
    sampling_policy_version = g.get("sampling_policy_version") or frames_meta.get("sampling_policy_version") or "v1"

    # Try to derive video_id if not set (best effort).
    video_id = g.get("video_id") or frames_meta.get("video_id")
    if not video_id:
        vp = frames_meta.get("video_path")
        if isinstance(vp, str) and vp:
            video_id = os.path.splitext(os.path.basename(vp))[0]
    video_id = video_id or "unknown_video"

    run_id = g.get("run_id") or frames_meta.get("run_id") or uuid.uuid4().hex[:12]

    # Hash the full config for reproducibility.
    cfg_dump = yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True)
    config_hash = g.get("config_hash") or _sha256_text(cfg_dump)[:16]

    return {
        "platform_id": platform_id,
        "video_id": video_id,
        "run_id": run_id,
        "config_hash": config_hash,
        "sampling_policy_version": sampling_policy_version,
        "created_at": _utc_iso_now(),
    }


def _build_subprocess_cmd(root_path, name, target, frames_dir, rs_path, cfg):
    """
    Унифицированная сборка команды для запуска модуля / core‑провайдера.
    process_dir: относительный путь от VisualProcessor (например, 'modules' или 'core/model_process').
    """
    vp_root = os.path.join(root_path, "VisualProcessor")

    if target == "core/model_process":
        venv = os.path.join(vp_root, target, ".model_process_venv")
    else:
        venv = os.path.join(vp_root, ".vp_venv")

    python_exec = os.path.join(venv, "bin", "python")

    entry = os.path.join(vp_root, target, name, "main.py")
    if target == "core/model_process" and not os.path.exists(entry):
        # Compat: canonical component names may differ from folder names.
        # Prefer canonical names in metadata/rs_path, but allow legacy folder layout.
        core_folder_alias = {
            "core_object_detections": "object_detections",
            "core_depth_midas": "depth_midas",
        }
        alt_name = core_folder_alias.get(name)
        if alt_name:
            alt_entry = os.path.join(vp_root, target, alt_name, "main.py")
            if os.path.exists(alt_entry):
                entry = alt_entry

    if not os.path.exists(entry):
        raise FileNotFoundError(f"Entry not found for {target}/{name}: {entry}")

    kwargs = []
    for k, v in cfg.items():
        if v is None or v == "False" or v is False:
            continue
        key = f"--{k.replace('_', '-')}"
        if v is True or v == "True":
            kwargs.append(key)
        else:
            kwargs.extend([key, str(v)])

    if not os.path.exists(python_exec):
        logger.warning(
            f"VisualProcessor | main | venv python not found at {python_exec}; "
            f"falling back to current interpreter: {sys.executable}"
        )
        python_exec = sys.executable

    cmd = [
        python_exec,
        entry,
        *kwargs,
        "--frames-dir",
        frames_dir,
        "--rs-path",
        rs_path,
    ]
    return cmd


def run_module(global_cfg, module_name, module_cfg):
    """
    Запуск модульного анализатора через subprocess.
    module_cfg — словарь с параметрами модуля из YAML.
    """
    root_path = global_cfg["root_path"]
    frames_dir = global_cfg["frames_dir"]
    rs_path = global_cfg["rs_path"]

    os.makedirs(rs_path, exist_ok=True)

    cmd = _build_subprocess_cmd(
        root_path=root_path,
        name=module_name,
        target="modules",
        frames_dir=frames_dir,
        rs_path=rs_path,
        cfg=module_cfg,
    )

    logger.info(f"VisualProcessor | main | run_module | ▶ Running module: {module_name}")

    try:
        subprocess.run(cmd, check=True)
        return True, None
    except Exception as e:
        logger.error(f"VisualProcessor | main | run_module | Error: {e}")
        return False, str(e)


def run_core_provider(global_cfg, provider_name, provider_cfg):
    """
    Запуск core‑провайдера моделей через subprocess.

    MVP‑реализация:
    - если есть отдельный скрипт в `core/model_process/<provider_name>/main.py` — используем его;
    - иначе для совместимости пробуем запустить одноимённый модуль из `modules/<provider_name>`.
    """
    root_path = global_cfg["root_path"]
    frames_dir = global_cfg["frames_dir"]
    rs_path = global_cfg["rs_path"]

    impl_kind = "core"

    cmd = _build_subprocess_cmd(
        root_path=root_path,
        name=provider_name,
        target="core/model_process",
        frames_dir=frames_dir,
        rs_path=rs_path,
        cfg=provider_cfg,
    )

    logger.info(
        f"VisualProcessor | main | run_core_provider | ▶ Running core provider "
        f"{provider_name} (impl={impl_kind})"
    )

    try:
        subprocess.run(cmd, check=True)
        return True, None
    except Exception as e:
        logger.error(f"VisualProcessor | main | run_core_provider | Error: {e}")
        return False, str(e)


def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_current_core_providers(config):
    """
    Возвращает список активных core‑провайдеров.
    Ожидает структуру:

    core_providers:
        optical_flow: true
        core_clip: false
    """
    core_cfg = config.get("core_providers") or {}
    return [name for name, enabled in core_cfg.items() if enabled]


def get_current_modules(config):
    """ Возвращает список активных модулей (modules.<name>: true) """
    return [name for name, enabled in config["modules"].items() if enabled]



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DataProcessor Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--cfg-path", type=str, required=True, help="Path to YAML config")

    args = parser.parse_args()

    logger.info(f"VisualProcessor | main | Начало обработки")

    config = load_config(args.cfg_path)

    g_config = config.get("global") or {}

    # Auto-detect root_path if missing/invalid (portable across machines).
    if not g_config.get("root_path") or not os.path.isdir(g_config.get("root_path")):
        # VisualProcessor/main.py lives at <root>/VisualProcessor/main.py
        g_config["root_path"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Derive run context + switch rs_path into per-run storage.
    run_ctx = _derive_run_context(config)
    base_rs_path = g_config["rs_path"]
    run_rs_path = os.path.join(
        base_rs_path,
        run_ctx["platform_id"],
        run_ctx["video_id"],
        run_ctx["run_id"],
    )
    g_config["rs_path"] = run_rs_path
    os.makedirs(run_rs_path, exist_ok=True)

    manifest_path = os.path.join(run_rs_path, "manifest.json")
    manifest = RunManifest(
        path=manifest_path,
        run_meta={
            **run_ctx,
            "frames_dir": g_config.get("frames_dir"),
            "root_path": g_config.get("root_path"),
        },
    )


    current_core = get_current_core_providers(config)
    if current_core:
        logger.info("VisualProcessor | main | Текущие core_providers:")
        for provider in current_core:
            logger.info(f"            {provider}")

        for provider in current_core:
            logger.info(f"VisualProcessor | main | core_provider {provider} start")
            provider_cfg = config.get(provider, {})
            logger.info(f"VisualProcessor | main | {provider} config:")
            for k, v in provider_cfg.items():
                logger.info(f"            {k}: {v}")

            started_at = _utc_iso_now()
            t0 = time.time()
            ok, err = run_core_provider(g_config, provider, provider_cfg)
            duration_ms = int((time.time() - t0) * 1000)
            finished_at = _utc_iso_now()

            comp_dir = os.path.join(run_rs_path, provider)
            artifacts = [{"path": p, "type": os.path.splitext(p)[1].lstrip(".")} for p in _find_latest_artifact(comp_dir)]

            status = "ok" if ok else "error"
            notes = None
            schema_version = None
            producer_version = None

            # Validate latest npz if present
            npz_files = [a["path"] for a in artifacts if a["path"].lower().endswith(".npz")]
            if ok and npz_files:
                v_ok, issues, meta = validate_npz(npz_files[0])
                if not v_ok:
                    status = "error"
                    notes = "artifact validation failed: " + "; ".join(i.message for i in issues[:5])
                schema_version = meta.get("schema_version") if isinstance(meta, dict) else None
                producer_version = meta.get("producer_version") if isinstance(meta, dict) else None

            manifest.upsert_component(
                ManifestComponent(
                    name=provider,
                    kind="core",
                    status=status,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=duration_ms,
                    artifacts=artifacts,
                    error=err,
                    notes=notes,
                    producer_version=producer_version,
                    schema_version=schema_version,
                )
            )

            if status != "ok":
                logger.error(f"VisualProcessor | main | core_provider {provider} failed")


    current_modules = get_current_modules(config)

    logger.info(f"VisualProcessor | main | Текущие модули:")

    for module in current_modules:
        logger.info(f"            {module}")

    for module in current_modules:
        logger.info(f"VisualProcessor | main | {module} start")

        module_cfg = config.get(module)

        logger.info(f"VisualProcessor | main | {module} config:")

        for k, v in module_cfg.items():
            logger.info(f"            {k}: {v}")

        if module_cfg is None:
            raise ValueError(f"❌ Config entry for module '{module}' not found in YAML")

        started_at = _utc_iso_now()
        t0 = time.time()
        ok, err = run_module(g_config, module, module_cfg)
        duration_ms = int((time.time() - t0) * 1000)
        finished_at = _utc_iso_now()

        comp_dir = os.path.join(run_rs_path, module)
        artifacts = [{"path": p, "type": os.path.splitext(p)[1].lstrip(".")} for p in _find_latest_artifact(comp_dir)]

        status = "ok" if ok else "error"
        notes = None
        schema_version = None
        producer_version = None

        npz_files = [a["path"] for a in artifacts if a["path"].lower().endswith(".npz")]
        if ok and npz_files:
            v_ok, issues, meta = validate_npz(npz_files[0])
            if not v_ok:
                status = "error"
                notes = "artifact validation failed: " + "; ".join(i.message for i in issues[:5])
            schema_version = meta.get("schema_version") if isinstance(meta, dict) else None
            producer_version = meta.get("producer_version") if isinstance(meta, dict) else None

        manifest.upsert_component(
            ManifestComponent(
                name=module,
                kind="module",
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                artifacts=artifacts,
                error=err,
                notes=notes,
                producer_version=producer_version,
                schema_version=schema_version,
            )
        )

        if not ok:
            logger.error(f"VisualProcessor | main | module {module} failed")
