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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from utils.logger import get_logger
from utils.results_store import ResultsStore
from utils.manifest import RunManifest, ManifestComponent
from utils.artifact_validator import validate_npz
from utils.resource_probe import get_cuda_mem_info

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
    dataprocessor_version = g.get("dataprocessor_version") or frames_meta.get("dataprocessor_version") or "unknown"
    analysis_fps = g.get("analysis_fps") or frames_meta.get("analysis_fps")
    analysis_width = g.get("analysis_width") or frames_meta.get("analysis_width")
    analysis_height = g.get("analysis_height") or frames_meta.get("analysis_height")
    resolved_model_mapping = cfg.get("resolved_model_mapping") if isinstance(cfg.get("resolved_model_mapping"), dict) else None

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
        "dataprocessor_version": dataprocessor_version,
        "analysis_fps": analysis_fps,
        "analysis_width": analysis_width,
        "analysis_height": analysis_height,
        "resolved_model_mapping": resolved_model_mapping,
        "created_at": _utc_iso_now(),
    }


def _build_subprocess_cmd(root_path, name, target, frames_dir, rs_path, cfg):
    """
    Унифицированная сборка команды для запуска модуля / core‑провайдера.
    process_dir: относительный путь от VisualProcessor (например, 'modules' или 'core/model_process').
    """
    vp_root = os.path.join(root_path, "VisualProcessor")

    if target == "core/model_process":
        # Allow per-core venv overrides (some cores have conflicting deps).
        # 1) Explicit override from config (recommended for special cases)
        cfg_venv = None
        try:
            cfg_venv = (cfg or {}).get("venv_path")
        except Exception:
            cfg_venv = None

        if isinstance(cfg_venv, str) and cfg_venv.strip():
            venv = cfg_venv.strip()
        else:
            # 2) Built-in override for known isolated environments
            if name == "core_face_landmarks":
                venv = os.path.join(vp_root, target, "core_face_landmarks", ".core_face_landmarks_venv")
            else:
                # Default: core providers run in the same VisualProcessor venv.
                # (.model_process_venv is deprecated in this repo.)
                venv = os.path.join(vp_root, ".vp_venv")
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
        # Orchestrator-only keys (must NOT be forwarded to component CLI).
        if k in ("venv_path", "sampling"):
            continue
        # Nested objects are config-only (e.g., sampling dicts); do not forward to CLI.
        if isinstance(v, dict):
            continue
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


def _component_uses_gpu(name: str, cfg: dict) -> bool:
    """
    Conservative heuristic:
    - if config has device in {"cuda","gpu","auto"} -> treat as GPU task (serialize by default)
    - else if component name suggests GPU-heavy core -> GPU task
    """
    try:
        dev = str((cfg or {}).get("device", "")).strip().lower()
    except Exception:
        dev = ""
    # Explicit runtime hint: Triton is assumed GPU-backed in this project.
    try:
        rt = str((cfg or {}).get("runtime", "")).strip().lower()
    except Exception:
        rt = ""
    if rt in ("triton", "triton-gpu", "triton_gpu"):
        return True
    if dev in ("cuda", "gpu", "auto"):
        return True
    # Fallback: GPU-heavy cores are usually GPU-bound unless explicitly cpu
    # Note: core_face_landmarks (MediaPipe) is typically CPU (TFLite/XNNPACK) in our baseline setup.
    if name in ("core_clip", "core_depth_midas", "core_object_detections", "core_optical_flow"):
        return True
    return False


def _device_used_for_component(name: str, cfg: dict) -> str:
    """
    Best-effort device string for manifest.
    Canonical values (MVP): "cpu" | "cuda" | "auto"
    """
    try:
        dev = str((cfg or {}).get("device", "")).strip().lower()
    except Exception:
        dev = ""
    # Explicit runtime hint: Triton is assumed GPU-backed in this project.
    try:
        rt = str((cfg or {}).get("runtime", "")).strip().lower()
    except Exception:
        rt = ""
    if rt in ("triton", "triton-gpu", "triton_gpu"):
        return "cuda"
    if dev in ("cpu", "cuda", "auto"):
        return dev
    if dev in ("gpu",):
        return "cuda"
    return "cuda" if _component_uses_gpu(name, cfg) else "cpu"

def _resolve_gpu_slots(global_cfg: dict) -> int:
    """
    Resolve max concurrent GPU tasks.
    Supported:
    - int (>=1)
    - "auto": 1 for small GPUs, 2 for >= ~20GB (best-effort)
    """
    raw = (global_cfg or {}).get("gpu_max_concurrent", "auto")
    if isinstance(raw, int):
        return max(1, int(raw))
    try:
        s = str(raw).strip().lower()
    except Exception:
        s = "auto"
    if s not in ("auto", ""):
        try:
            return max(1, int(s))
        except Exception:
            return 1
    mem = get_cuda_mem_info()
    if mem is None or mem.total_bytes <= 0:
        return 1
    # Very conservative: allow 2 concurrent GPU tasks only on big VRAM.
    total_gb = float(mem.total_bytes) / (1024.0 ** 3)
    return 2 if total_gb >= 19.0 else 1


def _run_component_subprocess(
    *,
    kind: str,  # "module"|"core"
    global_cfg: dict,
    name: str,
    cfg: dict,
    run_rs_path: str,
    gpu_sem: threading.Semaphore,
) -> tuple:
    """
    Run component in a subprocess with resource gating.
    Returns tuple: (ok, err, artifacts, status, notes, schema_version, producer_version, duration_ms)
    """
    root_path = global_cfg["root_path"]
    frames_dir = global_cfg["frames_dir"]
    rs_path = global_cfg["rs_path"]

    os.makedirs(rs_path, exist_ok=True)

    target = "modules" if kind == "module" else "core/model_process"
    cmd = _build_subprocess_cmd(root_path=root_path, name=name, target=target, frames_dir=frames_dir, rs_path=rs_path, cfg=cfg)

    needs_gpu = _component_uses_gpu(name, cfg)
    acquired = False
    started_at = _utc_iso_now()
    t0 = time.time()
    try:
        if needs_gpu:
            gpu_sem.acquire()
            acquired = True
            logger.info(f"VisualProcessor | main | {kind} {name} | GPU slot acquired")

        # Ensure repo-root packages (e.g., dp_models, dp_triton) are importable inside component venvs.
        env = os.environ.copy()
        repo_root = str(root_path)
        prev_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = repo_root if not prev_pp else (repo_root + os.pathsep + prev_pp)

        subprocess.run(cmd, check=True, env=env)
        ok, err = True, None
    except Exception as e:
        logger.error(f"VisualProcessor | main | {kind} {name} | Error: {e}")
        ok, err = False, str(e)
    finally:
        duration_ms = int((time.time() - t0) * 1000)
        finished_at = _utc_iso_now()
        if acquired:
            try:
                gpu_sem.release()
            except Exception:
                pass
            logger.info(f"VisualProcessor | main | {kind} {name} | GPU slot released")

    # Collect artifacts + validate
    comp_dir = os.path.join(run_rs_path, name)
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

    return (
        ok,
        err,
        artifacts,
        status,
        notes,
        schema_version,
        producer_version,
        started_at,
        finished_at,
        duration_ms,
    )

def run_module(global_cfg, module_name, module_cfg, run_rs_path: str, gpu_sem: threading.Semaphore):
    return _run_component_subprocess(
        kind="module",
        global_cfg=global_cfg,
        name=module_name,
        cfg=module_cfg,
        run_rs_path=run_rs_path,
        gpu_sem=gpu_sem,
    )


def run_core_provider(global_cfg, provider_name, provider_cfg, run_rs_path: str, gpu_sem: threading.Semaphore):
    return _run_component_subprocess(
        kind="core",
        global_cfg=global_cfg,
        name=provider_name,
        cfg=provider_cfg,
        run_rs_path=run_rs_path,
        gpu_sem=gpu_sem,
    )


def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _execution_order(cfg: dict) -> list[str]:
    """
    PR-6: optional DAG execution order (top-level list of component names).
    If provided, VisualProcessor executes enabled components sequentially in this order.
    """
    v = cfg.get("execution_order")
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for x in v:
        if isinstance(x, str) and x:
            out.append(x)
    return out

def _requirements_map(cfg: dict) -> dict:
    """
    PR-4: requirements map enables required/optional enforcement.
    Expected: top-level `requirements: {component_name: bool}`
    If missing/empty -> enforcement is disabled (backward compatible).
    """
    req = cfg.get("requirements")
    return req if isinstance(req, dict) else {}

def _resolved_model_mapping(cfg: dict) -> dict:
    rmm = cfg.get("resolved_model_mapping")
    return rmm if isinstance(rmm, dict) else {}

def _apply_resolved_model_mapping(cfg: dict) -> None:
    """
    PR-8: merge per-component resolved model mapping into component config.
    Only scalar keys matter (nested dict/list values are ignored because they won't be forwarded to component CLI).
    """
    rmm = _resolved_model_mapping(cfg)
    if not rmm:
        return
    for comp, m in rmm.items():
        if not isinstance(comp, str) or not comp:
            continue
        if not isinstance(m, dict) or not m:
            continue
        base = cfg.get(comp)
        if not isinstance(base, dict):
            base = {}
            cfg[comp] = base
        for k, v in m.items():
            if isinstance(v, (dict, list)):
                continue
            base[k] = v

def _is_required(req: dict, component_name: str) -> bool:
    # Default: required=true if map is enabled but key is missing.
    try:
        v = req.get(component_name, True)
    except Exception:
        v = True
    return bool(v)


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
    """Возвращает список активных модулей (modules.<name>: true) в корректном порядке зависимостей."""
    enabled = [name for name, on in (config.get("modules") or {}).items() if on]
    return order_modules_by_deps(enabled)


# Module dependency graph (module -> required modules)
# This enforces strict "no-fallback": if a module consumes another module's outputs, it must run after it.
MODULE_DEPS = {
    "shot_quality": ["cut_detection"],
}


def order_modules_by_deps(enabled_modules):
    enabled_set = set(enabled_modules)

    # validate required deps are enabled
    missing = []
    for m in enabled_modules:
        for dep in MODULE_DEPS.get(m, []):
            if dep not in enabled_set:
                missing.append((m, dep))
    if missing:
        msg = ", ".join([f"{m} requires {dep}" for m, dep in missing])
        raise ValueError(f"❌ Module dependency missing (enable required module): {msg}")

    # topo sort (stable-ish: preserves original ordering where possible)
    order = []
    visiting = set()
    visited = set()

    def dfs(m):
        if m in visited:
            return
        if m in visiting:
            raise ValueError(f"❌ Cycle in module dependencies at: {m}")
        visiting.add(m)
        for dep in MODULE_DEPS.get(m, []):
            if dep in enabled_set:
                dfs(dep)
        visiting.remove(m)
        visited.add(m)
        order.append(m)

    for m in enabled_modules:
        dfs(m)

    # dedupe while preserving topo order
    out = []
    seen = set()
    for m in order:
        if m not in seen:
            out.append(m)
            seen.add(m)
    return out



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DataProcessor Controller",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--cfg-path", type=str, required=True, help="Path to YAML config")

    args = parser.parse_args()

    logger.info(f"VisualProcessor | main | Начало обработки")

    config = load_config(args.cfg_path)
    # PR-8: enrich per-component configs from resolved mapping (profile/DB resolved).
    _apply_resolved_model_mapping(config)
    req_map = _requirements_map(config)
    enforce_requirements = bool(req_map)
    exec_order = _execution_order(config)

    g_config = config.get("global") or {}

    # Auto-detect root_path if missing/invalid (portable across machines).
    if not g_config.get("root_path") or not os.path.isdir(g_config.get("root_path")):
        # VisualProcessor/main.py lives at <root>/VisualProcessor/main.py
        g_config["root_path"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Derive run context + switch rs_path into per-run storage.
    run_ctx = _derive_run_context(config)
    base_rs_path = g_config["rs_path"]
    # Orchestrator may pass the already-resolved per-run directory.
    if bool(g_config.get("rs_path_is_run_dir")):
        run_rs_path = os.path.abspath(str(base_rs_path))
    else:
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
    # Always create/refresh manifest on start, even if no components are enabled.
    manifest.flush()

    # Resource limits for intra-video parallelism
    max_parallel_modules = int(g_config.get("max_parallel_modules", 1) or 1)
    gpu_slots = _resolve_gpu_slots(g_config)
    gpu_sem = threading.Semaphore(value=max(1, int(gpu_slots)))
    logger.info(
        f"VisualProcessor | main | parallelism: max_parallel_modules={max_parallel_modules} gpu_max_concurrent={gpu_slots}"
    )

    current_core = get_current_core_providers(config)
    current_modules = get_current_modules(config)

    if current_core:
        logger.info("VisualProcessor | main | Текущие core_providers:")
        for provider in current_core:
            logger.info(f"            {provider}")

    logger.info("VisualProcessor | main | Текущие модули:")
    for module in current_modules:
        logger.info(f"            {module}")

    enabled_set = set(current_core) | set(current_modules)

    def _run_one_component(name: str) -> None:
        if name in current_core:
            provider_cfg = config.get(name, {})
            logger.info(f"VisualProcessor | main | core_provider {name} start")
            (
                ok,
                err,
                artifacts,
                status,
                notes,
                schema_version,
                producer_version,
                started_at,
                finished_at,
                duration_ms,
            ) = run_core_provider(g_config, name, provider_cfg, run_rs_path=run_rs_path, gpu_sem=gpu_sem)

            manifest.upsert_component(
                ManifestComponent(
                    name=name,
                    kind="core",
                    status=status,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=duration_ms,
                    artifacts=artifacts,
                    error=err,
                    error_code=("component_failed" if status == "error" else None),
                    notes=notes,
                    producer_version=producer_version,
                    schema_version=schema_version,
                    device_used=_device_used_for_component(name, provider_cfg),
                )
            )
            if status != "ok":
                logger.error(f"VisualProcessor | main | core_provider {name} failed")
                if enforce_requirements and status == "error" and _is_required(req_map, name):
                    logger.error(f"VisualProcessor | main | required core_provider failed: {name}")
                    raise SystemExit(2)
            return

        if name in current_modules:
            module_cfg = config.get(name)
            if module_cfg is None:
                raise ValueError(f"❌ Config entry for module '{name}' not found in YAML")
            logger.info(f"VisualProcessor | main | module {name} start")
            (
                ok,
                err,
                artifacts,
                status,
                notes,
                schema_version,
                producer_version,
                started_at,
                finished_at,
                duration_ms,
            ) = run_module(g_config, name, module_cfg, run_rs_path, gpu_sem)
            manifest.upsert_component(
                ManifestComponent(
                    name=name,
                    kind="module",
                    status=status,
                    started_at=started_at,
                    finished_at=finished_at,
                    duration_ms=duration_ms,
                    artifacts=artifacts,
                    error=err,
                    error_code=("component_failed" if status == "error" else None),
                    notes=notes,
                    producer_version=producer_version,
                    schema_version=schema_version,
                    device_used=_device_used_for_component(name, module_cfg),
                )
            )
            if not ok:
                logger.error(f"VisualProcessor | main | module {name} failed")
                if enforce_requirements and status == "error" and _is_required(req_map, name):
                    logger.error(f"VisualProcessor | main | required module failed: {name}")
                    raise SystemExit(2)
            return

        # Unknown/unenabled => ignore
        logger.debug(f"VisualProcessor | main | skipping component not enabled: {name}")

    if exec_order:
        logger.info(f"VisualProcessor | main | PR-6: executing by DAG order (len={len(exec_order)})")
        for name in exec_order:
            if name in enabled_set:
                _run_one_component(name)
        # Run any remaining enabled components not covered by exec_order
        remaining = [n for n in sorted(enabled_set) if n not in exec_order]
        if remaining:
            logger.warning(f"VisualProcessor | main | PR-6: exec_order missing enabled components: {remaining}")
            for n in remaining:
                _run_one_component(n)
    else:
        # Backward-compatible behavior: keep old module scheduling (parallelism).
        if current_core:
            for provider in current_core:
                _run_one_component(provider)

        # Modules can be run in parallel (intra-video), with GPU gating.
        if current_modules:
            has_deps = any((MODULE_DEPS.get(m) or []) for m in current_modules)
            if has_deps:
                logger.info("VisualProcessor | main | module deps detected → running modules sequentially")
                for module in current_modules:
                    _run_one_component(module)
            else:
                max_workers = max(1, int(max_parallel_modules))
                with ThreadPoolExecutor(max_workers=max_workers) as ex:
                    fut_by_name = {}
                    for module in current_modules:
                        module_cfg = config.get(module)
                        if module_cfg is None:
                            raise ValueError(f"❌ Config entry for module '{module}' not found in YAML")
                        logger.info(f"VisualProcessor | main | scheduling module: {module}")
                        fut = ex.submit(run_module, g_config, module, module_cfg, run_rs_path, gpu_sem)
                        fut_by_name[fut] = module

                    for fut in as_completed(list(fut_by_name.keys())):
                        module = fut_by_name[fut]
                        try:
                            (
                                ok,
                                err,
                                artifacts,
                                status,
                                notes,
                                schema_version,
                                producer_version,
                                started_at,
                                finished_at,
                                duration_ms,
                            ) = fut.result()
                        except Exception as e:
                            ok = False
                            err = str(e)
                            artifacts = []
                            status = "error"
                            notes = "scheduler exception"
                            schema_version = None
                            producer_version = None
                            started_at = _utc_iso_now()
                            finished_at = _utc_iso_now()
                            duration_ms = 0

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
                                error_code=("exception" if status == "error" and notes == "scheduler exception" else ("component_failed" if status == "error" else None)),
                                notes=notes,
                                producer_version=producer_version,
                                schema_version=schema_version,
                                device_used=_device_used_for_component(module, config.get(module) or {}),
                            )
                        )
                        if not ok:
                            logger.error(f"VisualProcessor | main | module {module} failed")
                            if enforce_requirements and status == "error" and _is_required(req_map, module):
                                logger.error(f"VisualProcessor | main | required module failed: {module}")
                                raise SystemExit(2)
