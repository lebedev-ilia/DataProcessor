import os
import sys

_path = os.path.dirname(__file__)

if _path not in sys.path:
    sys.path.append(_path)

import yaml
import argparse
import logging
import subprocess

from utils.logger import get_logger
from utils.results_store import ResultsStore

logger = get_logger("VisualProcessor")


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

    kwargs = []
    for k, v in cfg.items():
        if v is None or v == "False" or v is False:
            continue
        key = f"--{k.replace('_', '-')}"
        if v is True or v == "True":
            kwargs.append(key)
        else:
            kwargs.extend([key, str(v)])

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
        return True
    except Exception as e:
        logger.error(f"VisualProcessor | main | run_module | Error: {e}")
        return False


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
        return True
    except Exception as e:
        logger.error(f"VisualProcessor | main | run_core_provider | Error: {e}")
        return False


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

    g_config = config["global"]

    # -----------------------------
    # Фаза 1 — core‑провайдеры
    # -----------------------------
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

            status = run_core_provider(g_config, provider, provider_cfg)
            if not status:
                logger.error(f"VisualProcessor | main | core_provider {provider} failed")

    # # -----------------------------
    # # Фаза 2 — модульные анализаторы
    # # -----------------------------
    # current_modules = get_current_modules(config)

    # logger.info(f"VisualProcessor | main | Текущие модули:")

    # for module in current_modules:
    #     logger.info(f"            {module}")

    # for module in current_modules:
    #     logger.info(f"VisualProcessor | main | {module} start")

    #     module_cfg = config.get(module)

    #     logger.info(f"VisualProcessor | main | {module} config:")

    #     for k, v in module_cfg.items():
    #         logger.info(f"            {k}: {v}")

    #     if module_cfg is None:
    #         raise ValueError(f"❌ Config entry for module '{module}' not found in YAML")

    #     status = run_module(g_config, module, module_cfg)
