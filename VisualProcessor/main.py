import os
import yaml
import argparse
import subprocess


def run_module(global_cfg, module_name, module_cfg):
    """
    Запуск модуля через subprocess c его конфигом.
    module_cfg — словарь с параметрами модуля из YAML.
    """
    
    root_path = global_cfg["root_path"]
    frames_dir = global_cfg["frames_dir"]
    rs_path = global_cfg["rs_path"]
    
    module_path = os.path.join(root_path, "modules", module_name)
    venv_name = f".{module_name}_venv"
    module_venv_path = os.path.join(module_path, venv_name)

    # Выбор виртуальной среды: локальная или глобальная
    if os.path.isdir(module_venv_path):
        venv_path = module_venv_path
    else:
        venv_path = os.path.join(root_path, ".global_venv")

    python_exec = os.path.join(venv_path, "bin", "python")
    module_entry = os.path.join(module_path, "main.py")

    kwargs = []
    for k, v in module_cfg.items():
        key = f"--{k.replace('_', '-')}"
        kwargs.extend([key, str(v)])
        
    cmd = [python_exec, module_entry] + \
        kwargs + \
        ["--frames-dir", frames_dir] + \
        ["--rs-path", rs_path]

    print("▶ Running:", " ".join(cmd))

    try:
        subprocess.run(cmd)
        return True
    except:
        return False


def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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

    config = load_config(args.cfg_path)

    root_path = config["global"]["root_path"]

    current_modules = get_current_modules(config)

    for module in current_modules:
        module_cfg = config.get(module)

        if module_cfg is None:
            raise ValueError(f"❌ Config entry for module '{module}' not found in YAML")

        status = run_module(root_path, module, module_cfg)
