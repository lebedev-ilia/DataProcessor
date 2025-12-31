from __future__ import annotations

import os
from typing import Any, Dict, Optional

import psutil

try:
    import pynvml  # type: ignore
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False


def _gpu_snapshot() -> Optional[Dict[str, Any]]:
    if not _NVML_AVAILABLE:
        return None
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name_raw = pynvml.nvmlDeviceGetName(handle)
            name = name_raw.decode("utf-8") if isinstance(name_raw, (bytes, bytearray)) else str(name_raw)
            gpus.append({
                "index": i,
                "name": name,
                "memory_total_mb": int(mem.total / 1024 / 1024),
                "memory_used_mb": int(mem.used / 1024 / 1024),
                "memory_free_mb": int(mem.free / 1024 / 1024),
                "utilization_gpu_percent": int(getattr(util, "gpu", 0)),
                "utilization_mem_percent": int(getattr(util, "memory", 0)),
            })
        return {"gpus": gpus}
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def system_snapshot() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    cpu_times = psutil.cpu_times_percent(interval=None)
    loadavg = None
    try:
        loadavg = os.getloadavg()
    except Exception:
        loadavg = None

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "cpu_times_percent": {
            "user": cpu_times.user,
            "system": cpu_times.system,
            "idle": cpu_times.idle,
        },
        "loadavg": loadavg,
        "ram": {
            "total_mb": int(vm.total / 1024 / 1024 ),
            "available_mb": int(vm.available / 1024 / 1024),
            "used_mb": int(vm.used / 1024 / 1024),
            "percent": float(vm.percent),
        },
        "gpu": _gpu_snapshot(),
    }


def process_memory_bytes() -> int:
    try:
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


