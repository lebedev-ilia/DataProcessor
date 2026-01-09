from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict

from dp_queue.celery_app import celery_app
from dp_queue.payloads import ProcessVideoPayload


import logging

LOGGER = logging.getLogger(__name__)


@celery_app.task(
    name="dataprocessor.process_video_job",
    bind=True,
    autoretry_for=(RuntimeError,),
    retry_backoff=True,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
)
def process_video_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery task that runs DataProcessor root `main.py` in a subprocess.
    MVP: in future we will replace subprocess orchestration with native python runner.
    """
    p = ProcessVideoPayload.from_dict(payload)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    main_py = os.path.join(repo_root, "main.py")

    cmd = [sys.executable, main_py, *p.to_cli_args()]
    LOGGER.info("process_video_job | run_id=%s | cmd=%s", p.run_id, " ".join(cmd))

    r = subprocess.run(cmd, cwd=repo_root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = "\n".join((r.stdout or "").splitlines()[-80:])

    if r.returncode != 0:
        raise RuntimeError(f"DataProcessor failed (exit={r.returncode}). tail:\n{tail}")

    return {"status": "ok", "run_id": p.run_id, "video_id": p.video_id, "platform_id": p.platform_id, "tail": tail}


