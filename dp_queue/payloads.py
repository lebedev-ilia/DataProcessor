from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ProcessVideoPayload:
    """
    Minimal payload for PR-7 Celery task.
    We keep it intentionally close to root `main.py` CLI args.
    """

    video_path: str
    platform_id: str
    video_id: str
    run_id: str

    # optional overrides
    rs_base: str = "./_runs/result_store"
    output: str = "./_runs/segmenter_out"
    sampling_policy_version: str = "v1"
    dataprocessor_version: str = "unknown"
    analysis_fps: float = 30.0
    analysis_width: int = 568
    analysis_height: int = 320
    chunk_size: int = 64

    visual_cfg_path: str = "./VisualProcessor/config_pr2_min.yaml"
    profile_path: Optional[str] = None
    dag_path: str = "./docs/reference/component_graph.yaml"
    dag_stage: str = "baseline"

    run_audio: bool = False
    audio_device: str = "auto"
    audio_extractors: str = "clap,tempo,loudness"

    run_text: bool = False
    text_input_json: Optional[str] = None
    text_enable_embeddings: bool = False

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ProcessVideoPayload":
        if not isinstance(d, dict):
            raise ValueError("payload must be a dict")
        required = ["video_path", "platform_id", "video_id", "run_id"]
        missing = [k for k in required if not d.get(k)]
        if missing:
            raise ValueError(f"payload missing required keys: {missing}")
        return ProcessVideoPayload(
            video_path=str(d["video_path"]),
            platform_id=str(d["platform_id"]),
            video_id=str(d["video_id"]),
            run_id=str(d["run_id"]),
            rs_base=str(d.get("rs_base") or "./_runs/result_store"),
            output=str(d.get("output") or "./_runs/segmenter_out"),
            sampling_policy_version=str(d.get("sampling_policy_version") or "v1"),
            dataprocessor_version=str(d.get("dataprocessor_version") or "unknown"),
            analysis_fps=float(d.get("analysis_fps") or 30.0),
            analysis_width=int(d.get("analysis_width") or 568),
            analysis_height=int(d.get("analysis_height") or 320),
            chunk_size=int(d.get("chunk_size") or 64),
            visual_cfg_path=str(d.get("visual_cfg_path") or "./VisualProcessor/config_pr2_min.yaml"),
            profile_path=str(d["profile_path"]) if d.get("profile_path") else None,
            dag_path=str(d.get("dag_path") or "./docs/reference/component_graph.yaml"),
            dag_stage=str(d.get("dag_stage") or "baseline"),
            run_audio=bool(d.get("run_audio") or False),
            audio_device=str(d.get("audio_device") or "auto"),
            audio_extractors=str(d.get("audio_extractors") or "clap,tempo,loudness"),
            run_text=bool(d.get("run_text") or False),
            text_input_json=str(d["text_input_json"]) if d.get("text_input_json") else None,
            text_enable_embeddings=bool(d.get("text_enable_embeddings") or False),
        )

    def to_cli_args(self) -> list[str]:
        args = [
            "--video-path",
            self.video_path,
            "--output",
            self.output,
            "--chunk-size",
            str(self.chunk_size),
            "--visual-cfg-path",
            self.visual_cfg_path,
            "--platform-id",
            self.platform_id,
            "--video-id",
            self.video_id,
            "--run-id",
            self.run_id,
            "--sampling-policy-version",
            self.sampling_policy_version,
            "--dataprocessor-version",
            self.dataprocessor_version,
            "--analysis-fps",
            str(self.analysis_fps),
            "--analysis-width",
            str(self.analysis_width),
            "--analysis-height",
            str(self.analysis_height),
            "--rs-base",
            self.rs_base,
            "--dag-path",
            self.dag_path,
            "--dag-stage",
            self.dag_stage,
        ]
        if self.profile_path:
            args.extend(["--profile-path", self.profile_path])

        if self.run_audio:
            args.append("--run-audio")
            args.extend(["--audio-device", self.audio_device])
            args.extend(["--audio-extractors", self.audio_extractors])

        if self.run_text:
            args.append("--run-text")
            if not self.text_input_json:
                raise ValueError("run_text=true requires text_input_json")
            args.extend(["--text-input-json", self.text_input_json])
            if self.text_enable_embeddings:
                args.append("--text-enable-embeddings")

        return args


