import os
import subprocess
import sys
import tempfile
import uuid
import hashlib

_path = os.path.dirname(__file__)

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video-path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default=f"{_path}/Segmenter/data", help='Base output directory for Segmenter')
    parser.add_argument('--chunk-size', type=int, default=64, help='Batch size for storing frames (union frames)')

    parser.add_argument('--visual-cfg-path', type=str, default=f"{_path}/VisualProcessor/config.yaml", help='Path to VisualProcessor/config.yaml')
    parser.add_argument('--platform-id', type=str, default="youtube")
    parser.add_argument('--video-id', type=str, default=None)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--sampling-policy-version', type=str, default="v1")

    parser.add_argument('--rs-base', type=str, default=f"{_path}/VisualProcessor/result_store", help='Base result_store for VisualProcessor (per-run will be inside)')
    parser.add_argument('--run-audio', action='store_true', help='Also run AudioProcessor Tier-0 extractors (clap/tempo/loudness) into the same per-run result_store')
    parser.add_argument('--audio-device', type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument('--audio-extractors', type=str, default="clap,tempo,loudness", help='Comma-separated audio extractor keys for AudioProcessor (clap,tempo,loudness)')
    parser.add_argument('--run-text', action='store_true', help='Also run TextProcessor into the same per-run result_store')
    parser.add_argument('--text-input-json', type=str, default=None, help='Path to TextProcessor VideoDocument JSON')
    parser.add_argument('--text-enable-embeddings', action='store_true', help='Enable GPU-heavy text embedders (optional)')
    args = parser.parse_args()

    root_path = os.path.abspath(_path)
    video_id = args.video_id
    if not video_id:
        video_id = os.path.splitext(os.path.basename(args.video_path))[0]
    run_id = args.run_id or uuid.uuid4().hex[:12]

    # Stable run config hash (shared across Segmenter/Visual/Audio/Text for idempotency)
    def _sha256_text(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    with open(args.visual_cfg_path, "r", encoding="utf-8") as f:
        vp_cfg_for_hash = yaml.safe_load(f) or {}
    cfg_for_hash = {
        "chunk_size": int(args.chunk_size),
        "sampling_policy_version": str(args.sampling_policy_version),
        "visual_cfg": vp_cfg_for_hash,
        "run_audio": bool(args.run_audio),
        "audio_device": str(args.audio_device),
        "audio_extractors": str(args.audio_extractors),
        "run_text": bool(args.run_text),
        "text_input_json": os.path.abspath(args.text_input_json) if args.text_input_json else None,
        "text_enable_embeddings": bool(args.text_enable_embeddings),
    }
    config_hash = _sha256_text(yaml.safe_dump(cfg_for_hash, sort_keys=True, allow_unicode=True))[:16]

    # 1) Segmenter (union-sampled frames_dir)
    seg_cmd = [
        sys.executable,
        f"{_path}/Segmenter/segmenter.py",
        "--video-path", args.video_path,
        "--output", args.output,
        "--chunk-size", str(args.chunk_size),
        "--visual-cfg-path", args.visual_cfg_path,
        "--platform-id", args.platform_id,
        "--video-id", video_id,
        "--run-id", run_id,
        "--sampling-policy-version", args.sampling_policy_version,
        "--config-hash", config_hash,
    ]
    subprocess.run(seg_cmd, check=True)

    frames_dir = os.path.join(args.output, video_id, "video")

    # 1.5) AudioProcessor (optional): write per-run NPZ artifacts into the same result_store
    if args.run_audio:
        audio_cmd = [
            sys.executable,
            f"{_path}/AudioProcessor/run_cli.py",
            "--video-path", args.video_path,
            "--rs-base", os.path.abspath(args.rs_base),
            "--platform-id", args.platform_id,
            "--video-id", video_id,
            "--run-id", run_id,
            "--sampling-policy-version", args.sampling_policy_version,
            "--config-hash", config_hash,
            "--device", args.audio_device,
            "--extractors", args.audio_extractors,
        ]
        subprocess.run(audio_cmd, check=False)

    # 1.6) TextProcessor (optional): write per-run NPZ artifact into the same result_store
    if args.run_text:
        if not args.text_input_json:
            raise ValueError("--run-text requires --text-input-json")
        text_cmd = [
            sys.executable,
            f"{_path}/TextProcessor/run_cli.py",
            "--input-json", os.path.abspath(args.text_input_json),
            "--rs-base", os.path.abspath(args.rs_base),
            "--platform-id", args.platform_id,
            "--video-id", video_id,
            "--run-id", run_id,
            "--sampling-policy-version", args.sampling_policy_version,
            "--config-hash", config_hash,
        ]
        if args.text_enable_embeddings:
            text_cmd.append("--enable-embeddings")
        subprocess.run(text_cmd, check=False)

    # 2) VisualProcessor: generate a temp cfg overriding global paths/ids
    vp_cfg = dict(vp_cfg_for_hash or {})
    vp_cfg = vp_cfg or {}
    vp_cfg["global"] = vp_cfg.get("global") or {}
    vp_cfg["global"].update(
        {
            "root_path": root_path,
            "frames_dir": frames_dir,
            "rs_path": os.path.abspath(args.rs_base),
            "platform_id": args.platform_id,
            "video_id": video_id,
            "run_id": run_id,
            "config_hash": config_hash,
            "sampling_policy_version": args.sampling_policy_version,
        }
    )

    fd, tmp_cfg_path = tempfile.mkstemp(prefix="vp_runtime_", suffix=".yaml", dir=root_path)
    os.close(fd)
    with open(tmp_cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(vp_cfg, f, sort_keys=False, allow_unicode=True)

    vp_cmd = [
        sys.executable,
        f"{_path}/VisualProcessor/main.py",
        "--cfg-path",
        tmp_cfg_path,
    ]
    subprocess.run(vp_cmd, check=True)