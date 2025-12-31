import os
import subprocess
import sys
import tempfile
import uuid

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
    args = parser.parse_args()

    root_path = os.path.abspath(_path)
    video_id = args.video_id
    if not video_id:
        video_id = os.path.splitext(os.path.basename(args.video_path))[0]
    run_id = args.run_id or uuid.uuid4().hex[:12]

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
            "--device", args.audio_device,
            "--extractors", args.audio_extractors,
        ]
        subprocess.run(audio_cmd, check=False)

    # 2) VisualProcessor: generate a temp cfg overriding global paths/ids
    with open(args.visual_cfg_path, "r", encoding="utf-8") as f:
        vp_cfg = yaml.safe_load(f)
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