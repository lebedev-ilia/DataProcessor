import os
import subprocess
import shutil
import sys
import tempfile
import uuid
import hashlib
import time

_path = os.path.dirname(__file__)

def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"Required executable not found in PATH: {name}")

def _probe_video_duration_sec(video_path: str) -> float:
    """
    Prod behavior: fail-fast early for too-long videos (before Segmenter).
    Uses ffprobe (required).
    """
    _require_executable("ffprobe")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        os.path.abspath(video_path),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {p.stderr.strip()}")
    try:
        return float((p.stdout or "").strip())
    except Exception as e:
        raise RuntimeError(f"ffprobe returned invalid duration: {p.stdout!r}") from e

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video-path', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default=f"{_path}/Segmenter/data", help='Base output directory for Segmenter')
    parser.add_argument('--chunk-size', type=int, default=64, help='Batch size for storing frames (union frames)')

    parser.add_argument('--visual-cfg-path', type=str, default=f"{_path}/VisualProcessor/config.yaml", help='Path to VisualProcessor/config.yaml')
    parser.add_argument('--profile-path', type=str, default=None, help='Optional analysis profile YAML (required/optional components)')
    parser.add_argument('--dag-path', type=str, default=f"{_path}/docs/reference/component_graph.yaml", help='Path to component_graph.yaml (PR-6)')
    parser.add_argument('--dag-stage', type=str, default="baseline", help='DAG stage: baseline|v1|v2 (PR-6)')
    parser.add_argument('--platform-id', type=str, default="youtube")
    parser.add_argument('--video-id', type=str, default=None)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--sampling-policy-version', type=str, default="v1")
    parser.add_argument('--dataprocessor-version', type=str, default="unknown")
    parser.add_argument('--analysis-fps', type=float, default=None, help="Analysis fps for Segmenter metadata (default: use source_fps)")
    parser.add_argument('--analysis-width', type=int, default=None, help="Resize width for analysis timeline (optional)")
    parser.add_argument('--analysis-height', type=int, default=None, help="Resize height for analysis timeline (optional)")

    parser.add_argument('--rs-base', type=str, default=f"{_path}/VisualProcessor/result_store", help='Base result_store for VisualProcessor (per-run will be inside)')
    parser.add_argument('--run-audio', action='store_true', help='Also run AudioProcessor Tier-0 extractors (clap/tempo/loudness) into the same per-run result_store')
    parser.add_argument('--audio-device', type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument('--audio-extractors', type=str, default="clap,tempo,loudness", help='Comma-separated audio extractor keys for AudioProcessor (clap,tempo,loudness)')
    parser.add_argument('--run-text', action='store_true', help='Also run TextProcessor into the same per-run result_store')
    parser.add_argument('--text-input-json', type=str, default=None, help='Path to TextProcessor VideoDocument JSON')
    parser.add_argument('--text-enable-embeddings', action='store_true', help='Enable GPU-heavy text embedders (optional)')
    args = parser.parse_args()

    # Prod guardrail: reject videos longer than 20 minutes before any heavy work.
    dur = _probe_video_duration_sec(args.video_path)
    if dur > 20.0 * 60.0:
        raise RuntimeError(f"Video too long for baseline (>{20} min): duration_sec={dur}")

    root_path = os.path.abspath(_path)
    video_id = args.video_id
    if not video_id:
        video_id = os.path.splitext(os.path.basename(args.video_path))[0]
    run_id = args.run_id or uuid.uuid4().hex[:12]

    # ----------------------------
    # PR-5: state-files/state-manager (Level2+Level3)
    # We store state under: <runs_root>/state/<platform>/<video>/<run>/...
    # runs_root is derived from rs-base: e.g. rs-base=_runs/result_store -> runs_root=_runs
    # ----------------------------
    runs_root = os.path.dirname(os.path.abspath(args.rs_base))
    try:
        from storage.fs import FileSystemStorage
        from storage.paths import KeyLayout
        from state.managers import RunStateManager, ProcessorStateManager
        from state.enums import Status
    except Exception:
        FileSystemStorage = None  # type: ignore
        KeyLayout = None  # type: ignore
        RunStateManager = None  # type: ignore
        ProcessorStateManager = None  # type: ignore
        Status = None  # type: ignore

    state_storage = FileSystemStorage(runs_root) if FileSystemStorage else None
    state_layout = KeyLayout(prefix="") if KeyLayout else None
    run_state_mgr = None
    proc_mgrs = {}

    # Stable run config hash (shared across Segmenter/Visual/Audio/Text for idempotency)
    def _sha256_text(s: str) -> str:
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    # Optional profile (PR-4): can override processor enablement and visual cfg path.
    profile = None
    if args.profile_path:
        with open(args.profile_path, "r", encoding="utf-8") as f:
            profile = yaml.safe_load(f) or {}

    visual_cfg_path = args.visual_cfg_path
    if isinstance(profile, dict):
        vis = profile.get("visual") or {}
        if isinstance(vis, dict) and vis.get("cfg_path"):
            visual_cfg_path = str(vis.get("cfg_path"))

        procs = profile.get("processors") or {}
        if isinstance(procs, dict):
            audio_cfg = procs.get("audio") or {}
            if isinstance(audio_cfg, dict) and audio_cfg.get("enabled") is True:
                args.run_audio = True
            if isinstance(audio_cfg, dict) and audio_cfg.get("enabled") is False:
                args.run_audio = False
            text_cfg = procs.get("text") or {}
            if isinstance(text_cfg, dict) and text_cfg.get("enabled") is True:
                args.run_text = True
            if isinstance(text_cfg, dict) and text_cfg.get("enabled") is False:
                args.run_text = False

    with open(visual_cfg_path, "r", encoding="utf-8") as f:
        vp_cfg_for_hash = yaml.safe_load(f) or {}

    # PR-6: load DAG and compute Visual execution order (subset of enabled components)
    exec_order: list[str] = []
    try:
        with open(args.dag_path, "r", encoding="utf-8") as f:
            dag_yaml = yaml.safe_load(f) or {}
        from dag.component_graph import ComponentGraph

        g = ComponentGraph.from_yaml_dict(dag_yaml, stage=str(args.dag_stage))
        enabled_visual: set[str] = set()
        enabled_visual.update([k for k, v in (vp_cfg_for_hash.get("core_providers") or {}).items() if v])
        enabled_visual.update([k for k, v in (vp_cfg_for_hash.get("modules") or {}).items() if v])

        # Build execution order only for nodes known to the DAG.
        # Unknown enabled components are allowed in MVP but will be executed after DAG-ordered ones.
        known_enabled = [n for n in enabled_visual if n in g.by_name]
        exec_order = [n for n in g.topo_order(known_enabled) if n in enabled_visual]
    except Exception:
        exec_order = []
    cfg_for_hash = {
        "chunk_size": int(args.chunk_size),
        "sampling_policy_version": str(args.sampling_policy_version),
        "dataprocessor_version": str(args.dataprocessor_version),
        "analysis_fps": args.analysis_fps,
        "analysis_width": args.analysis_width,
        "analysis_height": args.analysis_height,
        "visual_cfg": vp_cfg_for_hash,
        "profile": profile,
        "dag_stage": str(args.dag_stage),
        "dag_path": str(args.dag_path),
        "run_audio": bool(args.run_audio),
        "audio_device": str(args.audio_device),
        "audio_extractors": str(args.audio_extractors),
        "run_text": bool(args.run_text),
        "text_input_json": os.path.abspath(args.text_input_json) if args.text_input_json else None,
        "text_enable_embeddings": bool(args.text_enable_embeddings),
    }
    config_hash = _sha256_text(yaml.safe_dump(cfg_for_hash, sort_keys=True, allow_unicode=True))[:16]

    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Unified per-run result_store directory (single source of truth for all processors)
    run_rs_path = os.path.join(os.path.abspath(args.rs_base), args.platform_id, video_id, run_id)
    os.makedirs(run_rs_path, exist_ok=True)
    if state_storage and state_layout and RunStateManager and ProcessorStateManager and Status:
        run_meta = {
            "platform_id": args.platform_id,
            "video_id": video_id,
            "run_id": run_id,
            "config_hash": config_hash,
            "sampling_policy_version": args.sampling_policy_version,
            "dataprocessor_version": str(args.dataprocessor_version),
            "created_at": created_at,
        }
        run_state_mgr = RunStateManager(
            storage=state_storage,
            layout=state_layout,
            platform_id=args.platform_id,
            video_id=video_id,
            run_id=run_id,
            run_meta=run_meta,
        )
        run_state_mgr.init(["segmenter", "audio", "text", "visual"])
        for p in ("segmenter", "audio", "text", "visual"):
            proc_mgrs[p] = ProcessorStateManager(
                storage=state_storage,
                layout=state_layout,
                platform_id=args.platform_id,
                video_id=video_id,
                run_id=run_id,
                processor_name=p,
                run_meta=run_meta,
            )
            # Materialize initial waiting state-files (Level-3).
            proc_mgrs[p].flush()
            run_state_mgr.merge_processor_state(p, proc_mgrs[p].state)

    # 1) Segmenter (union-sampled frames_dir)
    if proc_mgrs.get("segmenter") and run_state_mgr and Status:
        proc_mgrs["segmenter"].set_status(Status.running, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        run_state_mgr.merge_processor_state("segmenter", proc_mgrs["segmenter"].state)
    seg_cmd = [
        sys.executable,
        f"{_path}/Segmenter/segmenter.py",
        "--video-path", args.video_path,
        "--output", args.output,
        "--chunk-size", str(args.chunk_size),
        "--visual-cfg-path", visual_cfg_path,
        "--platform-id", args.platform_id,
        "--video-id", video_id,
        "--run-id", run_id,
        "--sampling-policy-version", args.sampling_policy_version,
        "--config-hash", config_hash,
        "--dataprocessor-version", str(args.dataprocessor_version),
    ]
    if args.analysis_fps is not None:
        seg_cmd.extend(["--analysis-fps", str(args.analysis_fps)])
    if args.analysis_width is not None:
        seg_cmd.extend(["--analysis-width", str(args.analysis_width)])
    if args.analysis_height is not None:
        seg_cmd.extend(["--analysis-height", str(args.analysis_height)])
    t0 = time.time()
    r = subprocess.run(seg_cmd, check=False)
    seg_duration_ms = int((time.time() - t0) * 1000)
    if r.returncode == 0:
        if proc_mgrs.get("segmenter") and run_state_mgr and Status:
            proc_mgrs["segmenter"].set_status(
                Status.success,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                duration_ms=seg_duration_ms,
            )
            run_state_mgr.merge_processor_state("segmenter", proc_mgrs["segmenter"].state)
    elif r.returncode == 10:
        # Segmenter-level skip (e.g., video cannot be opened/decoded).
        if proc_mgrs.get("segmenter") and run_state_mgr and Status:
            proc_mgrs["segmenter"].set_status(
                Status.skipped,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                duration_ms=seg_duration_ms,
                error="segmenter_skipped",
                error_code="video_unreadable",
            )
            run_state_mgr.merge_processor_state("segmenter", proc_mgrs["segmenter"].state)
        # Do not proceed to audio/text/visual if Segmenter did not produce frames_dir/audio.
        raise SystemExit(0)
    else:
        if proc_mgrs.get("segmenter") and run_state_mgr and Status:
            proc_mgrs["segmenter"].set_status(
                Status.error,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                duration_ms=seg_duration_ms,
                error=f"exit={r.returncode}",
                error_code="non_zero_exit",
            )
            run_state_mgr.merge_processor_state("segmenter", proc_mgrs["segmenter"].state)
        raise RuntimeError(f"Segmenter failed (exit={r.returncode})")

    frames_dir = os.path.join(args.output, video_id, "video")

    # Create/merge manifest.json early so Audio/Text/Visual can upsert without racing on first write.
    # (VisualProcessor and Audio/Text CLIs will load+merge if it already exists.)
    try:
        vp_root = Path(__file__).resolve().parent / "VisualProcessor"
        if str(vp_root) not in sys.path:
            sys.path.insert(0, str(vp_root))
        from utils.manifest import RunManifest  # type: ignore

        manifest_path = os.path.join(run_rs_path, "manifest.json")
        manifest = RunManifest(
            path=manifest_path,
            run_meta={
                "platform_id": args.platform_id,
                "video_id": video_id,
                "run_id": run_id,
                "config_hash": config_hash,
                "sampling_policy_version": args.sampling_policy_version,
                "dataprocessor_version": str(args.dataprocessor_version),
                "created_at": created_at,
                "frames_dir": os.path.join(os.path.abspath(args.output), video_id, "video"),
            },
        )
        manifest.flush()
    except Exception:
        # Best-effort: processors still create/merge manifests themselves.
        pass

    # 1.5) AudioProcessor (optional): write per-run NPZ artifacts into the same result_store
    if args.run_audio:
        audio_required = False
        if isinstance(profile, dict):
            audio_required = bool(((profile.get("processors") or {}).get("audio") or {}).get("required") is True)
        audio_cmd = [
            sys.executable,
            f"{_path}/AudioProcessor/run_cli.py",
            "--video-path", args.video_path,
            "--frames-dir", os.path.join(os.path.abspath(args.output), video_id),
            "--rs-base", os.path.abspath(args.rs_base),
            "--run-rs-path", run_rs_path,
            "--platform-id", args.platform_id,
            "--video-id", video_id,
            "--run-id", run_id,
            "--sampling-policy-version", args.sampling_policy_version,
            "--config-hash", config_hash,
            "--dataprocessor-version", str(args.dataprocessor_version),
            "--device", args.audio_device,
            "--extractors", args.audio_extractors,
        ]
        if proc_mgrs.get("audio") and run_state_mgr and Status:
            proc_mgrs["audio"].set_status(Status.running, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            run_state_mgr.merge_processor_state("audio", proc_mgrs["audio"].state)
        t0 = time.time()
        r = subprocess.run(audio_cmd, check=False)
        if proc_mgrs.get("audio") and run_state_mgr and Status:
            st = Status.success if r.returncode == 0 else Status.error
            proc_mgrs["audio"].set_status(
                st,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                duration_ms=int((time.time() - t0) * 1000),
                error=None if r.returncode == 0 else f"exit={r.returncode}",
                error_code=None if r.returncode == 0 else "non_zero_exit",
            )
            run_state_mgr.merge_processor_state("audio", proc_mgrs["audio"].state)
        if audio_required and r.returncode != 0:
            raise RuntimeError(f"AudioProcessor failed for required=true (exit={r.returncode})")

    # 1.6) TextProcessor (optional): write per-run NPZ artifact into the same result_store
    if args.run_text:
        text_required = False
        if isinstance(profile, dict):
            text_required = bool(((profile.get("processors") or {}).get("text") or {}).get("required") is True)
        if not args.text_input_json:
            raise ValueError("--run-text requires --text-input-json")
        text_cmd = [
            sys.executable,
            f"{_path}/TextProcessor/run_cli.py",
            "--input-json", os.path.abspath(args.text_input_json),
            "--rs-base", os.path.abspath(args.rs_base),
            "--run-rs-path", run_rs_path,
            "--platform-id", args.platform_id,
            "--video-id", video_id,
            "--run-id", run_id,
            "--sampling-policy-version", args.sampling_policy_version,
            "--config-hash", config_hash,
            "--dataprocessor-version", str(args.dataprocessor_version),
        ]
        if args.text_enable_embeddings:
            text_cmd.append("--enable-embeddings")
        if proc_mgrs.get("text") and run_state_mgr and Status:
            proc_mgrs["text"].set_status(Status.running, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            run_state_mgr.merge_processor_state("text", proc_mgrs["text"].state)
        t0 = time.time()
        r = subprocess.run(text_cmd, check=False)
        if proc_mgrs.get("text") and run_state_mgr and Status:
            st = Status.success if r.returncode == 0 else Status.error
            proc_mgrs["text"].set_status(
                st,
                finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                duration_ms=int((time.time() - t0) * 1000),
                error=None if r.returncode == 0 else f"exit={r.returncode}",
                error_code=None if r.returncode == 0 else "non_zero_exit",
            )
            run_state_mgr.merge_processor_state("text", proc_mgrs["text"].state)
        if text_required and r.returncode != 0:
            raise RuntimeError(f"TextProcessor failed for required=true (exit={r.returncode})")

    # 2) VisualProcessor: generate a temp cfg overriding global paths/ids
    vp_cfg = dict(vp_cfg_for_hash or {})
    vp_cfg = vp_cfg or {}
    vp_cfg["global"] = vp_cfg.get("global") or {}
    vp_cfg["global"].update(
        {
            "root_path": root_path,
            "frames_dir": frames_dir,
            # VisualProcessor expects rs_path; we pass the per-run directory to avoid re-deriving.
            "rs_path": run_rs_path,
            "rs_path_is_run_dir": True,
            "platform_id": args.platform_id,
            "video_id": video_id,
            "run_id": run_id,
            "config_hash": config_hash,
            "sampling_policy_version": args.sampling_policy_version,
            "dataprocessor_version": str(args.dataprocessor_version),
        }
    )
    # PR-8: pass resolved model mapping into VisualProcessor runtime cfg (and manifest.run).
    # MVP source-of-truth is profile YAML; later this comes from DB.
    if isinstance(profile, dict):
        rmm = profile.get("resolved_model_mapping")
        if isinstance(rmm, dict) and rmm:
            vp_cfg["resolved_model_mapping"] = rmm
    # PR-4: pass requirements map to VisualProcessor (enables required/optional enforcement).
    if isinstance(profile, dict):
        vis = profile.get("visual") or {}
        if isinstance(vis, dict):
            req = vis.get("requirements")
            if isinstance(req, dict) and req:
                vp_cfg["requirements"] = req

    # PR-6: pass execution order into VisualProcessor (optional)
    if exec_order:
        vp_cfg["execution_order"] = exec_order

    fd, tmp_cfg_path = tempfile.mkstemp(prefix="vp_runtime_", suffix=".yaml")
    os.close(fd)
    try:
        with open(tmp_cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(vp_cfg, f, sort_keys=False, allow_unicode=True)

        vp_cmd = [
            sys.executable,
            f"{_path}/VisualProcessor/main.py",
            "--cfg-path",
            tmp_cfg_path,
        ]
        if proc_mgrs.get("visual") and run_state_mgr and Status:
            proc_mgrs["visual"].set_status(Status.running, started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            run_state_mgr.merge_processor_state("visual", proc_mgrs["visual"].state)
        t0 = time.time()
        try:
            subprocess.run(vp_cmd, check=True)
            if proc_mgrs.get("visual") and run_state_mgr and Status:
                proc_mgrs["visual"].set_status(
                    Status.success,
                    finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    duration_ms=int((time.time() - t0) * 1000),
                )
        except Exception as e:
            if proc_mgrs.get("visual") and run_state_mgr and Status:
                proc_mgrs["visual"].set_status(
                    Status.error,
                    finished_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    duration_ms=int((time.time() - t0) * 1000),
                    error=str(e),
                    error_code="exception",
                )
                run_state_mgr.merge_processor_state("visual", proc_mgrs["visual"].state)
            raise

        # After VisualProcessor finished, sync component statuses from manifest into processor state.
        try:
            manifest_path = os.path.join(os.path.abspath(args.rs_base), args.platform_id, video_id, run_id, "manifest.json")
            if os.path.exists(manifest_path) and proc_mgrs.get("visual") and run_state_mgr and Status:
                import json as _json
                with open(manifest_path, "r", encoding="utf-8") as f:
                    m = _json.load(f) or {}
                comps = m.get("components") or []
                if isinstance(comps, list):
                    for c in comps:
                        if not isinstance(c, dict):
                            continue
                        name = c.get("name")
                        st = c.get("status")
                        if not isinstance(name, str) or not name:
                            continue
                        if st == "ok":
                            sst = Status.success
                        elif st == "empty":
                            sst = Status.empty
                        elif st == "error":
                            sst = Status.error
                        else:
                            sst = Status.error
                        proc_mgrs["visual"].upsert_component(
                            component_name=name,
                            status=sst,
                            artifacts=c.get("artifacts") if isinstance(c.get("artifacts"), list) else None,
                            error=c.get("error"),
                            error_code=c.get("error_code"),
                            notes=c.get("notes"),
                            device_used=c.get("device_used"),
                            started_at=c.get("started_at"),
                            finished_at=c.get("finished_at"),
                            duration_ms=c.get("duration_ms"),
                        )
                run_state_mgr.merge_processor_state("visual", proc_mgrs["visual"].state)
        except Exception:
            # best-effort: do not break successful run if state sync fails
            if proc_mgrs.get("visual") and run_state_mgr and Status:
                run_state_mgr.merge_processor_state("visual", proc_mgrs["visual"].state)
    finally:
        try:
            os.remove(tmp_cfg_path)
        except Exception:
            pass