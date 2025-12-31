#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from huggingface_hub import hf_hub_download, HfApi  # type: ignore
import ijson  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent.parent


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_done_from_state_jsonl(path: Path) -> Set[str]:
    """
    Best-effort resume: if we previously reached process:ok (or skip:ok), we skip the video_id next run.
    """
    done: Set[str] = set()
    if not path.exists():
        return done
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                vid = obj.get("video_id")
                if not isinstance(vid, str) or not vid:
                    continue
                stage = obj.get("stage")
                status = obj.get("status")
                if stage == "process" and status == "ok":
                    done.add(vid)
                if stage == "skip" and status == "ok":
                    done.add(vid)
    except Exception:
        return done
    return done


def _parse_done_video_ids_from_out_repo_files(files: List[str], platform_id: str) -> Set[str]:
    """
    Our artifact paths are:
      runs/<platform_id>/<video_id>/<run_id>.tar.gz
    """
    done: Set[str] = set()
    prefix = f"runs/{platform_id}/"
    for p in files:
        if not isinstance(p, str):
            continue
        if not p.startswith(prefix):
            continue
        # runs/<platform>/<video>/<run>.tar.gz
        parts = p.split("/")
        if len(parts) < 4:
            continue
        vid = parts[2]
        if vid:
            done.add(vid)
    return done


def _load_or_build_remote_done_set(
    *,
    api: HfApi,
    token: str,
    out_repo_id: str,
    platform_id: str,
    cache_path: Path,
    refresh: bool,
    state_jsonl: Path,
) -> Set[str]:
    """
    Remote idempotency: build a set of video_ids that already have runs in hf-out-repo.
    We keep a local cache to avoid listing repo files on every invocation.
    """
    if not refresh and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and payload.get("out_repo_id") == out_repo_id and payload.get("platform_id") == platform_id:
                vids = payload.get("done_video_ids", [])
                if isinstance(vids, list):
                    out = {v for v in vids if isinstance(v, str) and v}
                    _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "stage": "remote_index", "status": "ok", "source": "cache", "count": len(out)})
                    return out
        except Exception:
            pass

    _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "stage": "remote_index", "status": "start", "source": "hf_list"})
    files = api.list_repo_files(repo_id=out_repo_id, repo_type="dataset", token=token)
    done = _parse_done_video_ids_from_out_repo_files(files, platform_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "out_repo_id": out_repo_id,
        "platform_id": platform_id,
        "created_at": _utc_iso_now(),
        "files_count": len(files),
        "done_video_ids": sorted(done),
    }
    cache_path.write_text(json.dumps(cache_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "stage": "remote_index", "status": "ok", "source": "hf_list", "count": len(done)})
    return done


def _write_remote_done_cache(cache_path: Path, *, out_repo_id: str, platform_id: str, done_video_ids: Set[str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "out_repo_id": out_repo_id,
        "platform_id": platform_id,
        "created_at": _utc_iso_now(),
        "done_video_ids": sorted(done_video_ids),
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _iter_video_ids_from_main(main_ready_dir: Path) -> Iterable[str]:
    # Stream: each shard is a top-level object {vid: rec}
    for fp in sorted(main_ready_dir.glob("data_*.json")):
        try:
            with fp.open("rb") as f:
                for vid, _rec in ijson.kvitems(f, ""):
                    if isinstance(vid, str) and vid:
                        yield vid
        except Exception:
            continue


def _pack_run_dir(run_dir: Path, out_tar_gz: Path) -> None:
    out_tar_gz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_tar_gz, "w:gz") as tf:
        tf.add(run_dir, arcname=run_dir.name)


@dataclass
class DownloadResult:
    video_id: str
    ok: bool
    path: Optional[str]
    error: Optional[str]


def _download_one(video_id: str, repo_id: str, cache_dir: Path) -> DownloadResult:
    try:
        # Assumption: file is at repo root named <video_id>.mp4
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=f"{video_id}.mp4",
            local_dir=str(cache_dir),
            local_dir_use_symlinks=False,
        )
        return DownloadResult(video_id=video_id, ok=True, path=local_path, error=None)
    except Exception as e:
        return DownloadResult(video_id=video_id, ok=False, path=None, error=str(e))


def _yt_download_video(video_id: str, out_dir: Path, *, max_duration_sec: int = 1200, cookies_path: Optional[str] = None) -> DownloadResult:
    """
    Fallback: download from YouTube using yt-dlp. Output is <video_id>.mp4 if possible.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_tmpl = str(out_dir / f"{video_id}.%(ext)s")
    cmd = [
        "yt-dlp",
        url,
        "-o",
        out_tmpl,
        "--no-part",
        "--merge-output-format",
        "mp4",
        "--max-downloads",
        "1",
        "--match-filter",
        f"duration and duration <= {int(max_duration_sec)}",
        "-f",
        "bv*+ba/b",
    ]
    if cookies_path:
        cmd.extend(["--cookies", cookies_path])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            return DownloadResult(video_id=video_id, ok=False, path=None, error=(r.stderr or r.stdout or "yt-dlp failed"))
        # Find resulting file
        for f in out_dir.glob(f"{video_id}.*"):
            if f.is_file():
                return DownloadResult(video_id=video_id, ok=True, path=str(f), error=None)
        return DownloadResult(video_id=video_id, ok=False, path=None, error="yt-dlp ok but file not found")
    except Exception as e:
        return DownloadResult(video_id=video_id, ok=False, path=None, error=str(e))


def _hf_upload_video_to_dataset(api: HfApi, token: str, repo_id: str, video_id: str, local_path: str) -> None:
    """
    Upload a single video file to HF dataset repo root as <video_id>.mp4 (keeps the 'videosX' contract).
    """
    path_in_repo = f"{video_id}.mp4"
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"add video {video_id}",
        token=token,
    )


def _hf_upload_folder_batch(api: HfApi, token: str, repo_id: str, local_folder: Path, path_in_repo: str, commit_message: str) -> None:
    """
    Upload a folder as a single commit (best-effort). Falls back to per-file upload if needed.
    """
    try:
        api.upload_folder(
            folder_path=str(local_folder),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
            token=token,
        )
        return
    except Exception:
        # Fallback: upload each file
        for f in sorted(local_folder.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(local_folder)
            api.upload_file(
                path_or_fileobj=str(f),
                path_in_repo=str(Path(path_in_repo) / rel),
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=commit_message,
                token=token,
            )


def main() -> int:
    p = argparse.ArgumentParser(description="Batch runner: HF videos -> DataProcessor -> HF artifacts")
    p.add_argument("--main-ready-dir", type=str, required=True)
    p.add_argument("--hf-index", type=str, required=True, help="BatchRunner/hf_video_index.json")
    p.add_argument("--result-store-base", type=str, required=True)
    p.add_argument("--hf-out-repo", type=str, required=True, help="HF dataset repo to upload artifacts to (files-based)")
    p.add_argument("--hf-token", type=str, default=None, help="HF token (or set HF_TOKEN env)")
    p.add_argument("--hf-videos11-repo", type=str, default="Ilialebedev/videos11", help="HF dataset repo to upload missing videos to")
    p.add_argument("--platform-id", type=str, default="youtube")
    p.add_argument("--sampling-policy-version", type=str, default="v1_fixed")
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--download-workers", type=int, default=8)
    p.add_argument("--max-videos", type=int, default=0, help="Limit total processed videos (0=all)")
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--max-duration-sec", type=int, default=1200, help="Skip videos longer than this (used by yt-dlp fallback too)")
    p.add_argument("--yt-cookies", type=str, default=None, help="Optional cookies.txt for yt-dlp fallback downloads")
    p.add_argument("--no-resume", action="store_true", help="Disable resume via local BatchRunner/state.jsonl")
    p.add_argument("--skip-remote-existing", action="store_true", help="Skip video_ids that already exist in hf-out-repo (remote idempotency)")
    p.add_argument("--remote-index-cache", type=str, default="BatchRunner/hf_out_existing.json", help="Local cache of hf-out-repo existing runs")
    p.add_argument("--refresh-remote-index", action="store_true", help="Force refresh remote index (list hf-out-repo files)")
    args = p.parse_args()

    main_ready_dir = Path(args.main_ready_dir).resolve()
    rs_base = Path(args.result_store_base).resolve()
    hf_index_path = Path(args.hf_index).resolve()

    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HF token is required (use --hf-token or env HF_TOKEN)")

    idx_payload = _load_json(hf_index_path, {})
    vid2repo: Dict[str, str] = idx_payload.get("video_id_to_repo") if isinstance(idx_payload, dict) else {}
    if not isinstance(vid2repo, dict) or not vid2repo:
        raise ValueError("hf index missing/invalid: video_id_to_repo")

    interpret_dir = REPO_ROOT / "Interpret"
    excluded_path = interpret_dir / "excluded_videos.json"
    excluded: Set[str] = set(_load_json(excluded_path, []))
    state_jsonl = REPO_ROOT / "BatchRunner" / "state.jsonl"

    api = HfApi(token=token)

    # Resume: skip anything already marked ok in local state.jsonl
    done_by_state: Set[str] = set()
    if not args.no_resume:
        done_by_state = _load_done_from_state_jsonl(state_jsonl)
        if done_by_state:
            _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "stage": "resume", "status": "ok", "count": len(done_by_state)})

    # Remote idempotency: skip anything already present in hf-out-repo
    done_by_remote: Set[str] = set()
    remote_cache_path: Optional[Path] = None
    if args.skip_remote_existing:
        remote_cache_path = (REPO_ROOT / args.remote_index_cache).resolve()
        done_by_remote = _load_or_build_remote_done_set(
            api=api,
            token=token,
            out_repo_id=args.hf_out_repo,
            platform_id=args.platform_id,
            cache_path=remote_cache_path,
            refresh=bool(args.refresh_remote_index),
            state_jsonl=state_jsonl,
        )

    # Local temp dirs
    cache_dir = REPO_ROOT / "BatchRunner" / "_cache_videos"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yt_tmp_dir = REPO_ROOT / "BatchRunner" / "_yt_tmp"
    yt_tmp_dir.mkdir(parents=True, exist_ok=True)
    pack_dir = REPO_ROOT / "BatchRunner" / "_packs"
    pack_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    batch: List[str] = []

    def already_done(video_id: str) -> bool:
        # If any run exists under rs_base/platform/video/*/manifest.json, treat as done (idempotency v1)
        pdir = rs_base / args.platform_id / video_id
        if not pdir.exists():
            return False
        for run_dir in pdir.iterdir():
            if (run_dir / "manifest.json").exists():
                return True
        return False

    def run_one(video_id: str, video_path: Path) -> Tuple[bool, Optional[str]]:
        run_id = f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:6]}"
        cmd = [
            str(REPO_ROOT / "main.py"),
            "--video-path",
            str(video_path),
            "--platform-id",
            args.platform_id,
            "--video-id",
            video_id,
            "--run-id",
            run_id,
            "--sampling-policy-version",
            args.sampling_policy_version,
            "--rs-base",
            str(rs_base),
            "--run-audio",
        ]
        # NOTE: main.py is a python script; invoke with current interpreter.
        full_cmd = [os.fspath(Path(os.sys.executable)), *cmd]
        try:
            subprocess.run(full_cmd, check=True)
            return True, run_id
        except Exception as e:
            return False, str(e)

    def pack_one(video_id: str, run_id: str) -> Optional[Path]:
        run_dir = rs_base / args.platform_id / video_id / run_id
        if not run_dir.exists():
            return None
        # pack folder into tar.gz to reduce HF file count
        tar_path = pack_dir / f"{video_id}__{run_id}.tar.gz"
        _pack_run_dir(run_dir, tar_path)
        return tar_path

    def upload_batch_packs(local_batch_pack_dir: Path, commit_message: str) -> None:
        # Upload all tar.gz in one commit (files-based dataset)
        _hf_upload_folder_batch(
            api=api,
            token=token,
            repo_id=args.hf_out_repo,
            local_folder=local_batch_pack_dir,
            path_in_repo="runs",
            commit_message=commit_message,
        )

    def process_batch(batch_ids: List[str]) -> None:
        nonlocal processed
        if not batch_ids:
            return

        # --- download batch ---
        repo_ids = {v: vid2repo[v] for v in batch_ids}
        _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "stage": "batch_start", "batch": list(batch_ids)})

        downloads: Dict[str, DownloadResult] = {}
        with ThreadPoolExecutor(max_workers=max(1, int(args.download_workers))) as ex:
            futs = {ex.submit(_download_one, v, repo_ids[v], cache_dir): v for v in batch_ids}
            for fut in as_completed(futs):
                res = fut.result()
                downloads[res.video_id] = res

        # prepare a temp folder for batch pack upload
        batch_pack_upload_dir = Path(tempfile.mkdtemp(prefix="batch_packs_"))

        # --- process sequentially (GPU) ---
        packed_ok_vids: List[str] = []
        for v in batch_ids:
            res = downloads.get(v)
            if not res or not res.ok or not res.path:
                _append_jsonl(
                    state_jsonl,
                    {"ts": _utc_iso_now(), "video_id": v, "stage": "download", "status": "error", "error": (res.error if res else "unknown")},
                )
                continue

            ok = False
            run_id_or_err: Optional[str] = None
            for attempt in range(int(args.retries)):
                _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "video_id": v, "stage": "process", "status": "start", "attempt": attempt + 1})
                ok, run_id_or_err = run_one(v, Path(res.path))
                if ok:
                    break
                time.sleep(1.0)

            if not ok or not run_id_or_err:
                excluded.add(v)
                _atomic_write_json(excluded_path, sorted(excluded))
                _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "video_id": v, "stage": "process", "status": "excluded", "error": run_id_or_err})
                continue

            run_id = run_id_or_err
            _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "video_id": v, "stage": "process", "status": "ok", "run_id": run_id})

            # pack (we upload batch as a folder afterwards)
            tar_path = pack_one(v, run_id)
            if tar_path is None:
                _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "video_id": v, "stage": "pack", "status": "error", "run_id": run_id})
            else:
                # Place into structured folder under batch dir: runs/<platform>/<video>/<run>.tar.gz
                target = batch_pack_upload_dir / args.platform_id / v
                target.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(tar_path), str(target / f"{run_id}.tar.gz"))
                _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "video_id": v, "stage": "pack", "status": "ok", "run_id": run_id})
                packed_ok_vids.append(v)

            # cleanup local video
            try:
                os.remove(res.path)
            except Exception:
                pass
            processed += 1

        # Upload this batch in one commit
        try:
            upload_batch_packs(batch_pack_upload_dir, commit_message=f"batch upload {args.platform_id} {int(time.time())}")
            _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "stage": "batch_upload", "status": "ok", "batch_size": len(batch_ids)})
            # Update remote cache in-place (so restarts skip already uploaded runs without re-listing HF)
            if args.skip_remote_existing and remote_cache_path is not None and packed_ok_vids:
                for v in packed_ok_vids:
                    done_by_remote.add(v)
                _write_remote_done_cache(remote_cache_path, out_repo_id=args.hf_out_repo, platform_id=args.platform_id, done_video_ids=done_by_remote)
                _append_jsonl(
                    state_jsonl,
                    {"ts": _utc_iso_now(), "stage": "remote_index", "status": "ok", "source": "append_after_upload", "count": len(done_by_remote)},
                )
        except Exception as e:
            _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "stage": "batch_upload", "status": "error", "error": str(e)})
        try:
            shutil.rmtree(batch_pack_upload_dir, ignore_errors=True)
        except Exception:
            pass

        # best-effort cleanup cache dir if it grows
        try:
            for f in cache_dir.glob("*.mp4"):
                # remove leftovers
                try:
                    f.unlink()
                except Exception:
                    pass
        except Exception:
            pass
        # cleanup yt tmp dir
        try:
            for f in yt_tmp_dir.glob("*"):
                if f.is_file():
                    try:
                        f.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

    for vid in _iter_video_ids_from_main(main_ready_dir):
        if args.max_videos and processed >= args.max_videos:
            break
        if vid in excluded:
            continue
        if vid in done_by_state:
            _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "video_id": vid, "stage": "skip", "status": "ok", "reason": "resume_state"})
            continue
        if done_by_remote and vid in done_by_remote:
            _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "video_id": vid, "stage": "skip", "status": "ok", "reason": "already_done_remote"})
            continue
        if vid not in vid2repo:
            # Fallback: download from YouTube and upload into videos11, then proceed.
            _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "video_id": vid, "stage": "fallback_download", "status": "start"})
            fb_ok = False
            fb_path: Optional[str] = None
            fb_err: Optional[str] = None
            for attempt in range(int(args.retries)):
                res = _yt_download_video(vid, yt_tmp_dir, max_duration_sec=int(args.max_duration_sec), cookies_path=args.yt_cookies)
                if res.ok and res.path:
                    fb_ok = True
                    fb_path = res.path
                    break
                fb_err = res.error
                time.sleep(1.0)
            if not fb_ok or not fb_path:
                excluded.add(vid)
                _atomic_write_json(excluded_path, sorted(excluded))
                _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "video_id": vid, "stage": "fallback_download", "status": "excluded", "error": fb_err})
                continue
            # Upload to videos11
            try:
                _hf_upload_video_to_dataset(api, token, args.hf_videos11_repo, vid, fb_path)
                vid2repo[vid] = args.hf_videos11_repo
                _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "video_id": vid, "stage": "fallback_upload", "status": "ok", "repo": args.hf_videos11_repo})
            except Exception as e:
                excluded.add(vid)
                _atomic_write_json(excluded_path, sorted(excluded))
                _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "video_id": vid, "stage": "fallback_upload", "status": "excluded", "error": str(e)})
                continue
        if already_done(vid):
            _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "video_id": vid, "stage": "skip", "status": "ok", "reason": "already_done_local"})
            continue

        batch.append(vid)
        if len(batch) < args.batch_size:
            continue

        process_batch(batch)
        batch = []

    # process remaining partial batch (optional)
    if batch:
        _append_jsonl(state_jsonl, {"ts": _utc_iso_now(), "stage": "batch_partial_left", "batch": list(batch)})
        process_batch(batch)

    _atomic_write_json(excluded_path, sorted(excluded))
    print(f"[ok] processed={processed} excluded={len(excluded)} state={state_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


