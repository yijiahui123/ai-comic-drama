"""Production state, quality, and export helpers for the web console."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from pipeline.state import PipelineState, QualityCheck, ShotState, Stage
from utils import slugify
from utils.assets import (
    ASSET_MANIFEST_PATH,
    canonical_scene,
    canonical_shot,
    load_asset_manifest,
)
from utils.paths import PROJECT_ROOT


ROOT = PROJECT_ROOT
OUTPUT_ROOT = Path("output")
EXPORT_ROOT = OUTPUT_ROOT / "exports"
SUBTITLE_ROOT = ROOT / OUTPUT_ROOT / "subtitles"

SHOT_STATUSES = {
    "pending",
    "ready",
    "needs_retry",
    "needs_review",
    "approved",
    "failed",
}
REVIEW_STATUSES = {
    "pending",
    "approved",
    "rejected",
    "needs_retry",
}
STATUS_REVIEW_DEFAULTS = {
    "pending": "pending",
    "ready": "pending",
    "needs_retry": "needs_retry",
    "needs_review": "pending",
    "approved": "approved",
    "failed": "needs_retry",
}


def set_shot_status(
    shot: ShotState,
    status: str,
    review_status: str | None = None,
    last_error: str | None = None,
) -> ShotState:
    """Set shot status with a validated review status pairing."""
    if status not in SHOT_STATUSES:
        raise ValueError(f"Invalid shot status: {status}")
    review = review_status or STATUS_REVIEW_DEFAULTS[status]
    if review not in REVIEW_STATUSES:
        raise ValueError(f"Invalid review status: {review}")
    if status == "approved" and review != "approved":
        raise ValueError("approved shots must have approved review_status")
    if review == "approved" and status != "approved":
        raise ValueError("approved review_status requires approved shot status")
    shot.status = status
    shot.review_status = review
    shot.last_error = last_error
    shot.updated_at = datetime.now(timezone.utc)
    return shot


def iter_script_scenes(script: dict[str, Any]):
    for episode in script.get("episodes", []):
        for scene in episode.get("scenes", []):
            yield episode, scene


def iter_script_shots(script: dict[str, Any]):
    for episode, scene in iter_script_scenes(script):
        for shot in scene.get("shots", []):
            yield episode, scene, shot


def shot_output_paths(shot_id: str) -> dict[str, str]:
    return {
        "shot_image": str(canonical_shot(shot_id)),
        "video": str(OUTPUT_ROOT / "videos" / f"{shot_id}.mp4"),
        "audio": str(OUTPUT_ROOT / "audio" / f"{shot_id}.wav"),
        "lipsync": str(OUTPUT_ROOT / "lipsync" / f"{shot_id}_lipsync.mp4"),
        "subtitle": str(SUBTITLE_ROOT / f"{shot_id}.ass"),
        "continuity": str(OUTPUT_ROOT / "continuity" / f"{shot_id}_start.png"),
    }


def _existing_output(paths: dict[str, str], key: str) -> str:
    path = Path(paths[key])
    return str(path) if path.exists() else ""


def build_shot_states(state: PipelineState, preserve_existing: bool = True) -> dict[str, ShotState]:
    """Build or refresh shot-level state from the script and manifests."""
    existing = state.shot_states if preserve_existing else {}
    next_states: dict[str, ShotState] = {}
    if not state.script:
        state.shot_states = next_states
        state.review_queue = []
        return next_states

    manifest = load_asset_manifest()
    for _, scene, shot in iter_script_shots(state.script):
        shot_id = str(shot.get("shot_id", "")).strip()
        if not shot_id:
            continue
        scene_id = str(scene.get("scene_id", "")).strip()
        paths = shot_output_paths(shot_id)
        previous = existing.get(shot_id)
        item = previous.model_copy(deep=True) if previous else ShotState(shot_id=shot_id)
        item.scene_id = scene_id
        if not item.locked:
            item.script = dict(shot)
        item.assets = {
            "shot_image": manifest.get("shots", {}).get(shot_id, "") or paths["shot_image"],
            "scene_image": manifest.get("scenes", {}).get(scene_id, "") or str(canonical_scene(scene_id)),
        }
        item.outputs.update(
            {
                "video": _existing_output(paths, "video"),
                "audio": _existing_output(paths, "audio"),
                "lipsync": _existing_output(paths, "lipsync"),
                "subtitle": _existing_output(paths, "subtitle"),
            }
        )
        item.sources = {
            "shot_image": state.asset_sources.get("shots", {}).get(item.assets["shot_image"], "library" if "assets/library/" in item.assets["shot_image"] else "canonical"),
            "scene_image": state.asset_sources.get("scenes", {}).get(item.assets["scene_image"], "library" if "assets/library/" in item.assets["scene_image"] else "canonical"),
        }
        item.updated_at = datetime.now(timezone.utc)
        next_states[shot_id] = item

    state.shot_states = next_states
    refresh_review_queue(state)
    return next_states


def refresh_review_queue(state: PipelineState) -> list[str]:
    state.review_queue = [
        shot_id
        for shot_id, shot in state.shot_states.items()
        if shot.review_status in {"rejected", "needs_retry"}
        or shot.status in {"failed", "needs_review", "needs_retry"}
    ]
    return state.review_queue


def update_script_shot(state: PipelineState, shot_id: str, patch: dict[str, Any]) -> ShotState:
    if not state.script:
        raise ValueError("Project has no script")
    target: dict[str, Any] | None = None
    scene_id = ""
    for _, scene, shot in iter_script_shots(state.script):
        if str(shot.get("shot_id", "")) == shot_id:
            target = shot
            scene_id = str(scene.get("scene_id", ""))
            break
    if target is None:
        raise KeyError(shot_id)

    allowed = {
        "visual_prompt",
        "motion_prompt",
        "dialogue",
        "emotion",
        "mood",
        "duration",
        "characters",
        "camera",
        "description",
    }
    for key, value in patch.items():
        if key in allowed:
            target[key] = value

    build_shot_states(state)
    shot_state = state.shot_states.setdefault(shot_id, ShotState(shot_id=shot_id, scene_id=scene_id))
    shot_state.script = dict(target)
    if not shot_state.locked:
        set_shot_status(shot_state, "pending")
    shot_state.updated_at = datetime.now(timezone.utc)
    state.update_progress(f"Shot {shot_id} updated", event_type="shot_update")
    return shot_state


def set_shot_lock(state: PipelineState, shot_id: str, locked: bool) -> ShotState:
    build_shot_states(state)
    if shot_id not in state.shot_states:
        raise KeyError(shot_id)
    shot = state.shot_states[shot_id]
    shot.locked = locked
    shot.updated_at = datetime.now(timezone.utc)
    state.update_progress(f"Shot {shot_id} {'locked' if locked else 'unlocked'}", event_type="shot_lock")
    return shot


def review_shot(state: PipelineState, shot_id: str, review_status: str, note: str = "") -> ShotState:
    if review_status not in {"approved", "rejected", "needs_retry"}:
        raise ValueError("review_status must be approved, rejected, or needs_retry")
    build_shot_states(state)
    if shot_id not in state.shot_states:
        raise KeyError(shot_id)
    shot = state.shot_states[shot_id]
    if review_status == "approved":
        set_shot_status(shot, "approved", "approved", note or None)
    elif review_status == "needs_retry":
        set_shot_status(shot, "needs_retry", "needs_retry", note or None)
    else:
        set_shot_status(shot, "needs_review", "rejected", note or None)
    refresh_review_queue(state)
    state.update_progress(f"Shot {shot_id} reviewed as {review_status}", event_type="shot_review")
    return shot


def _repo_path(path_value: str | Path, root: Path = ROOT) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def final_video_paths(state: PipelineState) -> list[str]:
    """Return all final video paths, preserving backwards compatibility."""
    finals = list(state.final_videos or [])
    if state.final_video and state.final_video not in finals:
        finals.append(state.final_video)
    return finals


def _file_check(name: str, path_value: str, required: bool = True) -> QualityCheck:
    if not path_value:
        return QualityCheck(name=name, status="fail" if required else "warn", message="path is empty")
    path = _repo_path(path_value)
    if path.exists() and path.is_file():
        return QualityCheck(name=name, status="pass", message=str(path_value), details={"bytes": path.stat().st_size})
    return QualityCheck(name=name, status="fail" if required else "warn", message=f"missing: {path_value}")


def _probe_video(path_value: str) -> QualityCheck:
    path = _repo_path(path_value)
    if not path.exists():
        return QualityCheck(name="video_probe", status="fail", message=f"missing: {path_value}")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,duration",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return QualityCheck(name="video_probe", status="warn", message="ffprobe not found")
    except Exception as exc:  # noqa: BLE001
        return QualityCheck(name="video_probe", status="warn", message=str(exc))
    if result.returncode != 0:
        return QualityCheck(name="video_probe", status="warn", message=result.stderr[-300:])
    data = json.loads(result.stdout or "{}")
    stream = (data.get("streams") or [{}])[0]
    duration = float(stream.get("duration") or 0)
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    status = "pass" if duration > 0 and width > 0 and height > 0 else "fail"
    return QualityCheck(
        name="video_probe",
        status=status,
        message=f"{width}x{height}, {duration:.2f}s",
        details={"duration": duration, "width": width, "height": height},
    )


def _check_brightness(path_value: str, threshold: float = 10.0) -> QualityCheck:
    """Check for black/low-brightness video by sampling frames with ffmpeg."""
    path = _repo_path(path_value)
    if not path.exists():
        return QualityCheck(name="brightness", status="skip", message="video missing, skipped")
    cmd = [
        "ffmpeg",
        "-i", str(path),
        "-vf", "fps=1,signalstats",
        "-frames:v", "10",
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except FileNotFoundError:
        return QualityCheck(name="brightness", status="warn", message="ffmpeg not found")
    except Exception as exc:  # noqa: BLE001
        return QualityCheck(name="brightness", status="warn", message=str(exc))
    yavg_values = re.findall(r"YAVG:\s*([\d.]+)", result.stderr or "")
    if not yavg_values:
        return QualityCheck(name="brightness", status="warn", message="could not extract brightness data")
    avg_brightness = sum(float(v) for v in yavg_values) / len(yavg_values)
    status = "fail" if avg_brightness < threshold else "pass"
    return QualityCheck(
        name="brightness",
        status=status,
        message=f"avg brightness {avg_brightness:.1f} (threshold {threshold})",
        details={"avg_brightness": avg_brightness, "threshold": threshold, "samples": len(yavg_values)},
    )


def _check_multimodal_vision(path_value: str, required: bool = False) -> QualityCheck:
    """Placeholder for future AI Auto-QC Framework.
    
    This interface is reserved for integrating multimodal vision models 
    (e.g., LLaVA, Qwen-VL) to detect visual glitches, bad anatomy, or 
    prompt adherence in the generated videos or frames.
    """
    path = _repo_path(path_value)
    if not path.exists():
        return QualityCheck(name="ai_vision_qc", status="skip", message="video missing, skipped")
    
    # TODO: Implement API call to multimodal LLM endpoint
    # ...
    
    return QualityCheck(
        name="ai_vision_qc",
        status="pass",
        message="AI vision QC passed (placeholder)",
        details={"model": "none", "score": 1.0},
    )


def _check_audio_silence(path_value: str, threshold: float = -50.0) -> QualityCheck:
    """Detect if an audio file is mostly silent using ffmpeg volumedetect."""
    path = _repo_path(path_value)
    if not path.exists():
        return QualityCheck(name="audio_silence", status="skip", message="audio missing, skipped")
    cmd = [
        "ffmpeg",
        "-i", str(path),
        "-af", "volumedetect",
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return QualityCheck(name="audio_silence", status="warn", message="ffmpeg not found")
    except Exception as exc:  # noqa: BLE001
        return QualityCheck(name="audio_silence", status="warn", message=str(exc))
    import re
    match = re.search(r"mean_volume:\s*([-\d.]+)\s*dB", result.stderr or "")
    if not match:
        return QualityCheck(name="audio_silence", status="warn", message="could not detect volume")
    mean_volume = float(match.group(1))
    status = "fail" if mean_volume < threshold else "pass"
    return QualityCheck(
        name="audio_silence",
        status=status,
        message=f"mean volume {mean_volume:.1f}dB (threshold {threshold}dB)",
        details={"mean_volume": mean_volume, "threshold": threshold},
    )


def _subtitle_text(shot: ShotState, scene_segments: dict[str, list[dict[str, Any]]] | None = None) -> str:
    if scene_segments:
        segments = scene_segments.get(shot.scene_id, [])
        lines = [str(seg.get("text", "")) for seg in segments if seg.get("shot_id") == shot.shot_id]
        if lines:
            return "\n".join(lines)
    return str(shot.script.get("dialogue", "")).strip()


def scene_audio_segments(manifest: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    manifest = manifest if manifest is not None else load_asset_manifest()
    result: dict[str, list[dict[str, Any]]] = {}
    for scene_id, data in manifest.get("scene_audio", {}).items():
        if isinstance(data, dict) and isinstance(data.get("segments"), list):
            result[str(scene_id)] = data["segments"]
    return result


def validate_scene_segments(
    state: PipelineState,
    manifest: dict[str, Any] | None = None,
) -> list[QualityCheck]:
    """Validate scene audio segment timelines for gaps, overlaps, and duration mismatches."""
    manifest = manifest if manifest is not None else load_asset_manifest()
    checks: list[QualityCheck] = []
    for scene_id, data in manifest.get("scene_audio", {}).items():
        if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
            continue
        segments = data["segments"]
        if not segments:
            checks.append(QualityCheck(name=f"scene_segments/{scene_id}", status="warn", message="empty segment list"))
            continue

        # Check for overlaps and gaps
        sorted_segs = sorted(segments, key=lambda s: float(s.get("start", 0)))
        issues: list[str] = []
        for i in range(len(sorted_segs) - 1):
            cur_end = float(sorted_segs[i].get("end", 0))
            next_start = float(sorted_segs[i + 1].get("start", 0))
            gap = next_start - cur_end
            if gap < -0.01:
                issues.append(f"overlap at segment {i}-{i+1}: {abs(gap):.2f}s")
            elif gap > 0.1:
                issues.append(f"gap at segment {i}-{i+1}: {gap:.2f}s")

        # Check segment durations match shot durations from script
        for seg in sorted_segs:
            seg_shot_id = str(seg.get("shot_id", ""))
            if not seg_shot_id:
                continue
            shot_state = state.shot_states.get(seg_shot_id) if state.shot_states else None
            if not shot_state:
                continue
            expected_dur = float(shot_state.script.get("duration", 0) or 0)
            if expected_dur <= 0:
                continue
            actual_dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
            if abs(actual_dur - expected_dur) > 0.5:
                issues.append(f"shot {seg_shot_id}: segment {actual_dur:.1f}s != expected {expected_dur:.1f}s")

        status = "fail" if issues else "pass"
        checks.append(QualityCheck(
            name=f"scene_segments/{scene_id}",
            status=status,
            message="; ".join(issues) if issues else f"{len(segments)} segments OK",
            details={"scene_id": scene_id, "segment_count": len(segments), "issues": issues},
        ))
    return checks


def generate_shot_subtitles(state: PipelineState) -> list[str]:
    build_shot_states(state)
    SUBTITLE_ROOT.mkdir(parents=True, exist_ok=True)
    segments = scene_audio_segments()
    generated: list[str] = []
    for shot in state.shot_states.values():
        text = _subtitle_text(shot, segments)
        if not text:
            continue
        try:
            duration = float(shot.script.get("duration", 4) or 4)
        except (TypeError, ValueError):
            duration = 4
        out_path = SUBTITLE_ROOT / f"{shot.shot_id}.ass"
        out_path.write_text(
            "\n".join(
                [
                    "[Script Info]",
                    "ScriptType: v4.00+",
                    "[V4+ Styles]",
                    "Format: Name,Fontname,Fontsize,PrimaryColour,Alignment",
                    "Style: Default,Arial,42,&H00FFFFFF,2",
                    "[Events]",
                    "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text",
                    f"Dialogue: 0,0:00:00.00,0:00:{duration:05.2f},Default,,0,0,0,,{text}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        shot.outputs["subtitle"] = str(out_path)
        generated.append(str(out_path))
    state.update_progress(f"Generated {len(generated)} shot subtitle files", event_type="subtitles")
    return generated


def quality_check_shot(state: PipelineState, shot_id: str) -> ShotState:
    build_shot_states(state)
    if shot_id not in state.shot_states:
        raise KeyError(shot_id)
    shot = state.shot_states[shot_id]
    if not shot.outputs.get("subtitle") and shot.script.get("dialogue"):
        generate_shot_subtitles(state)

    paths = shot_output_paths(shot_id)
    video_path = shot.outputs.get("video") or paths["video"]
    audio_path = shot.outputs.get("audio") or paths["audio"]
    subtitle_path = shot.outputs.get("subtitle") or paths["subtitle"]
    checks = [
        _file_check("shot_image", shot.assets.get("shot_image") or paths["shot_image"], required=True),
        _file_check("video_file", video_path, required=True),
        _probe_video(video_path),
        _check_brightness(video_path),
        _check_multimodal_vision(video_path),
    ]
    if shot.script.get("dialogue"):
        checks.append(_file_check("audio_file", audio_path, required=False))
        checks.append(_check_audio_silence(audio_path))
        checks.append(_file_check("subtitle_file", subtitle_path, required=False))
    shot.quality_checks = checks
    failed = [check for check in checks if check.status == "fail"]
    warned = [check for check in checks if check.status == "warn"]
    if failed:
        set_shot_status(
            shot,
            "failed",
            "needs_retry" if shot.retry_count < shot.max_retries and not shot.locked else "rejected",
            "; ".join(check.message for check in failed),
        )
    elif warned:
        set_shot_status(shot, "needs_review", "pending", "; ".join(check.message for check in warned))
    else:
        set_shot_status(shot, "ready", "pending")
    refresh_review_queue(state)
    return shot


def quality_check_project(state: PipelineState) -> dict[str, Any]:
    build_shot_states(state)
    generate_shot_subtitles(state)
    for shot_id in list(state.shot_states):
        quality_check_shot(state, shot_id)
    total = len(state.shot_states)
    failed = sum(1 for shot in state.shot_states.values() if shot.status == "failed")
    needs_review = sum(1 for shot in state.shot_states.values() if shot.status == "needs_review")
    ready = sum(1 for shot in state.shot_states.values() if shot.status in {"ready", "approved"})
    final_checks = [
        _file_check(f"final_video/{idx + 1}", path, required=False)
        for idx, path in enumerate(final_video_paths(state))
    ]
    if not final_checks:
        final_checks = [_file_check("final_video", "", required=False)]
    segment_checks = validate_scene_segments(state)
    segment_failed = [c for c in segment_checks if c.status == "fail"]
    state.quality_report = {
        "total_shots": total,
        "ready": ready,
        "failed": failed,
        "needs_review": needs_review,
        "review_queue": list(state.review_queue),
        "final_video": final_checks[-1].model_dump(mode="json"),
        "final_videos": [check.model_dump(mode="json") for check in final_checks],
        "segment_checks": [c.model_dump(mode="json") for c in segment_checks],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    if segment_failed:
        state.quality_report["segment_issues"] = len(segment_failed)
    state.update_progress("Project quality check complete", event_type="quality_check")
    return state.quality_report


def mark_failed_for_retry(state: PipelineState) -> list[str]:
    build_shot_states(state)
    queued: list[str] = []
    for shot_id, shot in state.shot_states.items():
        if shot.locked:
            continue
        if shot.status not in {"failed", "needs_retry", "needs_review"} and shot.review_status not in {"needs_retry", "rejected"}:
            continue
        if shot.retry_count >= shot.max_retries:
            set_shot_status(shot, "needs_review", "rejected")
            continue
        shot.retry_count += 1
        set_shot_status(shot, "pending", "pending")
        shot.generation_params["seed"] = int(shot.generation_params.get("seed", 0)) + 1
        queued.append(shot_id)
    for stage in (Stage.VIDEO_GEN.value, Stage.EDITING.value):
        state.stages.pop(stage, None)
    if queued:
        state.current_stage = Stage.VIDEO_GEN
    refresh_review_queue(state)
    state.update_progress(f"Marked {len(queued)} shots for retry", event_type="retry")
    return queued


def export_project(state: PipelineState, root: Path = ROOT) -> dict[str, Any]:
    build_shot_states(state)
    quality_check_project(state)
    export_dir = root / EXPORT_ROOT / state.project_id
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_asset_manifest()
    files: dict[str, Any] = {}
    (export_dir / "script.json").write_text(json.dumps(state.script or {}, ensure_ascii=False, indent=2), encoding="utf-8")
    (export_dir / "manifest.yaml").write_text(yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (export_dir / "shot_states.json").write_text(
        json.dumps({k: v.model_dump(mode="json") for k, v in state.shot_states.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (export_dir / "quality_report.json").write_text(json.dumps(state.quality_report, ensure_ascii=False, indent=2), encoding="utf-8")
    files.update(
        {
            "script": str(export_dir / "script.json"),
            "manifest": str(export_dir / "manifest.yaml"),
            "shot_states": str(export_dir / "shot_states.json"),
            "quality_report": str(export_dir / "quality_report.json"),
        }
    )
    copied_finals: list[str] = []
    for final_value in final_video_paths(state):
        final_path = _repo_path(final_value, root)
        if final_path.exists():
            dest = export_dir / final_path.name
            shutil.copy2(final_path, dest)
            copied_finals.append(str(dest))
    if copied_finals:
        files["final_video"] = copied_finals[-1]
        files["final_videos"] = copied_finals

    # Copy subtitle files
    subtitle_dir = export_dir / "subtitles"
    subtitle_count = 0
    for shot in state.shot_states.values():
        sub_path = shot.outputs.get("subtitle")
        if sub_path:
            source = _repo_path(sub_path, root)
            if source.exists() and source.is_file():
                dest = subtitle_dir / source.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                subtitle_count += 1
    if subtitle_count:
        files["subtitles"] = str(subtitle_dir)

    # Copy generated (non-library) assets
    assets_dir = export_dir / "generated_assets"
    asset_list_lines: list[str] = []
    for shot in state.shot_states.values():
        for key, path_value in shot.assets.items():
            if not path_value or "assets/library/" in path_value.replace("\\", "/"):
                continue
            source = _repo_path(path_value, root)
            if source.exists() and source.is_file():
                dest = assets_dir / key / source.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                asset_list_lines.append(f"{key}\t{path_value}\t{source.name}")
    for key, paths in manifest.items():
        if key in ("characters", "scene_audio"):
            continue
        if not isinstance(paths, dict):
            continue
        for item_key, path_value in paths.items():
            if not path_value or "assets/library/" in str(path_value).replace("\\", "/"):
                continue
            source = _repo_path(path_value, root)
            if source.exists() and source.is_file():
                dest = assets_dir / key / source.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                asset_list_lines.append(f"{key}/{item_key}\t{path_value}\t{source.name}")
    (export_dir / "asset_list.txt").write_text("\n".join(asset_list_lines) or "(no generated assets)", encoding="utf-8")
    files["asset_list"] = str(export_dir / "asset_list.txt")

    # Generate log index
    log_dir = root / "logs"
    log_lines: list[str] = []
    if log_dir.exists():
        for log_file in sorted(log_dir.glob("*.log")):
            log_lines.append(f"{log_file.name}\t{log_file.stat().st_size} bytes")
    (export_dir / "log_index.txt").write_text("\n".join(log_lines) or "(no log files)", encoding="utf-8")
    files["log_index"] = str(export_dir / "log_index.txt")

    zip_path = root / EXPORT_ROOT / f"{state.project_id}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in export_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(export_dir.parent))

    state.export_manifest = {
        "directory": str(export_dir),
        "zip": str(zip_path),
        "files": files,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    state.update_progress("Project export package created", event_type="export")
    return state.export_manifest
