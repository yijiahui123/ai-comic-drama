"""FastAPI web console for the AI Comic Drama pipeline."""

from __future__ import annotations

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import asyncio
import json
import os
import shutil
import uuid
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.state import PipelineState, Stage
from utils import slugify
from utils.assets import (
    ASSET_CONFIG_PATH,
    ASSET_MANIFEST_PATH,
    EXPRESSION_VARIANTS,
    canonical_shot,
    load_asset_config,
    load_asset_manifest,
    save_asset_config,
    save_asset_manifest,
    validate_asset_setup,
)
from utils.logger import get_logger
from utils.production import (
    build_shot_states,
    export_project,
    mark_failed_for_retry,
    quality_check_project,
    quality_check_shot,
    review_shot,
    set_shot_lock,
    update_script_shot,
)

from utils.paths import PROJECT_ROOT

logger = get_logger("web_server")

ROOT = PROJECT_ROOT
WEB_DIR = ROOT / "web"
STATE_DIR = ROOT / "output" / "state"
QUEUE_STATE_PATH = ROOT / "output" / "queue_state.json"

app = FastAPI(title="AI Comic Drama Console")
_tasks: dict[str, asyncio.Task] = {}
_queue_records: dict[str, dict[str, Any]] = {}
_queue_order: list[str] = []
_queue_runner: asyncio.Task | None = None
_queued_work: asyncio.Queue[str] | None = None
_active_queue_task_id: str | None = None

ENABLE_MODEL_TASKS = os.getenv("AI_COMIC_ENABLE_MODEL_TASKS", "0") == "1"

_RUNNER_MAP = {
    "create_project": "_model_task_runner",
    "resume_project": "_model_task_runner",
    "rerun_stage": "_model_task_runner",
    "retry_failed_shots": "_model_task_runner",
    "export_project": "_export_task_runner",
}

_RUNNER_FUNCS: dict[str, Any] = {}


def _save_queue() -> None:
    QUEUE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = {
        tid: {k: v for k, v in rec.items() if k != "runner"}
        for tid, rec in _queue_records.items()
    }
    data = {"records": records, "order": _queue_order}
    QUEUE_STATE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_queue() -> None:
    global _queue_records, _queue_order
    if not QUEUE_STATE_PATH.exists():
        return
    try:
        data = json.loads(QUEUE_STATE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    records = data.get("records", {})
    order = data.get("order", [])
    runner_name_map = _RUNNER_MAP
    for tid, rec in records.items():
        kind = rec.get("kind", "")
        runner_name = runner_name_map.get(kind, "")
        rec["runner"] = _RUNNER_FUNCS.get(runner_name) or _model_task_runner
        # Reset queued/running tasks to queued so they can be re-processed
        if rec.get("status") in ("queued", "running"):
            rec["status"] = "queued"
            rec["message"] = "Restored after restart"
    _queue_records = records
    _queue_order = [tid for tid in order if tid in records]


class CreateProjectRequest(BaseModel):
    prompt: str = Field(min_length=1)
    profile: str = "default"


class BatchRequest(BaseModel):
    prompts: list[str] = Field(min_length=1)
    profile: str = "default"


class RerunRequest(BaseModel):
    stage: str
    force: bool = False
    start: bool = True


class CleanupRequest(BaseModel):
    stage: str
    delete_files: bool = True


class ShotCleanupRequest(BaseModel):
    include_asset: bool = True
    delete_files: bool = True


class ShotRerunRequest(BaseModel):
    include_asset: bool = True
    delete_files: bool = True
    start: bool = False


class AssetTextUpdate(BaseModel):
    content: str


class AssetMappingUpdate(BaseModel):
    data: dict[str, Any]


class CharacterLoraUpdate(BaseModel):
    character: str = Field(min_length=1)
    enabled: bool = False
    name: str = ""
    trigger: str = ""
    strength_model: float = 0.85
    strength_clip: float = 0.8


class SceneAudioUpdate(BaseModel):
    scene_id: str = Field(min_length=1)
    audio_path: str = ""
    segments: list[dict[str, Any]] = Field(default_factory=list)


class ShotUpdateRequest(BaseModel):
    data: dict[str, Any]


class ReviewRequest(BaseModel):
    status: str
    note: str = ""


def _state_to_dict(state: PipelineState) -> dict[str, Any]:
    return json.loads(state.model_dump_json())


def _load_state(project_id: str) -> PipelineState:
    try:
        return PipelineState.load(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}") from exc


def _save_state(state: PipelineState) -> None:
    build_shot_states(state)
    state.save()


def _task_record(task_id: str, status: str, message: str = "", **updates: Any) -> dict[str, Any]:
    record = _queue_records[task_id]
    record.update(
        {
            "status": status,
            "message": message,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **updates,
        }
    )
    _save_queue()
    return record


def _ensure_queue() -> asyncio.Queue[str]:
    global _queued_work
    if _queued_work is None:
        _queued_work = asyncio.Queue()
    return _queued_work


async def _queue_worker() -> None:
    global _active_queue_task_id
    queue = _ensure_queue()
    while True:
        task_id = await queue.get()
        _active_queue_task_id = task_id
        record = _queue_records.get(task_id)
        if not record or record.get("status") == "canceled":
            _active_queue_task_id = None
            queue.task_done()
            continue
        _task_record(task_id, "running", "Task started")
        project_id = record.get("project_id")
        try:
            state = _load_state(project_id) if project_id else None
            if state:
                state.queue_status = "running"
                state.update_progress(f"Queue task {record['kind']} started", event_type="queue")
                state.save()

            runner = record.get("runner")
            result = await runner(record) if runner else None

            if state:
                state = _load_state(project_id)
                state.queue_status = "completed"
                state.update_progress(f"Queue task {record['kind']} completed", event_type="queue")
                state.save()
            _task_record(task_id, "completed", "Task completed", result=result)
        except asyncio.CancelledError:
            if project_id:
                try:
                    state = _load_state(project_id)
                    state.queue_status = "canceled"
                    state.update_progress(f"Queue task {record['kind']} canceled", event_type="queue")
                    state.save()
                except HTTPException:
                    pass
            _task_record(task_id, "canceled", "Task canceled")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Queue task %s failed: %s", task_id, exc)
            if project_id:
                try:
                    state = _load_state(project_id)
                    state.queue_status = "failed"
                    state.update_progress(f"Queue task {record['kind']} failed: {exc}", event_type="queue")
                    state.save()
                except HTTPException:
                    pass
            _task_record(task_id, "failed", str(exc), error=str(exc))
        finally:
            _active_queue_task_id = None
            queue.task_done()


def _enqueue(kind: str, project_id: str | None, payload: dict[str, Any], runner) -> dict[str, Any]:
    global _queue_runner
    task_id = uuid.uuid4().hex[:10]
    now = datetime.now(timezone.utc).isoformat()
    _queue_records[task_id] = {
        "task_id": task_id,
        "kind": kind,
        "project_id": project_id,
        "payload": payload,
        "status": "queued",
        "message": "Queued",
        "created_at": now,
        "updated_at": now,
        "runner": runner,
    }
    _queue_order.append(task_id)
    # Calculate queue position
    queued_before = sum(
        1 for tid in _queue_order
        if _queue_records.get(tid, {}).get("status") == "queued"
        and tid != task_id
    )
    _queue_records[task_id]["queue_position"] = queued_before + 1
    if project_id:
        state = _load_state(project_id)
        state.queue_status = "queued"
        state.update_progress(f"Queue task {kind} queued (position {queued_before + 1})", event_type="queue")
        state.save()
    _ensure_queue().put_nowait(task_id)
    if _queue_runner is None or _queue_runner.done():
        _queue_runner = asyncio.create_task(_queue_worker())
    _save_queue()
    return {k: v for k, v in _queue_records[task_id].items() if k != "runner"}


async def _model_task_runner(record: dict[str, Any]) -> dict[str, Any]:
    project_id = str(record["project_id"])
    if not ENABLE_MODEL_TASKS:
        state = _load_state(project_id)
        state.update_progress(
            "Model execution is disabled; task recorded only. Set AI_COMIC_ENABLE_MODEL_TASKS=1 to run real services.",
            event_type="queue_skip",
        )
        state.save()
        return {"model_execution": "disabled"}
    orchestrator = PipelineOrchestrator.resume(project_id)
    await orchestrator.run()
    return {"model_execution": "completed"}


async def _export_task_runner(record: dict[str, Any]) -> dict[str, Any]:
    project_id = str(record["project_id"])
    state = _load_state(project_id)
    manifest = export_project(state, ROOT)
    state.save()
    return manifest


def _public_task(record: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in record.items() if k != "runner"}


def _load_services_for_health() -> dict[str, Any]:
    path = ROOT / "configs" / "services.yaml"
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}


def _command_available(command: str) -> dict[str, Any]:
    path = shutil.which(command)
    if not path:
        return {"ok": False, "path": "", "message": f"{command} not found"}
    return {"ok": True, "path": path, "message": "available"}


def _local_http_check(url: str) -> dict[str, Any]:
    if not url:
        return {"ok": False, "url": "", "message": "not configured"}
    try:
        with urllib.request.urlopen(url, timeout=1) as response:  # noqa: S310 - local operator URL check
            return {"ok": 200 <= response.status < 500, "url": url, "status": response.status}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "url": url, "message": str(exc)}


def _repo_file(path_value: str | Path | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    try:
        resolved = path.resolve()
        resolved.relative_to(ROOT.resolve())
    except (OSError, ValueError):
        return None
    return resolved


def _delete_file(path_value: str | Path | None) -> dict[str, Any]:
    path = _repo_file(path_value)
    if path is None:
        return {"path": str(path_value or ""), "deleted": False, "reason": "outside_repo_or_empty"}
    try:
        display_path = str(path.relative_to(ROOT.resolve()))
    except ValueError:
        display_path = str(path)
    if not path.exists():
        return {"path": display_path, "deleted": False, "reason": "missing"}
    if not path.is_file():
        return {"path": display_path, "deleted": False, "reason": "not_file"}
    path.unlink()
    return {"path": display_path, "deleted": True}


def _shot_output_paths(shot_id: str) -> list[Path]:
    return [
        ROOT / "output" / "videos" / f"{shot_id}.mp4",
        ROOT / "output" / "audio" / f"{shot_id}.wav",
        ROOT / "output" / "lipsync" / f"{shot_id}_lipsync.mp4",
        ROOT / "output" / "continuity" / f"{shot_id}_start.png",
    ]


def _script_shot_ids(state: PipelineState) -> list[str]:
    ids: list[str] = []
    if not state.script:
        return ids
    for episode in state.script.get("episodes", []):
        for scene in episode.get("scenes", []):
            for shot in scene.get("shots", []):
                shot_id = str(shot.get("shot_id", "")).strip()
                if shot_id:
                    ids.append(shot_id)
    return ids


def _script_scene_ids(state: PipelineState) -> list[str]:
    ids: list[str] = []
    if not state.script:
        return ids
    for episode in state.script.get("episodes", []):
        for scene in episode.get("scenes", []):
            scene_id = str(scene.get("scene_id", "")).strip()
            if scene_id:
                ids.append(scene_id)
    return ids


def _source_for_path(state: PipelineState, path_value: str) -> str:
    for mapping in state.asset_sources.values():
        if mapping.get(path_value):
            return mapping[path_value]
    if "assets/library/" in path_value.replace("\\", "/"):
        return "library"
    return "unknown"


def _generated_asset_paths(state: PipelineState) -> list[str]:
    paths: list[str] = []
    for group_paths in state.asset_manifest.values():
        for path in group_paths:
            if _source_for_path(state, path) != "library":
                paths.append(path)
    return paths


def _stage_cleanup_paths(state: PipelineState, stage: Stage) -> list[str | Path]:
    paths: list[str | Path] = []
    if stage in (Stage.SCRIPTING, Stage.ASSET_GEN):
        paths.extend(_generated_asset_paths(state))
    if stage in (Stage.SCRIPTING, Stage.ASSET_GEN, Stage.VIDEO_GEN):
        for group in ("videos", "audio", "lipsync"):
            paths.extend(state.video_manifest.get(group, []))
        for shot_id in _script_shot_ids(state):
            paths.extend(_shot_output_paths(shot_id))
        for scene_id in _script_scene_ids(state):
            paths.append(ROOT / "output" / "audio" / f"{slugify(scene_id)}_scene.wav")
    if stage in (Stage.SCRIPTING, Stage.ASSET_GEN, Stage.VIDEO_GEN, Stage.EDITING):
        if state.final_video:
            paths.append(state.final_video)
        paths.extend((ROOT / "output" / "final").glob(f"{state.project_id}*.mp4"))
    return list(dict.fromkeys(paths))


def _cleanup_stage_files(state: PipelineState, stage: Stage) -> list[dict[str, Any]]:
    return [_delete_file(path) for path in _stage_cleanup_paths(state, stage)]


def _clear_stage_state(state: PipelineState, stage: Stage) -> None:
    reset = False
    for candidate in (Stage.SCRIPTING, Stage.ASSET_GEN, Stage.VIDEO_GEN, Stage.EDITING):
        if candidate == stage:
            reset = True
        if reset:
            state.stages.pop(candidate.value, None)
    if stage == Stage.SCRIPTING:
        state.script = None
        state.asset_manifest = {}
        state.asset_sources = {}
        state.video_manifest = {}
        state.final_video = None
        state.final_videos = []
    elif stage == Stage.ASSET_GEN:
        state.asset_manifest = {}
        state.asset_sources = {}
        state.video_manifest = {}
        state.final_video = None
        state.final_videos = []
    elif stage == Stage.VIDEO_GEN:
        state.video_manifest = {}
        state.final_video = None
        state.final_videos = []
    elif stage == Stage.EDITING:
        state.final_video = None
        state.final_videos = []
    state.current_stage = stage
    state.update_progress(f"Marked {stage.value} for regeneration", 0, 0, "cleanup")
    state.save()


def _cleanup_shot_state(state: PipelineState, shot_id: str, include_asset: bool) -> None:
    build_shot_states(state)
    if include_asset:
        state.asset_manifest["shots"] = [
            path for path in state.asset_manifest.get("shots", []) if Path(path).stem != slugify(shot_id)
        ]
        state.asset_sources.get("shots", {}).pop(str(canonical_shot(shot_id)), None)
        state.stages.pop(Stage.ASSET_GEN.value, None)
        state.current_stage = Stage.ASSET_GEN
    else:
        state.current_stage = Stage.VIDEO_GEN

    for key in ("videos", "audio", "lipsync"):
        suffix = "_lipsync" if key == "lipsync" else ""
        state.video_manifest[key] = [
            path for path in state.video_manifest.get(key, []) if Path(path).stem != f"{shot_id}{suffix}"
        ]
    state.stages.pop(Stage.VIDEO_GEN.value, None)
    state.stages.pop(Stage.EDITING.value, None)
    state.final_video = None
    if shot_id in state.shot_states:
        shot_state = state.shot_states[shot_id]
        shot_state.status = "pending"
        shot_state.review_status = "pending"
        shot_state.last_error = None
        for key in ("video", "audio", "lipsync"):
            shot_state.outputs[key] = ""
        if include_asset:
            shot_state.assets["shot_image"] = str(canonical_shot(shot_id, ROOT / "assets"))
    state.update_progress(f"Marked shot {shot_id} for regeneration", 0, 0, "shot_cleanup")
    state.save()


async def _run_project(orchestrator: PipelineOrchestrator) -> None:
    project_id = orchestrator.state.project_id
    try:
        await orchestrator.run()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Project %s failed: %s", project_id, exc)
        orchestrator.state.current_stage = Stage.ERROR
        orchestrator.state.update_progress(f"Unhandled error: {exc}", event_type="stage_error")
        orchestrator.state.save()
    finally:
        _tasks.pop(project_id, None)


def _start_background(orchestrator: PipelineOrchestrator) -> None:
    project_id = orchestrator.state.project_id
    existing = _tasks.get(project_id)
    if existing and not existing.done():
        return
    _tasks[project_id] = asyncio.create_task(_run_project(orchestrator))


@app.on_event("startup")
async def _startup() -> None:
    (ROOT / "output").mkdir(exist_ok=True)
    WEB_DIR.mkdir(exist_ok=True)
    _RUNNER_FUNCS["_model_task_runner"] = _model_task_runner
    _RUNNER_FUNCS["_export_task_runner"] = _export_task_runner
    _load_queue()
    # Restart queue worker if there are pending tasks
    if any(rec.get("status") == "queued" for rec in _queue_records.values()):
        for tid in _queue_order:
            if _queue_records.get(tid, {}).get("status") == "queued":
                _ensure_queue().put_nowait(tid)
        global _queue_runner
        _queue_runner = asyncio.create_task(_queue_worker())


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")

app.mount("/assets", StaticFiles(directory=ROOT / "assets"), name="assets")
app.mount("/output", StaticFiles(directory=ROOT / "output"), name="output")


@app.post("/api/projects")
async def create_project(req: CreateProjectRequest) -> dict[str, Any]:
    orchestrator = PipelineOrchestrator.new(req.prompt)
    orchestrator.state.update_progress(
        "Project created; model execution is disabled by default",
        0,
        0,
        "created",
    )
    orchestrator.state.save()
    task = _enqueue("create_project", orchestrator.state.project_id, {}, _model_task_runner)
    data = _state_to_dict(_load_state(orchestrator.state.project_id))
    data["queued_task"] = task
    return data


@app.post("/api/batch")
async def batch_create_projects(req: BatchRequest) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for prompt in req.prompts:
        prompt = prompt.strip()
        if not prompt:
            continue
        orchestrator = PipelineOrchestrator.new(prompt)
        orchestrator.state.update_progress(
            "Batch project created",
            0, 0, "batch_created",
        )
        orchestrator.state.save()
        task = _enqueue(
            "create_project",
            orchestrator.state.project_id,
            {},
            _model_task_runner,
        )
        results.append({
            "project_id": orchestrator.state.project_id,
            "prompt": prompt,
            "queued_task": task,
        })
    return {"ok": True, "count": len(results), "projects": results}


@app.get("/api/projects")
async def list_projects() -> list[dict[str, Any]]:
    projects: list[dict[str, Any]] = []
    if STATE_DIR.exists():
        for path in sorted(STATE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                state = PipelineState.model_validate(json.loads(path.read_text(encoding="utf-8")))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping invalid state file %s: %s", path, exc)
                continue
            build_shot_states(state)
            data = _state_to_dict(state)
            data["running"] = path.stem in _tasks and not _tasks[path.stem].done()
            projects.append(data)
    return projects


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str) -> dict[str, Any]:
    state = _load_state(project_id)
    build_shot_states(state)
    state.save()
    data = _state_to_dict(state)
    data["running"] = project_id in _tasks and not _tasks[project_id].done()
    return data


@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str) -> dict[str, Any]:
    existing = _tasks.get(project_id)
    if existing and not existing.done():
        raise HTTPException(status_code=409, detail="Project is running, cancel it first")
    state_path = STATE_DIR / f"{project_id}.json"
    if not state_path.exists():
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}")
    # Clean up output files
    deleted_files: list[str] = []
    try:
        state = _load_state(project_id)
        for shot_id in _script_shot_ids(state):
            for p in _shot_output_paths(shot_id):
                if p.exists():
                    p.unlink()
                    deleted_files.append(str(p))
        # Final videos
        for v in getattr(state, "final_videos", []) or []:
            vp = Path(v)
            if vp.exists():
                vp.unlink()
                deleted_files.append(str(vp))
        if state.final_video:
            vp = Path(state.final_video)
            if vp.exists() and str(vp) not in deleted_files:
                vp.unlink()
                deleted_files.append(str(vp))
    except Exception:
        logger.warning("Failed to clean output files for %s", project_id, exc_info=True)
    state_path.unlink()
    # Remove queue records for this project
    to_remove = [tid for tid, rec in _queue_records.items() if rec.get("project_id") == project_id]
    for tid in to_remove:
        _queue_records.pop(tid, None)
        if tid in _queue_order:
            _queue_order.remove(tid)
    _save_queue()
    return {"ok": True, "deleted": project_id, "files_removed": len(deleted_files)}


@app.post("/api/projects/{project_id}/resume")
async def resume_project(project_id: str) -> dict[str, Any]:
    _load_state(project_id)
    task = _enqueue("resume_project", project_id, {}, _model_task_runner)
    data = _state_to_dict(_load_state(project_id))
    data["queued_task"] = task
    return data


@app.post("/api/projects/{project_id}/rerun")
async def rerun_project(project_id: str, req: RerunRequest) -> dict[str, Any]:
    existing = _tasks.get(project_id)
    if existing and not existing.done():
        raise HTTPException(status_code=409, detail="Project is already running")
    try:
        stage = Stage(req.stage)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {req.stage}") from exc
    orchestrator = PipelineOrchestrator.resume(project_id)
    deleted: list[dict[str, Any]] = []
    if req.force:
        deleted = _cleanup_stage_files(orchestrator.state, stage)
    orchestrator.reset_from_stage(stage)
    build_shot_states(orchestrator.state)
    orchestrator.state.save()
    if req.start:
        task = _enqueue("rerun_stage", project_id, {"stage": stage.value, "force": req.force}, _model_task_runner)
    else:
        task = None
    data = _state_to_dict(orchestrator.state)
    data["deleted"] = deleted
    data["queued_task"] = task
    return data


@app.post("/api/projects/{project_id}/cleanup")
async def cleanup_project_stage(project_id: str, req: CleanupRequest) -> dict[str, Any]:
    existing = _tasks.get(project_id)
    if existing and not existing.done():
        raise HTTPException(status_code=409, detail="Project is already running")
    try:
        stage = Stage(req.stage)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid stage: {req.stage}") from exc
    if stage not in {Stage.SCRIPTING, Stage.ASSET_GEN, Stage.VIDEO_GEN, Stage.EDITING}:
        raise HTTPException(status_code=400, detail=f"Stage cannot be cleaned: {stage.value}")

    state = _load_state(project_id)
    deleted = _cleanup_stage_files(state, stage) if req.delete_files else []
    _clear_stage_state(state, stage)
    data = _state_to_dict(state)
    data["deleted"] = deleted
    return data


@app.post("/api/projects/{project_id}/shots/{shot_id}/cleanup")
async def cleanup_project_shot(project_id: str, shot_id: str, req: ShotCleanupRequest) -> dict[str, Any]:
    existing = _tasks.get(project_id)
    if existing and not existing.done():
        raise HTTPException(status_code=409, detail="Project is already running")
    state = _load_state(project_id)
    paths: list[str | Path] = list(_shot_output_paths(shot_id))
    if req.include_asset:
        paths.append(canonical_shot(shot_id, ROOT / "assets"))
    deleted = [_delete_file(path) for path in paths] if req.delete_files else []
    _cleanup_shot_state(state, shot_id, req.include_asset)
    data = _state_to_dict(state)
    data["deleted"] = deleted
    return data


@app.post("/api/projects/{project_id}/shots/{shot_id}/rerun")
async def rerun_project_shot(project_id: str, shot_id: str, req: ShotRerunRequest) -> dict[str, Any]:
    existing = _tasks.get(project_id)
    if existing and not existing.done():
        raise HTTPException(status_code=409, detail="Project is already running")
    state = _load_state(project_id)
    paths: list[str | Path] = list(_shot_output_paths(shot_id))
    if req.include_asset:
        paths.append(canonical_shot(shot_id, ROOT / "assets"))
    deleted = [_delete_file(path) for path in paths] if req.delete_files else []
    _cleanup_shot_state(state, shot_id, req.include_asset)
    if req.start:
        _start_background(PipelineOrchestrator(state))
    data = _state_to_dict(state)
    data["deleted"] = deleted
    return data


@app.get("/api/projects/{project_id}/events")
async def project_events(project_id: str) -> StreamingResponse:
    async def _stream():
        last_count = 0
        while True:
            try:
                state = PipelineState.load(project_id)
            except FileNotFoundError:
                yield "event: error\ndata: {\"detail\":\"project not found\"}\n\n"
                return
            events = state.events[last_count:]
            last_count = len(state.events)
            if events:
                for event in events:
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            else:
                payload = {
                    "type": "heartbeat",
                    "stage": state.current_stage.value,
                    "message": state.last_message or "",
                    "current": state.progress_current,
                    "total": state.progress_total,
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if state.current_stage in (Stage.DONE, Stage.ERROR) and not (
                project_id in _tasks and not _tasks[project_id].done()
            ):
                return
            await asyncio.sleep(1)

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/api/queue")
async def get_queue() -> dict[str, Any]:
    return {
        "model_execution_enabled": ENABLE_MODEL_TASKS,
        "active_task_id": _active_queue_task_id,
        "tasks": [_public_task(_queue_records[task_id]) for task_id in _queue_order if task_id in _queue_records],
    }


@app.post("/api/queue/{task_id}/cancel")
async def cancel_queue_task(task_id: str) -> dict[str, Any]:
    global _queue_runner
    record = _queue_records.get(task_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if record["status"] in {"completed", "failed", "canceled"}:
        return _public_task(record)
    if task_id == _active_queue_task_id and _queue_runner and not _queue_runner.done():
        _queue_runner.cancel()
        _queue_runner = None
    _task_record(task_id, "canceled", "Task canceled by user")
    # Restart worker if there are still queued tasks
    queue = _ensure_queue()
    if (_queue_runner is None or _queue_runner.done()) and not queue.empty():
        _queue_runner = asyncio.create_task(_queue_worker())
    project_id = record.get("project_id")
    if project_id:
        try:
            state = _load_state(project_id)
            state.queue_status = "canceled"
            state.update_progress(f"Queue task {record['kind']} canceled", event_type="queue")
            state.save()
        except HTTPException:
            pass
    return _public_task(record)


@app.get("/api/health")
async def health_check() -> dict[str, Any]:
    services = _load_services_for_health()
    asset_validation = validate_asset_setup()
    output_root = ROOT / "output"
    checks = {
        "ffmpeg": _command_available("ffmpeg"),
        "ffprobe": _command_available("ffprobe"),
        "output_writable": {
            "ok": output_root.exists() and os.access(output_root, os.W_OK),
            "path": str(output_root),
        },
        "assets_writable": {
            "ok": (ROOT / "assets").exists() and os.access(ROOT / "assets", os.W_OK),
            "path": str(ROOT / "assets"),
        },
        "models": {
            "ok": asset_validation["ok"],
            "errors": asset_validation.get("errors", []),
            "warnings": asset_validation.get("warnings", []),
        },
        "llm": _local_http_check(services.get("llm", {}).get("url", "")),
        "comfyui": _local_http_check(services.get("comfyui", {}).get("url", "")),
        "chattts": _local_http_check(services.get("chattts", {}).get("url", "")),
        "sadtalker": _local_http_check(services.get("sadtalker", {}).get("url", "")),
    }
    return {
        "ok": all(value.get("ok", False) for key, value in checks.items() if key not in {"llm", "comfyui", "chattts", "sadtalker"}),
        "model_execution_enabled": ENABLE_MODEL_TASKS,
        "checks": checks,
    }


@app.get("/api/projects/{project_id}/shots")
async def list_project_shots(project_id: str) -> list[dict[str, Any]]:
    state = _load_state(project_id)
    build_shot_states(state)
    state.save()
    return [shot.model_dump(mode="json") for shot in state.shot_states.values()]


@app.get("/api/projects/{project_id}/shots/{shot_id}")
async def get_project_shot(project_id: str, shot_id: str) -> dict[str, Any]:
    state = _load_state(project_id)
    build_shot_states(state)
    if shot_id not in state.shot_states:
        raise HTTPException(status_code=404, detail=f"Shot not found: {shot_id}")
    state.save()
    return state.shot_states[shot_id].model_dump(mode="json")


@app.put("/api/projects/{project_id}/shots/{shot_id}")
async def update_project_shot(project_id: str, shot_id: str, req: ShotUpdateRequest) -> dict[str, Any]:
    state = _load_state(project_id)
    try:
        shot = update_script_shot(state, shot_id, req.data)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Shot not found: {shot_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    state.save()
    return shot.model_dump(mode="json")


@app.post("/api/projects/{project_id}/shots/{shot_id}/lock")
async def lock_project_shot(project_id: str, shot_id: str) -> dict[str, Any]:
    state = _load_state(project_id)
    try:
        shot = set_shot_lock(state, shot_id, True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Shot not found: {shot_id}") from exc
    state.save()
    return shot.model_dump(mode="json")


@app.post("/api/projects/{project_id}/shots/{shot_id}/unlock")
async def unlock_project_shot(project_id: str, shot_id: str) -> dict[str, Any]:
    state = _load_state(project_id)
    try:
        shot = set_shot_lock(state, shot_id, False)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Shot not found: {shot_id}") from exc
    state.save()
    return shot.model_dump(mode="json")


@app.post("/api/projects/{project_id}/shots/{shot_id}/review")
async def review_project_shot(project_id: str, shot_id: str, req: ReviewRequest) -> dict[str, Any]:
    state = _load_state(project_id)
    try:
        shot = review_shot(state, shot_id, req.status, req.note)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Shot not found: {shot_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    state.save()
    return shot.model_dump(mode="json")


@app.post("/api/projects/{project_id}/shots/{shot_id}/quality-check")
async def quality_check_project_shot(project_id: str, shot_id: str) -> dict[str, Any]:
    state = _load_state(project_id)
    try:
        shot = await asyncio.get_running_loop().run_in_executor(
            None, quality_check_shot, state, shot_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Shot not found: {shot_id}") from exc
    state.save()
    return shot.model_dump(mode="json")


@app.post("/api/projects/{project_id}/quality-check")
async def quality_check_full_project(project_id: str) -> dict[str, Any]:
    state = _load_state(project_id)
    report = await asyncio.get_running_loop().run_in_executor(
        None, quality_check_project, state,
    )
    state.save()
    return report


@app.post("/api/projects/{project_id}/retry-failed")
async def retry_failed_project_shots(project_id: str) -> dict[str, Any]:
    state = _load_state(project_id)
    queued = mark_failed_for_retry(state)
    state.save()
    task = _enqueue("retry_failed_shots", project_id, {"shot_ids": queued}, _model_task_runner) if queued else None
    return {"ok": True, "shot_ids": queued, "queued_task": task, "state": _state_to_dict(_load_state(project_id))}


@app.post("/api/projects/{project_id}/export")
async def create_project_export(project_id: str) -> dict[str, Any]:
    _load_state(project_id)
    return _enqueue("export_project", project_id, {}, _export_task_runner)


@app.get("/api/projects/{project_id}/export")
async def get_project_export(project_id: str, download: bool = Query(False)):
    state = _load_state(project_id)
    if download:
        zip_path = state.export_manifest.get("zip")
        if not zip_path or not Path(zip_path).exists():
            raise HTTPException(status_code=404, detail="Export zip not found")
        return FileResponse(zip_path, media_type="application/zip", filename=Path(zip_path).name)
    return state.export_manifest or {"ok": False, "message": "No export package has been created"}


@app.post("/api/assets/validate")
async def validate_assets() -> dict[str, Any]:
    return validate_asset_setup()


@app.get("/api/assets")
async def get_assets_config() -> dict[str, Any]:
    return {
        "config": load_asset_config(),
        "manifest": load_asset_manifest(),
        "config_yaml": ASSET_CONFIG_PATH.read_text(encoding="utf-8") if ASSET_CONFIG_PATH.exists() else "",
        "manifest_yaml": ASSET_MANIFEST_PATH.read_text(encoding="utf-8") if ASSET_MANIFEST_PATH.exists() else "",
    }


@app.put("/api/assets/manifest")
async def update_manifest(req: AssetTextUpdate) -> dict[str, Any]:
    try:
        data = yaml.safe_load(req.content) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Manifest must be a YAML mapping")
    save_asset_manifest(data)
    return {"ok": True, "manifest": load_asset_manifest()}


@app.put("/api/assets/config")
async def update_config(req: AssetTextUpdate) -> dict[str, Any]:
    try:
        data = yaml.safe_load(req.content) or {}
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Config must be a YAML mapping")
    save_asset_config(data)
    return {"ok": True, "config": load_asset_config()}


@app.put("/api/assets/config-json")
async def update_config_json(req: AssetMappingUpdate) -> dict[str, Any]:
    save_asset_config(req.data)
    return {
        "ok": True,
        "config": load_asset_config(),
        "config_yaml": ASSET_CONFIG_PATH.read_text(encoding="utf-8"),
    }


@app.post("/api/assets/character-lora")
async def update_character_lora(req: CharacterLoraUpdate) -> dict[str, Any]:
    manifest = load_asset_manifest()
    char = manifest.setdefault("characters", {}).setdefault(req.character, {})
    char["lora"] = {
        "enabled": req.enabled,
        "name": req.name,
        "trigger": req.trigger,
        "strength_model": req.strength_model,
        "strength_clip": req.strength_clip,
    }
    save_asset_manifest(manifest)
    return {"ok": True, "manifest": manifest}


@app.put("/api/assets/scene-audio")
async def update_scene_audio(req: SceneAudioUpdate) -> dict[str, Any]:
    manifest = load_asset_manifest()
    manifest.setdefault("scene_audio", {})[req.scene_id] = {
        "audio_path": req.audio_path,
        "segments": req.segments,
    }
    save_asset_manifest(manifest)
    return {"ok": True, "manifest": manifest}


@app.delete("/api/assets/binding")
async def delete_asset_binding(
    asset_type: str = Query(...),
    key: str = Query(...),
    emotion: str = Query(""),
    delete_file: bool = Query(False),
) -> dict[str, Any]:
    manifest = load_asset_manifest()
    path_value: str | None = None

    if asset_type == "character_reference":
        char = manifest.get("characters", {}).get(key, {})
        if isinstance(char, dict):
            path_value = char.pop("reference", None)
    elif asset_type == "character_expression":
        normalized_emotion = emotion.strip() or "neutral"
        char = manifest.get("characters", {}).get(key, {})
        expressions = char.get("expressions", {}) if isinstance(char, dict) else {}
        path_value = expressions.pop(normalized_emotion, None)
    elif asset_type == "character_lora":
        char = manifest.get("characters", {}).get(key, {})
        if isinstance(char, dict):
            path_value = None
            char.pop("lora", None)
    elif asset_type == "character":
        char = manifest.get("characters", {}).pop(key, None)
        if isinstance(char, dict):
            path_value = char.get("reference")
    elif asset_type == "scene":
        path_value = manifest.get("scenes", {}).pop(key, None)
    elif asset_type == "shot":
        path_value = manifest.get("shots", {}).pop(key, None)
    elif asset_type == "scene_audio":
        scene_audio = manifest.get("scene_audio", {}).pop(key, None)
        if isinstance(scene_audio, dict):
            path_value = scene_audio.get("audio_path")
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported asset_type: {asset_type}")

    deleted = _delete_file(path_value) if delete_file and path_value else None
    save_asset_manifest(manifest)
    return {"ok": True, "removed_path": path_value, "deleted": deleted, "manifest": manifest}


@app.post("/api/assets/upload")
async def upload_asset(
    asset_type: str = Form(...),
    key: str = Form(...),
    emotion: str = Form(""),
    file: UploadFile = File(...),
) -> dict[str, Any]:
    if asset_type not in {"character_reference", "character_expression", "scene", "shot"}:
        raise HTTPException(status_code=400, detail=f"Unsupported asset_type: {asset_type}")
    if not key.strip():
        raise HTTPException(status_code=400, detail="key is required")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise HTTPException(status_code=400, detail="Only png, jpg, jpeg, and webp uploads are supported")

    manifest = load_asset_manifest()
    safe_key = slugify(key)
    if asset_type == "character_reference":
        rel = Path("assets/library/characters") / safe_key / f"reference{suffix}"
        manifest.setdefault("characters", {}).setdefault(key, {})["reference"] = str(rel)
    elif asset_type == "character_expression":
        normalized_emotion = emotion.strip() or "neutral"
        if normalized_emotion not in EXPRESSION_VARIANTS:
            raise HTTPException(status_code=400, detail=f"Unsupported emotion: {normalized_emotion}")
        rel = Path("assets/library/characters") / safe_key / f"{normalized_emotion}{suffix}"
        char = manifest.setdefault("characters", {}).setdefault(key, {})
        char.setdefault("expressions", {})[normalized_emotion] = str(rel)
    elif asset_type == "scene":
        rel = Path("assets/library/scenes") / f"{safe_key}{suffix}"
        manifest.setdefault("scenes", {})[key] = str(rel)
    else:
        rel = Path("assets/library/shots") / f"{safe_key}{suffix}"
        manifest.setdefault("shots", {})[key] = str(rel)

    dest = ROOT / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh)

    save_asset_manifest(manifest)
    return {"ok": True, "path": str(rel), "manifest": manifest}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("AI_COMIC_HOST", "0.0.0.0")
    port = int(os.getenv("AI_COMIC_PORT", "8080"))
    uvicorn.run(app, host=host, port=port)
