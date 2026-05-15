"""FastAPI web console for the AI Comic Drama pipeline."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
    load_asset_config,
    load_asset_manifest,
    save_asset_config,
    save_asset_manifest,
    validate_asset_setup,
)
from utils.logger import get_logger

logger = get_logger("web_server")

ROOT = Path.cwd()
WEB_DIR = ROOT / "web"
STATE_DIR = ROOT / "output" / "state"

app = FastAPI(title="AI Comic Drama Console")
_tasks: dict[str, asyncio.Task] = {}


class CreateProjectRequest(BaseModel):
    prompt: str = Field(min_length=1)
    profile: str = "default"


class RerunRequest(BaseModel):
    stage: str


class AssetTextUpdate(BaseModel):
    content: str


class CharacterLoraUpdate(BaseModel):
    character: str = Field(min_length=1)
    enabled: bool = False
    name: str = ""
    trigger: str = ""
    strength_model: float = 0.85
    strength_clip: float = 0.8


def _state_to_dict(state: PipelineState) -> dict[str, Any]:
    return json.loads(state.model_dump_json())


def _load_state(project_id: str) -> PipelineState:
    try:
        return PipelineState.load(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Project not found: {project_id}") from exc


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
    orchestrator.state.update_progress(f"Project created with profile {req.profile}", 0, 0, "created")
    orchestrator.state.save()
    _start_background(orchestrator)
    return _state_to_dict(orchestrator.state)


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
            data = _state_to_dict(state)
            data["running"] = path.stem in _tasks and not _tasks[path.stem].done()
            projects.append(data)
    return projects


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str) -> dict[str, Any]:
    state = _load_state(project_id)
    data = _state_to_dict(state)
    data["running"] = project_id in _tasks and not _tasks[project_id].done()
    return data


@app.post("/api/projects/{project_id}/resume")
async def resume_project(project_id: str) -> dict[str, Any]:
    orchestrator = PipelineOrchestrator.resume(project_id)
    _start_background(orchestrator)
    return _state_to_dict(orchestrator.state)


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
    orchestrator.reset_from_stage(stage)
    _start_background(orchestrator)
    return _state_to_dict(orchestrator.state)


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

    uvicorn.run(app, host="127.0.0.1", port=8080)
