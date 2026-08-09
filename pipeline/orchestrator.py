"""Pipeline Orchestrator.

Coordinates the four skills in order:

    SCRIPTING → ASSET_GEN → VIDEO_GEN → EDITING

State is persisted after each stage so the pipeline can resume from the last
successfully completed stage.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pipeline.state import PipelineState, Stage, StageResult
from skills.script_writer.skill import ScriptWriter
from skills.asset_generator.skill import AssetGenerator
from skills.video_generator.skill import VideoGenerator
from skills.editor.skill import Editor
from utils.assets import load_asset_config, normalize_script_for_generation
from utils.config import load_services_config
from utils.logger import get_pipeline_logger
from utils.model_unloader import force_gc, unload_comfyui_models, kill_omlx_server


class PipelineOrchestrator:
    """Runs the full AI comic-drama generation pipeline.

    Attributes:
        state: Current :class:`~pipeline.state.PipelineState`.
    """

    def __init__(self, state: PipelineState) -> None:
        """
        Args:
            state: Initial (or resumed) pipeline state.
        """
        self.state = state
        self.logger = get_pipeline_logger(state.project_id)
        self._services = load_services_config()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def new(cls, user_prompt: str) -> "PipelineOrchestrator":
        """Create a fresh pipeline run.

        Args:
            user_prompt: Natural-language description from the user.

        Returns:
            :class:`PipelineOrchestrator` with a new project ID.
        """
        project_id = uuid.uuid4().hex[:8]
        state = PipelineState(project_id=project_id, user_prompt=user_prompt)
        return cls(state)

    @classmethod
    def resume(cls, project_id: str) -> "PipelineOrchestrator":
        """Resume an existing pipeline run.

        Args:
            project_id: Project identifier of the run to resume.

        Returns:
            :class:`PipelineOrchestrator` loaded from persisted state.

        Raises:
            FileNotFoundError: If no state file is found for *project_id*.
        """
        state = PipelineState.load(project_id)
        return cls(state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> PipelineState:
        """Execute all pending stages and return the final state.

        Stages that are already marked as ``done`` are skipped automatically,
        enabling seamless resume after an interruption.

        Returns:
            Final :class:`~pipeline.state.PipelineState`.
        """
        self.logger.info(
            "Pipeline started — project_id=%s prompt=%r",
            self.state.project_id,
            self.state.user_prompt[:80],
        )
        self.state.save()

        svc = self._services
        mem_cfg = svc.get("memory", {})
        unload_enabled: bool = mem_cfg.get("unload_between_stages", True)
        gc_delay: float = float(mem_cfg.get("gc_delay", 2))

        llm_url: str = svc.get("llm", {}).get("url", "http://127.0.0.1:8000/v1")
        llm_model: str = svc.get("llm", {}).get("model", "Qwen3.6-35B-A3B-MLX-8bit")
        llm_api_key: str = svc.get("llm", {}).get("api_key", "")
        comfyui_url: str = svc.get("comfyui", {}).get("url", "http://localhost:8188")

        # --- Stage: SCRIPTING ---
        if not self.state.is_stage_done(Stage.SCRIPTING):
            script = await self._run_stage(
                Stage.SCRIPTING,
                self._do_scripting,
                llm_url=llm_url,
                model=llm_model,
                api_key=llm_api_key,
            )
            if script is None:
                return self.state
            self.state.script = script
            self.state.save(force=True)

        # Unload LLM after scripting to free memory before asset generation
        if unload_enabled:
            try:
                await kill_omlx_server()
                force_gc()
                await asyncio.sleep(gc_delay)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Memory unload after SCRIPTING failed (non-fatal): %s", exc)

        # --- Stage: ASSET_GEN ---
        if not self.state.is_stage_done(Stage.ASSET_GEN):
            manifest = await self._run_stage(
                Stage.ASSET_GEN,
                self._do_asset_gen,
                comfyui_url=comfyui_url,
            )
            if manifest is None:
                return self.state
            self.state.asset_manifest = {k: [str(p) for p in v] for k, v in manifest.items()}
            self.state.save(force=True)

        # Unload ComfyUI SDXL models after asset generation
        if unload_enabled:
            try:
                await unload_comfyui_models(comfyui_url)
                force_gc()
                await asyncio.sleep(gc_delay)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Memory unload after ASSET_GEN failed (non-fatal): %s", exc)

        # --- Stage: VIDEO_GEN ---
        if not self.state.is_stage_done(Stage.VIDEO_GEN):
            manifest = await self._run_stage(
                Stage.VIDEO_GEN,
                self._do_video_gen,
                comfyui_url=comfyui_url,
                chattts_url=svc.get("chattts", {}).get("url", "http://localhost:9966"),
                sadtalker_url=svc.get("sadtalker", {}).get("url", "http://localhost:7860"),
                chattts_enabled=svc.get("chattts", {}).get("enabled", True),
                sadtalker_enabled=svc.get("sadtalker", {}).get("enabled", True),
            )
            if manifest is None:
                return self.state
            self.state.video_manifest = {k: [str(p) for p in v] for k, v in manifest.items()}
            self.state.save(force=True)

        # Unload ComfyUI Wan 2.2 models after video generation
        if unload_enabled:
            try:
                await unload_comfyui_models(comfyui_url)
                force_gc()
                await asyncio.sleep(gc_delay)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Memory unload after VIDEO_GEN failed (non-fatal): %s", exc)

        # --- Stage: EDITING ---
        if not self.state.is_stage_done(Stage.EDITING):
            final_paths = await self._run_stage(
                Stage.EDITING,
                self._do_editing,
            )
            if final_paths:
                self.state.final_videos = [str(p) for p in final_paths]
                self.state.final_video = str(final_paths[-1])  # backwards compat
                self.state.save(force=True)

        self.state.current_stage = Stage.DONE
        self.state.save(force=True)
        self.logger.info(
            "Pipeline DONE — final_video=%s", self.state.final_video
        )

        # Trigger background cleanup of intermediate storage
        try:
            import subprocess
            from utils.paths import PROJECT_ROOT
            cleanup_script = PROJECT_ROOT / "scripts" / "cleanup_storage.py"
            if cleanup_script.exists():
                subprocess.Popen(["python3", str(cleanup_script), "--ttl", "7"])
                self.logger.info("Triggered background storage cleanup")
        except Exception as exc:
            self.logger.warning("Failed to trigger background storage cleanup: %s", exc)

        return self.state

    def status(self) -> str:
        """Return a human-readable status summary for this project."""
        return self.state.status_summary()

    def reset_from_stage(self, stage: Stage) -> None:
        """Mark *stage* and following stages as pending for a safe rerun."""
        reset = False
        for candidate in (Stage.SCRIPTING, Stage.ASSET_GEN, Stage.VIDEO_GEN, Stage.EDITING):
            if candidate == stage:
                reset = True
            if reset:
                self.state.stages.pop(candidate.value, None)

        if stage == Stage.SCRIPTING:
            self.state.script = None
            self.state.asset_manifest = {}
            self.state.asset_sources = {}
            self.state.video_manifest = {}
            self.state.final_video = None
            self.state.final_videos = []
        elif stage == Stage.ASSET_GEN:
            self.state.asset_manifest = {}
            self.state.asset_sources = {}
            self.state.video_manifest = {}
            self.state.final_video = None
            self.state.final_videos = []
        elif stage == Stage.VIDEO_GEN:
            self.state.video_manifest = {}
            self.state.final_video = None
            self.state.final_videos = []
        elif stage == Stage.EDITING:
            self.state.final_video = None
            self.state.final_videos = []

        self.state.current_stage = stage
        self.state.update_progress(f"Reset from stage {stage.value}", 0, 0, "reset")
        self.state.save(force=True)

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    async def _run_stage(self, stage: Stage, coro_fn, **kwargs) -> Any:
        """Execute *coro_fn* as a pipeline stage with timing and error handling.

        Args:
            stage: The :class:`~pipeline.state.Stage` being executed.
            coro_fn: Async callable that performs the stage work.
            **kwargs: Arguments forwarded to *coro_fn*.

        Returns:
            Return value of *coro_fn*, or ``None`` if it raised an exception.
        """
        self.logger.info("=== Stage %s: STARTED ===", stage.value)
        self.state.current_stage = stage
        self.state.update_progress(f"Stage {stage.value} started", 0, 0, "stage_start")
        result = StageResult(stage=stage, status="running", started_at=datetime.now(timezone.utc))
        self.state.stages[stage.value] = result
        self.state.save(force=True)

        t0 = time.monotonic()
        try:
            output = await coro_fn(**kwargs)
            elapsed = time.monotonic() - t0
            result.status = "done"
            result.completed_at = datetime.now(timezone.utc)
            result.elapsed_seconds = elapsed
            result.output_summary = str(output)[:200] if output is not None else None
            self.logger.info(
                "=== Stage %s: DONE (%.1fs) ===", stage.value, elapsed
            )
            self.state.update_progress(f"Stage {stage.value} done", self.state.progress_total, self.state.progress_total, "stage_done")
            self.state.save(force=True)
            return output
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - t0
            result.status = "error"
            result.completed_at = datetime.now(timezone.utc)
            result.elapsed_seconds = elapsed
            result.error = str(exc)
            self.state.current_stage = Stage.ERROR
            self.logger.error(
                "=== Stage %s: ERROR (%.1fs): %s ===", stage.value, elapsed, exc
            )
            self.state.update_progress(f"Stage {stage.value} error: {exc}", self.state.progress_current, self.state.progress_total, "stage_error")
            self.state.save(force=True)
            return None

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    async def _do_scripting(self, llm_url: str, model: str, api_key: str = "") -> dict[str, Any]:
        """Run the ScriptWriter skill."""
        async with ScriptWriter(base_url=llm_url, model=model, api_key=api_key or None) as writer:
            script = await writer.generate(self.state.user_prompt)
        asset_cfg = load_asset_config()
        continuity = asset_cfg.get("continuity", {})
        return normalize_script_for_generation(
            script,
            max_duration=float(continuity.get("max_shot_duration_seconds", 5)),
            split_long_actions=bool(continuity.get("split_long_actions", True)),
        )

    async def _do_asset_gen(self, comfyui_url: str) -> dict[str, Any]:
        """Run the AssetGenerator skill."""
        if self.state.script is None:
            raise RuntimeError("No script available for asset generation.")
        async with AssetGenerator(
            comfyui_url=comfyui_url,
            progress_callback=self._progress_callback("AssetGen"),
            shot_progress_callback=self._shot_progress_callback,
        ) as gen:
            result = await gen.generate_all_assets(self.state.script)
            self.state.asset_sources = gen.asset_sources
            return result

    async def _do_video_gen(
        self, comfyui_url: str, chattts_url: str, sadtalker_url: str,
        chattts_enabled: bool = True, sadtalker_enabled: bool = True,
    ) -> dict[str, Any]:
        """Run the VideoGenerator skill."""
        if self.state.script is None:
            raise RuntimeError("No script available for video generation.")
        async with VideoGenerator(
            comfyui_url=comfyui_url,
            chattts_url=chattts_url,
            sadtalker_url=sadtalker_url,
            progress_callback=self._progress_callback("VideoGen"),
            shot_progress_callback=self._shot_progress_callback,
            chattts_enabled=chattts_enabled,
            sadtalker_enabled=sadtalker_enabled,
        ) as gen:
            return await gen.generate_all(self.state.script)

    async def _do_editing(self) -> list[Path]:
        """Run the Editor skill."""
        if self.state.script is None:
            raise RuntimeError("No script available for editing.")
        editor = Editor(project_id=self.state.project_id)
        return await editor.edit(self.state.script)

    def _progress_callback(self, prefix: str):
        def _callback(message: str, current: int, total: int) -> None:
            self.logger.info("  %s [%d/%d] %s", prefix, current, total, message)
            self.state.update_progress(f"{prefix}: {message}", current, total)
            self.state.save()

        return _callback

    def _shot_progress_callback(self, shot_id: str, pct: int, eta: float, node: str) -> None:
        """Update per-shot progress from ComfyUI WebSocket."""
        shot = self.state.shot_states.get(shot_id)
        if shot:
            shot.progress_pct = pct
            shot.eta_seconds = round(eta, 1)
            shot.progress_node = node
            shot.updated_at = datetime.now(timezone.utc)
