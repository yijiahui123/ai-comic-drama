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

import yaml

from pipeline.state import PipelineState, Stage, StageResult
from skills.script_writer.skill import ScriptWriter
from skills.asset_generator.skill import AssetGenerator
from skills.video_generator.skill import VideoGenerator
from skills.editor.skill import Editor
from utils.logger import get_pipeline_logger
from utils.model_unloader import force_gc, unload_comfyui_models, unload_ollama_model


_CONFIGS_DIR = Path("configs")


def _load_services_config() -> dict[str, Any]:
    """Load ``configs/services.yaml`` if it exists."""
    path = _CONFIGS_DIR / "services.yaml"
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    return {}


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
        self._services = _load_services_config()

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

        ollama_url: str = svc.get("ollama", {}).get("url", "http://localhost:11434")
        ollama_model: str = svc.get("ollama", {}).get("model", "llama3.1:70b-instruct-q4_K_M")
        comfyui_url: str = svc.get("comfyui", {}).get("url", "http://localhost:8188")

        # --- Stage: SCRIPTING ---
        if not self.state.is_stage_done(Stage.SCRIPTING):
            script = await self._run_stage(
                Stage.SCRIPTING,
                self._do_scripting,
                ollama_url=ollama_url,
                model=ollama_model,
            )
            if script is None:
                return self.state
            self.state.script = script

        # Unload Ollama LLM after scripting to free memory before asset generation
        if unload_enabled:
            try:
                await unload_ollama_model(ollama_url, ollama_model)
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
            )
            if manifest is None:
                return self.state
            self.state.video_manifest = {k: [str(p) for p in v] for k, v in manifest.items()}

        # Unload ComfyUI Wan2.1 models after video generation
        if unload_enabled:
            try:
                await unload_comfyui_models(comfyui_url)
                force_gc()
                await asyncio.sleep(gc_delay)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Memory unload after VIDEO_GEN failed (non-fatal): %s", exc)

        # --- Stage: EDITING ---
        if not self.state.is_stage_done(Stage.EDITING):
            final_path = await self._run_stage(
                Stage.EDITING,
                self._do_editing,
            )
            if final_path:
                self.state.final_video = str(final_path)

        self.state.current_stage = Stage.DONE
        self.state.save()
        self.logger.info(
            "Pipeline DONE — final_video=%s", self.state.final_video
        )
        return self.state

    def status(self) -> str:
        """Return a human-readable status summary for this project."""
        return self.state.status_summary()

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
        result = StageResult(stage=stage, status="running", started_at=datetime.now(timezone.utc))
        self.state.stages[stage.value] = result
        self.state.save()

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
            self.state.save()
            return output
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
            self.state.save()
            return None

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    async def _do_scripting(self, ollama_url: str, model: str) -> dict[str, Any]:
        """Run the ScriptWriter skill."""
        writer = ScriptWriter(ollama_url=ollama_url, model=model)
        script = await writer.generate(self.state.user_prompt)
        return script

    async def _do_asset_gen(self, comfyui_url: str) -> dict[str, Any]:
        """Run the AssetGenerator skill."""
        if self.state.script is None:
            raise RuntimeError("No script available for asset generation.")
        async with AssetGenerator(
            comfyui_url=comfyui_url,
            progress_callback=lambda msg, cur, tot: self.logger.info(
                "  AssetGen [%d/%d] %s", cur, tot, msg
            ),
        ) as gen:
            return await gen.generate_all_assets(self.state.script)

    async def _do_video_gen(
        self, comfyui_url: str, chattts_url: str, sadtalker_url: str
    ) -> dict[str, Any]:
        """Run the VideoGenerator skill."""
        if self.state.script is None:
            raise RuntimeError("No script available for video generation.")
        async with VideoGenerator(
            comfyui_url=comfyui_url,
            chattts_url=chattts_url,
            sadtalker_url=sadtalker_url,
        ) as gen:
            return await gen.generate_all(self.state.script)

    async def _do_editing(self) -> Optional[Path]:
        """Run the Editor skill."""
        if self.state.script is None:
            raise RuntimeError("No script available for editing.")
        editor = Editor(project_id=self.state.project_id)
        return await editor.edit(self.state.script)
