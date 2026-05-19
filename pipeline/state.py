"""Pipeline state model and persistence.

Uses Pydantic for data modelling and plain JSON files for persistence so that
the pipeline can resume after an interruption.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from utils.paths import PROJECT_ROOT


class Stage(str, Enum):
    """Ordered pipeline stages."""

    INIT = "INIT"
    SCRIPTING = "SCRIPTING"
    ASSET_GEN = "ASSET_GEN"
    VIDEO_GEN = "VIDEO_GEN"
    EDITING = "EDITING"
    DONE = "DONE"
    ERROR = "ERROR"


# Ordered list used to determine stage progression
STAGE_ORDER: list[Stage] = [
    Stage.INIT,
    Stage.SCRIPTING,
    Stage.ASSET_GEN,
    Stage.VIDEO_GEN,
    Stage.EDITING,
    Stage.DONE,
]


class StageResult(BaseModel):
    """Result record for a single pipeline stage."""

    stage: Stage
    status: str = "pending"           # pending | running | done | error
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    elapsed_seconds: Optional[float] = None
    error: Optional[str] = None
    output_summary: Optional[str] = None


class QualityCheck(BaseModel):
    """One heuristic quality check result for a shot or project artifact."""

    name: str
    status: str = "pending"  # pass | warn | fail | pending | skipped
    message: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class ShotState(BaseModel):
    """Production state for a single shot."""

    shot_id: str
    scene_id: str = ""
    status: str = "pending"  # pending | ready | needs_retry | needs_review | approved | failed
    locked: bool = False
    review_status: str = "pending"  # pending | approved | rejected | needs_retry
    script: dict[str, Any] = Field(default_factory=dict)
    assets: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    sources: dict[str, str] = Field(default_factory=dict)
    quality_checks: list[QualityCheck] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 2
    last_error: Optional[str] = None
    generation_params: dict[str, Any] = Field(default_factory=dict)
    progress_pct: int = 0
    progress_node: str = ""
    eta_seconds: float = 0
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PipelineState(BaseModel):
    """Complete state for one pipeline run.

    Attributes:
        project_id: Unique identifier for this run.
        user_prompt: The original natural-language prompt.
        current_stage: The stage currently being executed (or last completed).
        stages: Per-stage result records.
        script: Generated script dictionary (populated after SCRIPTING).
        asset_manifest: Paths of all generated asset files.
        video_manifest: Paths of all generated video files.
        final_video: Path to the finished episode video.
        created_at: Run creation timestamp.
        updated_at: Last update timestamp.
    """

    project_id: str
    user_prompt: str
    current_stage: Stage = Stage.INIT
    stages: dict[str, StageResult] = Field(default_factory=dict)
    script: Optional[dict[str, Any]] = None
    asset_manifest: dict[str, list[str]] = Field(default_factory=dict)
    video_manifest: dict[str, list[str]] = Field(default_factory=dict)
    asset_sources: dict[str, dict[str, str]] = Field(default_factory=dict)
    queue_status: str = "idle"  # idle | queued | running | canceled | completed | failed
    shot_states: dict[str, ShotState] = Field(default_factory=dict)
    quality_report: dict[str, Any] = Field(default_factory=dict)
    review_queue: list[str] = Field(default_factory=list)
    export_manifest: dict[str, Any] = Field(default_factory=dict)
    progress_current: int = 0
    progress_total: int = 0
    last_message: Optional[str] = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    previews: dict[str, list[str]] = Field(default_factory=dict)
    final_video: Optional[str] = None  # Deprecated: kept for backwards compat
    final_videos: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    _last_save_time: float = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_stage_done(self, stage: Stage) -> bool:
        """Return ``True`` if *stage* has completed successfully."""
        result = self.stages.get(stage.value)
        return result is not None and result.status == "done"

    def update_progress(
        self,
        message: str,
        current: Optional[int] = None,
        total: Optional[int] = None,
        event_type: str = "progress",
    ) -> None:
        """Update progress fields and append a small event for UI clients."""
        if current is not None:
            self.progress_current = current
        if total is not None:
            self.progress_total = total
        self.last_message = message
        self.events.append(
            {
                "type": event_type,
                "stage": self.current_stage.value,
                "message": message,
                "current": self.progress_current,
                "total": self.progress_total,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        # Keep state files bounded while still useful for UI event replay.
        self.events = self.events[-500:]

    def set_previews(self) -> None:
        """Populate preview lists from current manifests and final output."""
        finals = self.final_videos or ([self.final_video] if self.final_video else [])
        self.previews = {
            "characters": self.asset_manifest.get("characters", [])[:50],
            "scenes": self.asset_manifest.get("scenes", [])[:50],
            "shots": self.asset_manifest.get("shots", [])[:100],
            "videos": self.video_manifest.get("videos", [])[:100],
            "audio": self.video_manifest.get("audio", [])[:100],
            "lipsync": self.video_manifest.get("lipsync", [])[:100],
            "final": finals,
        }

    def next_stage(self) -> Optional[Stage]:
        """Return the next stage to execute, or ``None`` if done/error."""
        if self.current_stage in (Stage.DONE, Stage.ERROR):
            return None
        try:
            idx = STAGE_ORDER.index(self.current_stage)
            return STAGE_ORDER[idx + 1]
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, state_dir: "Path" = PROJECT_ROOT / "output" / "state", force: bool = False) -> "Path":
        """Persist this state object to a JSON file.

        Args:
            state_dir: Directory in which to write the state file.
            force: If ``True``, skip the 2-second throttle (e.g. on stage change).

        Returns:
            Path to the written JSON file, or the expected path if throttled.
        """
        path = state_dir / f"{self.project_id}.json"
        # Throttle: skip write if < 2 seconds since last save (unless forced)
        now = time.monotonic()
        if not force and (now - self._last_save_time) < 2.0:
            return path
        state_dir.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(timezone.utc)
        self.set_previews()
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )
        os.replace(str(tmp_path), str(path))
        self._last_save_time = time.monotonic()
        return path

    @classmethod
    def load(cls, project_id: str, state_dir: "Path" = PROJECT_ROOT / "output" / "state") -> "PipelineState":
        """Load a state object from disk.

        Args:
            project_id: Project identifier.
            state_dir: Directory containing state files.

        Returns:
            Loaded :class:`PipelineState` instance.

        Raises:
            FileNotFoundError: If no state file exists for *project_id*.
        """
        path = state_dir / f"{project_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"No state file found for project '{project_id}'")
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    def status_summary(self) -> str:
        """Return a human-readable multi-line status summary."""
        lines = [
            f"Project:  {self.project_id}",
            f"Prompt:   {self.user_prompt[:80]}{'…' if len(self.user_prompt) > 80 else ''}",
            f"Stage:    {self.current_stage.value}",
            "",
            "Stage Results:",
        ]
        for stage in STAGE_ORDER:
            result = self.stages.get(stage.value)
            if result:
                elapsed = (
                    f" ({result.elapsed_seconds:.1f}s)" if result.elapsed_seconds else ""
                )
                err = f" — {result.error}" if result.error else ""
                lines.append(f"  {stage.value:<12} {result.status.upper()}{elapsed}{err}")
            else:
                lines.append(f"  {stage.value:<12} PENDING")

        finals = self.final_videos or ([self.final_video] if self.final_video else [])
        for v in finals:
            lines.append(f"\nFinal video: {v}")

        return "\n".join(lines)
