"""Pipeline state model and persistence.

Uses Pydantic for data modelling and plain JSON files for persistence so that
the pipeline can resume after an interruption.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


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
    final_video: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def is_stage_done(self, stage: Stage) -> bool:
        """Return ``True`` if *stage* has completed successfully."""
        result = self.stages.get(stage.value)
        return result is not None and result.status == "done"

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

    def save(self, state_dir: Path = Path("output/state")) -> Path:
        """Persist this state object to a JSON file.

        Args:
            state_dir: Directory in which to write the state file.

        Returns:
            Path to the written JSON file.
        """
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / f"{self.project_id}.json"
        self.updated_at = datetime.utcnow()
        path.write_text(
            self.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, project_id: str, state_dir: Path = Path("output/state")) -> "PipelineState":
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

        if self.final_video:
            lines.append(f"\nFinal video: {self.final_video}")

        return "\n".join(lines)
