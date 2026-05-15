"""Script JSON schema validation utilities."""

from __future__ import annotations

from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Expected structure constants
# ---------------------------------------------------------------------------

REQUIRED_SCRIPT_KEYS = {"title", "style", "characters", "episodes"}
REQUIRED_EPISODE_KEYS = {"episode", "scenes"}
REQUIRED_SCENE_KEYS = {"scene_id", "location", "shots"}
REQUIRED_SHOT_KEYS = {
    "shot_id",
    "type",
    "characters",
    "dialogue",
    "visual_prompt",
    "motion_prompt",
    "camera_move",
    "duration",
    "mood",
}
MAX_SHOT_DURATION_SECONDS = 5.0
VALID_SHOT_TYPES = {"全景", "中景", "特写", "近景", "远景", "俯视", "仰视"}
VALID_CAMERA_MOVES = {
    "static", "slow pan down", "slight zoom in",
    "slow push-in", "slow tracking", "handheld shake",
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_keys(obj: dict[str, Any], required: set[str], context: str) -> list[str]:
    """Return a list of missing required keys in *obj*.

    Args:
        obj: Dictionary to inspect.
        required: Set of required key names.
        context: Human-readable location string for error messages.
    """
    missing = required - set(obj.keys())
    return [f"[{context}] Missing required key: '{k}'" for k in sorted(missing)]


def validate_script(script: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a structured script dictionary against the expected schema.

    Performs the following checks:

    * Top-level required keys (``title``, ``style``, ``episodes``).
    * Each episode contains ``episode`` and ``scenes``.
    * Each scene contains ``scene_id``, ``location``, and ``shots``.
    * Each shot contains all required fields and has a valid ``type``.
    * Character name consistency across shots (no unnamed / empty strings).

    Args:
        script: Parsed script dictionary (typically produced by ScriptWriter).

    Returns:
        A ``(is_valid, errors)`` tuple where *errors* is a list of human-readable
        problem descriptions. *is_valid* is ``True`` when *errors* is empty.
    """
    errors: list[str] = []

    # --- Top-level ---
    errors.extend(_check_keys(script, REQUIRED_SCRIPT_KEYS, "script"))
    if errors:
        return False, errors

    character_names: set[str] = set()

    # --- Character design cards ---
    for char in script.get("characters", []):
        if not isinstance(char, dict):
            errors.append("[characters] Each entry must be a dict.")
            continue
        for key in ("name", "appearance", "flux_prompt"):
            if key not in char or not str(char[key]).strip():
                errors.append(f"[characters] Character missing '{key}'.")

    for ep_idx, episode in enumerate(script.get("episodes", [])):
        ep_ctx = f"episode[{ep_idx}]"
        errors.extend(_check_keys(episode, REQUIRED_EPISODE_KEYS, ep_ctx))

        for sc_idx, scene in enumerate(episode.get("scenes", [])):
            sc_ctx = f"{ep_ctx}.scene[{sc_idx}]"
            errors.extend(_check_keys(scene, REQUIRED_SCENE_KEYS, sc_ctx))

            for sh_idx, shot in enumerate(scene.get("shots", [])):
                sh_ctx = f"{sc_ctx}.shot[{sh_idx}]"
                errors.extend(_check_keys(shot, REQUIRED_SHOT_KEYS, sh_ctx))

                # Validate shot type
                shot_type = shot.get("type", "")
                if shot_type and shot_type not in VALID_SHOT_TYPES:
                    errors.append(
                        f"[{sh_ctx}] Unknown shot type '{shot_type}'. "
                        f"Valid values: {sorted(VALID_SHOT_TYPES)}"
                    )

                # Validate camera move
                cam = shot.get("camera_move", "")
                if cam and cam not in VALID_CAMERA_MOVES:
                    errors.append(
                        f"[{sh_ctx}] Unknown camera_move '{cam}'. "
                        f"Valid values: {sorted(VALID_CAMERA_MOVES)}"
                    )

                # Validate motion_prompt is non-empty
                mp = shot.get("motion_prompt", "")
                if isinstance(mp, str) and not mp.strip():
                    errors.append(f"[{sh_ctx}] 'motion_prompt' must not be empty.")

                # Validate characters list
                for char in shot.get("characters", []):
                    if not isinstance(char, str) or not char.strip():
                        errors.append(f"[{sh_ctx}] Character name must be a non-empty string.")
                    else:
                        character_names.add(char.strip())

                # Validate duration. Over-limit durations are split before video generation.
                duration = shot.get("duration")
                if duration is not None and (not isinstance(duration, (int, float)) or duration <= 0):
                    errors.append(f"[{sh_ctx}] 'duration' must be a positive number, got {duration!r}.")

                # Validate visual_prompt is non-empty
                vp = shot.get("visual_prompt", "")
                if isinstance(vp, str) and not vp.strip():
                    errors.append(f"[{sh_ctx}] 'visual_prompt' must not be empty.")

    is_valid = len(errors) == 0
    if is_valid:
        logger.info(
            "Script validation passed. Title='%s', characters=%s",
            script.get("title"),
            sorted(character_names),
        )
    else:
        logger.warning("Script validation failed with %d error(s).", len(errors))
        for err in errors:
            logger.warning("  %s", err)

    return is_valid, errors
