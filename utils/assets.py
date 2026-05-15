"""Asset library, configuration, and generation helpers."""

from __future__ import annotations

import copy
import math
import shutil
from pathlib import Path
from typing import Any, Optional

import yaml

from utils import slugify
from utils.validators import MAX_SHOT_DURATION_SECONDS


ASSET_CONFIG_PATH = Path("configs/assets.yaml")
ASSET_MANIFEST_PATH = Path("assets/library/manifest.yaml")
ASSETS_ROOT = Path("assets")

EXPRESSION_VARIANTS = ["neutral", "happy", "surprised", "angry", "sad"]

DEFAULT_ASSET_CONFIG: dict[str, Any] = {
    "comfyui_models_root": "",
    "style_lora": {
        "enabled": False,
        "name": "",
        "strength_model": 0.7,
        "strength_clip": 0.7,
    },
    "character_loras": {
        "enabled": False,
        "optional": True,
    },
    "references": {
        "prefer_library_assets": True,
        "generate_missing_assets": True,
        "use_character_reference_for_shots": True,
        "use_expression_reference_for_shots": True,
    },
    "continuity": {
        "max_shot_duration_seconds": MAX_SHOT_DURATION_SECONDS,
        "split_long_actions": True,
        "use_previous_tail_frame": True,
    },
    "voice": {
        "scene_level_tts": True,
        "fallback_per_shot_tts": True,
    },
}

_EMOTION_ALIASES = {
    "neutral": "neutral",
    "calm": "neutral",
    "peaceful": "neutral",
    "normal": "neutral",
    "tense": "neutral",
    "serious": "neutral",
    "happy": "happy",
    "joy": "happy",
    "joyful": "happy",
    "smile": "happy",
    "excited": "happy",
    "angry": "angry",
    "rage": "angry",
    "furious": "angry",
    "sad": "sad",
    "sorrow": "sad",
    "crying": "sad",
    "melancholy": "sad",
    "surprised": "surprised",
    "shock": "surprised",
    "shocked": "surprised",
    "fear": "surprised",
    "scared": "surprised",
    "中性": "neutral",
    "平静": "neutral",
    "紧张": "neutral",
    "严肃": "neutral",
    "开心": "happy",
    "快乐": "happy",
    "高兴": "happy",
    "兴奋": "happy",
    "愤怒": "angry",
    "生气": "angry",
    "悲伤": "sad",
    "难过": "sad",
    "哭泣": "sad",
    "惊讶": "surprised",
    "震惊": "surprised",
    "恐惧": "surprised",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def load_asset_config(path: Path = ASSET_CONFIG_PATH) -> dict[str, Any]:
    """Load asset policy config with defaults applied."""
    return _deep_merge(DEFAULT_ASSET_CONFIG, _load_yaml(path))


def load_asset_manifest(path: Path = ASSET_MANIFEST_PATH) -> dict[str, Any]:
    """Load the optional user asset manifest."""
    manifest = _load_yaml(path)
    manifest.setdefault("characters", {})
    manifest.setdefault("scenes", {})
    manifest.setdefault("shots", {})
    return manifest


def save_asset_config(config: dict[str, Any], path: Path = ASSET_CONFIG_PATH) -> Path:
    """Persist asset policy config as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def save_asset_manifest(manifest: dict[str, Any], path: Path = ASSET_MANIFEST_PATH) -> Path:
    """Persist the user asset manifest as YAML."""
    manifest.setdefault("characters", {})
    manifest.setdefault("scenes", {})
    manifest.setdefault("shots", {})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(manifest, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def repo_path(path_value: str | Path | None) -> Optional[Path]:
    """Return a repository-relative path as an absolute-ish Path object."""
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def canonical_character_reference(character: str, assets_root: Path = ASSETS_ROOT) -> Path:
    return assets_root / "characters" / slugify(character) / "reference.png"


def canonical_character_expression(
    character: str, emotion: str, assets_root: Path = ASSETS_ROOT
) -> Path:
    return assets_root / "characters" / slugify(character) / "expressions" / f"{emotion}.png"


def canonical_scene(scene_id: str, assets_root: Path = ASSETS_ROOT) -> Path:
    return assets_root / "scenes" / f"{slugify(scene_id)}.png"


def canonical_shot(shot_id: str, assets_root: Path = ASSETS_ROOT) -> Path:
    return assets_root / "shots" / f"{slugify(shot_id)}.png"


def copy_asset_to_canonical(source: Path, destination: Path) -> Path:
    """Copy a user-library asset into the canonical generated path if needed."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve() and not destination.exists():
        shutil.copy2(source, destination)
    return destination


def normalize_emotion(value: Any) -> str:
    if not value:
        return "neutral"
    text = str(value).strip().lower()
    return _EMOTION_ALIASES.get(text, "neutral")


def normalize_shot_emotion(shot: dict[str, Any]) -> str:
    emotion = normalize_emotion(shot.get("emotion") or shot.get("mood"))
    shot["emotion"] = emotion
    return emotion


def iter_script_shots(script: dict[str, Any]):
    """Yield ``(episode, scene, shot)`` tuples for a script."""
    for episode in script.get("episodes", []):
        for scene in episode.get("scenes", []):
            for shot in scene.get("shots", []):
                yield episode, scene, shot


def normalize_script_for_generation(
    script: dict[str, Any],
    max_duration: float = MAX_SHOT_DURATION_SECONDS,
    split_long_actions: bool = True,
) -> dict[str, Any]:
    """Normalize emotions and split over-limit shots in place."""
    for _, scene, shot in list(iter_script_shots(script)):
        normalize_shot_emotion(shot)

    if not split_long_actions:
        for _, _, shot in iter_script_shots(script):
            try:
                duration = float(shot.get("duration", 4))
            except (TypeError, ValueError):
                shot["duration"] = 4
                continue
            shot["duration"] = min(max(duration, 0.1), max_duration)
        return script

    for episode in script.get("episodes", []):
        for scene in episode.get("scenes", []):
            expanded: list[dict[str, Any]] = []
            for shot in scene.get("shots", []):
                try:
                    duration = float(shot.get("duration", 4))
                except (TypeError, ValueError):
                    duration = 4
                if duration <= max_duration:
                    shot["duration"] = max(duration, 0.1)
                    expanded.append(shot)
                    continue

                parts = max(1, math.ceil(duration / max_duration))
                remaining = duration
                original_id = str(shot.get("shot_id", "shot"))
                for idx in range(parts):
                    part = copy.deepcopy(shot)
                    part["shot_id"] = f"{original_id}-{idx + 1:02d}"
                    part["duration"] = min(max_duration, remaining)
                    if idx > 0:
                        part["dialogue"] = ""
                    motion = part.get("motion_prompt", "")
                    suffix = f" Continuity segment {idx + 1} of {parts}; preserve pose and movement direction."
                    part["motion_prompt"] = f"{motion}{suffix}".strip()
                    normalize_shot_emotion(part)
                    expanded.append(part)
                    remaining -= max_duration
            scene["shots"] = expanded
    return script


class AssetLibrary:
    """Resolver for optional user-provided assets and LoRA metadata."""

    def __init__(
        self,
        manifest: Optional[dict[str, Any]] = None,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.manifest = manifest if manifest is not None else load_asset_manifest()
        self.config = config if config is not None else load_asset_config()

    def character_reference(self, character: str) -> Optional[Path]:
        if not self.config.get("references", {}).get("prefer_library_assets", True):
            return None
        data = self.manifest.get("characters", {}).get(character, {})
        return self._existing(data.get("reference"))

    def character_expression(self, character: str, emotion: str) -> Optional[Path]:
        if not self.config.get("references", {}).get("prefer_library_assets", True):
            return None
        data = self.manifest.get("characters", {}).get(character, {})
        expressions = data.get("expressions", {}) if isinstance(data, dict) else {}
        return self._existing(expressions.get(emotion))

    def scene(self, scene_id: str) -> Optional[Path]:
        if not self.config.get("references", {}).get("prefer_library_assets", True):
            return None
        return self._existing(self.manifest.get("scenes", {}).get(scene_id))

    def shot(self, shot_id: str) -> Optional[Path]:
        if not self.config.get("references", {}).get("prefer_library_assets", True):
            return None
        return self._existing(self.manifest.get("shots", {}).get(shot_id))

    def character_lora(self, character: str) -> Optional[dict[str, Any]]:
        if not self.config.get("character_loras", {}).get("enabled", False):
            return None
        data = self.manifest.get("characters", {}).get(character, {})
        lora = data.get("lora", {}) if isinstance(data, dict) else {}
        if not lora.get("enabled", False):
            return None
        name = str(lora.get("name", "")).strip()
        if not name:
            return None
        return {
            "name": name,
            "trigger": str(lora.get("trigger", "")).strip(),
            "strength_model": float(lora.get("strength_model", 0.85)),
            "strength_clip": float(lora.get("strength_clip", 0.8)),
        }

    def style_lora(self) -> Optional[dict[str, Any]]:
        cfg = self.config.get("style_lora", {})
        if not cfg.get("enabled", False):
            return None
        name = str(cfg.get("name", "")).strip()
        if not name:
            return None
        return {
            "name": name,
            "trigger": str(cfg.get("trigger", "")).strip(),
            "strength_model": float(cfg.get("strength_model", 0.7)),
            "strength_clip": float(cfg.get("strength_clip", 0.7)),
        }

    @staticmethod
    def _existing(value: Any) -> Optional[Path]:
        path = repo_path(value)
        if path and path.exists():
            return path
        return None


def validate_asset_setup(
    config_path: Path = ASSET_CONFIG_PATH,
    manifest_path: Path = ASSET_MANIFEST_PATH,
) -> dict[str, Any]:
    """Validate configured library assets and optional LoRA references."""
    config = load_asset_config(config_path)
    manifest = load_asset_manifest(manifest_path)
    errors: list[str] = []
    warnings: list[str] = []

    for character, data in manifest.get("characters", {}).items():
        if not isinstance(data, dict):
            errors.append(f"characters.{character} must be a mapping")
            continue
        ref = data.get("reference")
        ref_path = repo_path(ref)
        if ref and (ref_path is None or not ref_path.exists()):
            errors.append(f"Missing reference image for {character}: {ref}")
        for emotion, path_value in (data.get("expressions") or {}).items():
            if emotion not in EXPRESSION_VARIANTS:
                warnings.append(f"Unknown expression '{emotion}' for {character}")
            path = repo_path(path_value)
            if path_value and (path is None or not path.exists()):
                errors.append(f"Missing expression image for {character}/{emotion}: {path_value}")
        lora = data.get("lora") or {}
        if lora.get("enabled") and not lora.get("name"):
            errors.append(f"Character LoRA for {character} is enabled but has no name")

    for scene_id, path_value in manifest.get("scenes", {}).items():
        path = repo_path(path_value)
        if path_value and (path is None or not path.exists()):
            errors.append(f"Missing scene image for {scene_id}: {path_value}")

    for shot_id, path_value in manifest.get("shots", {}).items():
        path = repo_path(path_value)
        if path_value and (path is None or not path.exists()):
            errors.append(f"Missing shot image for {shot_id}: {path_value}")

    style_lora = config.get("style_lora", {})
    if style_lora.get("enabled") and not style_lora.get("name"):
        errors.append("style_lora is enabled but has no name")

    root = repo_path(config.get("comfyui_models_root"))
    if root and root.exists():
        lora_root = root / "loras"
        if style_lora.get("enabled") and style_lora.get("name"):
            if not (lora_root / style_lora["name"]).exists():
                errors.append(f"Missing style LoRA in ComfyUI loras directory: {style_lora['name']}")
        for character, data in manifest.get("characters", {}).items():
            lora = (data or {}).get("lora") or {}
            if lora.get("enabled") and lora.get("name") and not (lora_root / lora["name"]).exists():
                optional = config.get("character_loras", {}).get("optional", True)
                msg = f"Missing character LoRA in ComfyUI loras directory for {character}: {lora['name']}"
                (warnings if optional else errors).append(msg)

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "config": config,
        "manifest": manifest,
    }
