"""AssetGenerator Skill.

Reads a structured script dictionary and drives ComfyUI's REST API to generate:

* **Character reference sheets** — one base image per character, plus expression variants.
* **Scene background images** — one background per unique location.
* **Shot images** — each script shot rendered using its ``visual_prompt``.

All generated assets are de-duplicated: if an asset file already exists on disk it
is reused rather than regenerated.  Output files are organised as::

    assets/
    ├── characters/<name>/reference.png
    │                     expressions/<emotion>.png
    ├── scenes/<scene_id>.png
    └── shots/<shot_id>.png
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp

from utils.logger import get_logger

logger = get_logger(__name__)

# Default paths to ComfyUI workflow template files
_WORKFLOWS_DIR = Path(__file__).parent / "workflows"

# ComfyUI API paths
_COMFYUI_PROMPT_PATH = "/prompt"
_COMFYUI_HISTORY_PATH = "/history/{prompt_id}"
_COMFYUI_VIEW_PATH = "/view"

# Polling configuration
_POLL_INTERVAL = 3.0   # seconds
_POLL_TIMEOUT = 600.0  # seconds (10 min)

# Assets root
_ASSETS_ROOT = Path("assets")

EXPRESSION_VARIANTS = ["neutral", "happy", "surprised", "angry", "sad"]


def _load_workflow(filename: str) -> dict[str, Any]:
    """Load a ComfyUI workflow JSON template from the workflows directory.

    Args:
        filename: File name (e.g. ``character_gen.json``).

    Returns:
        Parsed workflow dictionary.
    """
    path = _WORKFLOWS_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


def _slugify(text: str) -> str:
    """Convert *text* into a safe file-system-friendly slug.

    Args:
        text: Input string.

    Returns:
        Lower-cased, alphanumeric-and-hyphen-only slug.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:64]


class AssetGenerator:
    """Generates visual assets for a comic-drama script using ComfyUI.

    Attributes:
        comfyui_url: Base URL of the ComfyUI server.
        assets_root: Root directory for generated assets.
    """

    def __init__(
        self,
        comfyui_url: str = "http://localhost:8188",
        assets_root: Path = _ASSETS_ROOT,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> None:
        """
        Args:
            comfyui_url: Base URL of the ComfyUI API server.
            assets_root: Root directory where generated assets will be saved.
            progress_callback: Optional callable ``(message, current, total)`` for
                progress reporting.
        """
        self.comfyui_url = comfyui_url.rstrip("/")
        self.assets_root = Path(assets_root)
        self._progress_callback = progress_callback
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "AssetGenerator":
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30, connect=10)
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def health_check(self) -> bool:
        """Return ``True`` if ComfyUI is reachable."""
        url = f"{self.comfyui_url}/"
        session = self._get_session()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status < 400
        except Exception:  # noqa: BLE001
            return False

    async def generate_all_assets(self, script: dict[str, Any]) -> dict[str, list[Path]]:
        """Generate all assets required by *script*.

        Runs three generation passes in sequence: characters → scenes → shots.

        Args:
            script: Validated script dictionary.

        Returns:
            Dictionary with keys ``"characters"``, ``"scenes"``, ``"shots"`` mapping
            to lists of generated file paths.
        """
        self.assets_root.mkdir(parents=True, exist_ok=True)

        characters = self._extract_characters(script)
        scene_stubs = self._extract_scenes(script)
        shots = self._extract_shots(script)

        total = len(characters) * (1 + len(EXPRESSION_VARIANTS)) + len(scene_stubs) + len(shots)
        current = 0

        # ---- Characters ----
        char_paths: list[Path] = []
        for char_name in characters:
            paths = await self._generate_character(char_name, script.get("style", "anime"))
            char_paths.extend(paths)
            current += 1 + len(EXPRESSION_VARIANTS)
            self._report(f"Character '{char_name}' generated", current, total)

        # ---- Scenes ----
        scene_paths: list[Path] = []
        for scene in scene_stubs:
            path = await self._generate_scene(scene, script.get("style", "anime"))
            if path:
                scene_paths.append(path)
            current += 1
            self._report(f"Scene '{scene['scene_id']}' generated", current, total)

        # ---- Shots ----
        shot_paths: list[Path] = []
        for shot in shots:
            path = await self._generate_shot(shot)
            if path:
                shot_paths.append(path)
            current += 1
            self._report(f"Shot '{shot['shot_id']}' generated", current, total)

        logger.info(
            "AssetGenerator complete: %d character files, %d scenes, %d shots",
            len(char_paths),
            len(scene_paths),
            len(shot_paths),
        )
        return {
            "characters": char_paths,
            "scenes": scene_paths,
            "shots": shot_paths,
        }

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_characters(script: dict[str, Any]) -> list[str]:
        """Return a de-duplicated, ordered list of character names from *script*."""
        seen: set[str] = set()
        result: list[str] = []
        for episode in script.get("episodes", []):
            for scene in episode.get("scenes", []):
                for shot in scene.get("shots", []):
                    for char in shot.get("characters", []):
                        if char and char not in seen:
                            seen.add(char)
                            result.append(char)
        return result

    @staticmethod
    def _extract_scenes(script: dict[str, Any]) -> list[dict[str, Any]]:
        """Return a de-duplicated list of scene stubs (by ``scene_id``)."""
        seen: set[str] = set()
        result: list[dict[str, Any]] = []
        for episode in script.get("episodes", []):
            for scene in episode.get("scenes", []):
                sid = scene.get("scene_id", "")
                if sid and sid not in seen:
                    seen.add(sid)
                    result.append(scene)
        return result

    @staticmethod
    def _extract_shots(script: dict[str, Any]) -> list[dict[str, Any]]:
        """Return all shots from *script*."""
        result: list[dict[str, Any]] = []
        for episode in script.get("episodes", []):
            for scene in episode.get("scenes", []):
                for shot in scene.get("shots", []):
                    result.append(shot)
        return result

    # ------------------------------------------------------------------
    # Generation methods
    # ------------------------------------------------------------------

    async def _generate_character(self, char_name: str, style: str) -> list[Path]:
        """Generate reference sheet + expression variants for *char_name*.

        Args:
            char_name: Character name.
            style: Visual style description.

        Returns:
            List of saved image paths.
        """
        char_dir = self.assets_root / "characters" / _slugify(char_name)
        char_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []

        # Reference image
        ref_path = char_dir / "reference.png"
        if not ref_path.exists():
            prompt = (
                f"Character design sheet for '{char_name}', {style}, "
                "front view, full body, white background, detailed anime character design, "
                "high quality illustration"
            )
            workflow = _load_workflow("character_gen.json")
            workflow = self._inject_prompt(workflow, prompt)
            image_bytes = await self._run_workflow(workflow)
            if image_bytes:
                ref_path.write_bytes(image_bytes)
                logger.info("Saved character reference: %s", ref_path)
        paths.append(ref_path)

        # Expression variants
        expr_dir = char_dir / "expressions"
        expr_dir.mkdir(exist_ok=True)
        for expression in EXPRESSION_VARIANTS:
            expr_path = expr_dir / f"{expression}.png"
            if not expr_path.exists():
                prompt = (
                    f"Close-up portrait of '{char_name}', {expression} expression, "
                    f"{style}, detailed face, white background, high quality anime art"
                )
                workflow = _load_workflow("character_gen.json")
                workflow = self._inject_prompt(workflow, prompt)
                image_bytes = await self._run_workflow(workflow)
                if image_bytes:
                    expr_path.write_bytes(image_bytes)
                    logger.debug("Saved expression variant: %s", expr_path)
            paths.append(expr_path)

        return paths

    async def _generate_scene(self, scene: dict[str, Any], style: str) -> Optional[Path]:
        """Generate a background image for *scene*.

        Args:
            scene: Scene dictionary with ``scene_id`` and ``location``.
            style: Visual style description.

        Returns:
            Path to the saved image, or ``None`` on failure.
        """
        scene_dir = self.assets_root / "scenes"
        scene_dir.mkdir(parents=True, exist_ok=True)
        out_path = scene_dir / f"{_slugify(scene['scene_id'])}.png"

        if out_path.exists():
            return out_path

        location = scene.get("location", "unknown location")
        time_of_day = scene.get("time", "day")
        atmosphere = scene.get("atmosphere", "")
        prompt = (
            f"Background scene: {location}, {time_of_day}, {atmosphere}, "
            f"{style}, no characters, cinematic composition, detailed environment art, "
            "high quality digital painting"
        )
        workflow = _load_workflow("scene_gen.json")
        workflow = self._inject_prompt(workflow, prompt)
        image_bytes = await self._run_workflow(workflow)
        if image_bytes:
            out_path.write_bytes(image_bytes)
            logger.info("Saved scene background: %s", out_path)
            return out_path

        return None

    async def _generate_shot(self, shot: dict[str, Any]) -> Optional[Path]:
        """Generate the storyboard image for *shot*.

        Args:
            shot: Shot dictionary containing ``shot_id`` and ``visual_prompt``.

        Returns:
            Path to the saved image, or ``None`` on failure.
        """
        shots_dir = self.assets_root / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)
        out_path = shots_dir / f"{_slugify(shot['shot_id'])}.png"

        if out_path.exists():
            return out_path

        visual_prompt = shot.get("visual_prompt", "")
        if not visual_prompt:
            logger.warning("Shot %s has no visual_prompt — skipping", shot.get("shot_id"))
            return None

        workflow = _load_workflow("shot_gen.json")
        workflow = self._inject_prompt(workflow, visual_prompt)
        image_bytes = await self._run_workflow(workflow)
        if image_bytes:
            out_path.write_bytes(image_bytes)
            logger.info("Saved shot image: %s", out_path)
            return out_path

        return None

    # ------------------------------------------------------------------
    # ComfyUI interaction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _inject_prompt(workflow: dict[str, Any], prompt_text: str) -> dict[str, Any]:
        """Replace the ``__PROMPT_PLACEHOLDER__`` value in *workflow* with *prompt_text*.

        The placeholder is expected to appear as the ``text`` input inside any
        ``CLIPTextEncode`` node in the workflow.

        Args:
            workflow: ComfyUI workflow dictionary (modified in-place).
            prompt_text: The positive prompt text to inject.

        Returns:
            The modified workflow dictionary.
        """
        for node in workflow.values():
            if isinstance(node, dict):
                inputs = node.get("inputs", {})
                if "__PROMPT_PLACEHOLDER__" in inputs.get("text", ""):
                    inputs["text"] = prompt_text
        return workflow

    async def _run_workflow(self, workflow: dict[str, Any]) -> Optional[bytes]:
        """Submit *workflow* to ComfyUI and poll until the image is ready.

        Args:
            workflow: ComfyUI workflow dictionary.

        Returns:
            Raw image bytes, or ``None`` if the workflow failed.
        """
        client_id = str(uuid.uuid4())
        payload = {"prompt": workflow, "client_id": client_id}
        session = self._get_session()

        # Submit
        try:
            async with session.post(
                f"{self.comfyui_url}{_COMFYUI_PROMPT_PATH}",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                prompt_id: str = data["prompt_id"]
        except Exception as exc:  # noqa: BLE001
            logger.error("ComfyUI prompt submission failed: %s", exc)
            return None

        # Poll for completion
        deadline = time.monotonic() + _POLL_TIMEOUT
        while time.monotonic() < deadline:
            await asyncio.sleep(_POLL_INTERVAL)
            try:
                async with session.get(
                    f"{self.comfyui_url}{_COMFYUI_HISTORY_PATH.format(prompt_id=prompt_id)}"
                ) as resp:
                    resp.raise_for_status()
                    history = await resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("ComfyUI history poll failed: %s", exc)
                continue

            if prompt_id not in history:
                continue  # Still running

            outputs = history[prompt_id].get("outputs", {})
            for node_output in outputs.values():
                images = node_output.get("images", [])
                if images:
                    image_info = images[0]
                    return await self._download_image(
                        image_info["filename"],
                        image_info.get("subfolder", ""),
                        image_info.get("type", "output"),
                    )

        logger.error("ComfyUI workflow timed out after %.0fs", _POLL_TIMEOUT)
        return None

    async def _download_image(
        self, filename: str, subfolder: str, image_type: str
    ) -> Optional[bytes]:
        """Download a generated image from the ComfyUI ``/view`` endpoint.

        Args:
            filename: Image filename.
            subfolder: ComfyUI subfolder.
            image_type: Image type (e.g. ``output``).

        Returns:
            Raw image bytes, or ``None`` on failure.
        """
        params = {"filename": filename, "subfolder": subfolder, "type": image_type}
        session = self._get_session()
        try:
            async with session.get(
                f"{self.comfyui_url}{_COMFYUI_VIEW_PATH}", params=params
            ) as resp:
                resp.raise_for_status()
                return await resp.read()
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to download image %s: %s", filename, exc)
            return None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30, connect=10)
            )
        return self._session

    def _report(self, message: str, current: int, total: int) -> None:
        """Invoke the progress callback if one was provided."""
        logger.info("[%d/%d] %s", current, total, message)
        if self._progress_callback:
            self._progress_callback(message, current, total)
