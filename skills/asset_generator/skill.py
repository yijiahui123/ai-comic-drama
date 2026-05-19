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
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp

from utils import slugify as _slugify
from utils.assets import (
    EXPRESSION_VARIANTS,
    AssetLibrary,
    canonical_character_expression,
    canonical_character_reference,
    canonical_scene,
    canonical_shot,
    copy_asset_to_canonical,
    load_asset_config,
    load_asset_manifest,
    normalize_shot_emotion,
)
from utils.logger import get_logger
from utils.paths import PROJECT_ROOT

logger = get_logger(__name__)

from utils.comfyui import post_with_retry as _post_with_retry, poll_comfyui_ws as _poll_comfyui_ws

# Default paths to ComfyUI workflow template files
_WORKFLOWS_DIR = Path(__file__).parent / "workflows"

# ComfyUI API paths
_COMFYUI_PROMPT_PATH = "/prompt"
_COMFYUI_HISTORY_PATH = "/history/{prompt_id}"
_COMFYUI_VIEW_PATH = "/view"
_COMFYUI_UPLOAD_PATH = "/upload/image"

# Polling configuration
_POLL_INTERVAL = 3.0   # seconds
_POLL_TIMEOUT = 600.0  # seconds (10 min)

# Assets root
_ASSETS_ROOT = PROJECT_ROOT / "assets"


def _load_workflow(filename: str) -> dict[str, Any]:
    """Load a ComfyUI workflow JSON template from the workflows directory.

    Args:
        filename: File name (e.g. ``character_gen.json``).

    Returns:
        Parsed workflow dictionary.
    """
    path = _WORKFLOWS_DIR / filename
    return json.loads(path.read_text(encoding="utf-8"))


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
        shot_progress_callback: Optional[Callable[[str, int, float, str], None]] = None,
    ) -> None:
        """
        Args:
            comfyui_url: Base URL of the ComfyUI API server.
            assets_root: Root directory where generated assets will be saved.
            progress_callback: Optional callable ``(message, current, total)`` for
                progress reporting.
            shot_progress_callback: ``(asset_key, pct, eta_seconds, node)`` for per-asset progress.
        """
        self.comfyui_url = comfyui_url.rstrip("/")
        self.assets_root = Path(assets_root)
        self._progress_callback = progress_callback
        self._shot_progress_callback = shot_progress_callback
        self._session: Optional[aiohttp.ClientSession] = None
        self._asset_config = load_asset_config()
        self._library = AssetLibrary(load_asset_manifest(), self._asset_config)
        self.asset_sources: dict[str, dict[str, str]] = {
            "characters": {},
            "expressions": {},
            "scenes": {},
            "shots": {},
        }

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

        # Build character design card lookup: name → flux_prompt
        char_cards = {
            c["name"]: c
            for c in script.get("characters", [])
            if isinstance(c, dict) and "name" in c
        }

        total = len(characters) * (1 + len(EXPRESSION_VARIANTS)) + len(scene_stubs) + len(shots)
        current = 0

        # ---- Characters ----
        char_paths: list[Path] = []
        for char_name in characters:
            card = char_cards.get(char_name)
            paths = await self._generate_character(
                char_name, script.get("style", "anime"), card,
            )
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
            path = await self._generate_shot(shot, char_cards)
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

    async def _generate_character(
        self,
        char_name: str,
        style: str,
        card: Optional[dict[str, Any]] = None,
    ) -> list[Path]:
        """Generate reference sheet + expression variants for *char_name*.

        Uses ``character_gen.json`` (pure text-to-image) for the reference sheet
        and ``character_expression.json`` (Flux.2 + ReferenceLatent) for
        expression variants so the character appearance stays consistent.

        When a character design card (from ScriptWriter) is provided, its
        ``flux_prompt`` field is used as the base prompt for better consistency.

        Args:
            char_name: Character name.
            style: Visual style description.
            card: Optional character design card dict with ``flux_prompt``.

        Returns:
            List of saved image paths.
        """
        char_dir = self.assets_root / "characters" / _slugify(char_name)
        char_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []

        # Reference image (pure text-to-image)
        ref_path = canonical_character_reference(char_name, self.assets_root)
        library_ref = self._library.character_reference(char_name)
        if library_ref:
            copy_asset_to_canonical(library_ref, ref_path)
            self.asset_sources["characters"][str(ref_path)] = "library"
        if not ref_path.exists():
            if card and card.get("flux_prompt"):
                # Use character design card prompt for consistency
                prompt = (
                    f"Character design sheet for '{char_name}', "
                    f"{card['flux_prompt']}, "
                    f"front view, full body, white background, {style}"
                )
            else:
                prompt = (
                    f"Character design sheet for '{char_name}', {style}, "
                    "front view, full body, white background, detailed anime character design, "
                    "high quality illustration"
                )
            workflow = _load_workflow("character_gen.json")
            workflow = self._inject_prompt(workflow, prompt)
            workflow = self._inject_loras(workflow, [self._library.style_lora(), self._library.character_lora(char_name)])
            image_bytes = await self._run_workflow(workflow, context=f"char:{char_name}")
            if image_bytes:
                ref_path.write_bytes(image_bytes)
                logger.info("Saved character reference: %s", ref_path)
                self.asset_sources["characters"][str(ref_path)] = "generated"
        else:
            self.asset_sources["characters"].setdefault(str(ref_path), "canonical")
        paths.append(ref_path)

        # Upload reference image to ComfyUI for use as ReferenceLatent source
        ref_comfyui_name: Optional[str] = None
        if ref_path.exists():
            ref_comfyui_name = await self._upload_image(ref_path)
            if not ref_comfyui_name:
                logger.warning(
                    "Failed to upload reference image for '%s' — "
                    "expression variants will use pure text-to-image fallback",
                    char_name,
                )

        # Expression variants (ReferenceLatent when possible, fallback to pure t2i)
        expr_dir = char_dir / "expressions"
        expr_dir.mkdir(exist_ok=True)
        for expression in EXPRESSION_VARIANTS:
            expr_path = canonical_character_expression(char_name, expression, self.assets_root)
            library_expr = self._library.character_expression(char_name, expression)
            if library_expr:
                copy_asset_to_canonical(library_expr, expr_path)
                self.asset_sources["expressions"][str(expr_path)] = "library"
            if not expr_path.exists():
                if card and card.get("flux_prompt"):
                    prompt = (
                        f"Close-up portrait of '{char_name}', {expression} expression, "
                        f"{card['flux_prompt']}, detailed face, white background, {style}"
                    )
                else:
                    prompt = (
                        f"Close-up portrait of '{char_name}', {expression} expression, "
                        f"{style}, detailed face, white background, high quality anime art"
                    )
                if ref_comfyui_name:
                    workflow = _load_workflow("character_expression.json")
                    workflow = self._inject_prompt(workflow, prompt)
                    workflow = self._inject_reference_image(workflow, ref_comfyui_name)
                else:
                    workflow = _load_workflow("character_gen.json")
                    workflow = self._inject_prompt(workflow, prompt)
                workflow = self._inject_loras(workflow, [self._library.style_lora(), self._library.character_lora(char_name)])
                image_bytes = await self._run_workflow(workflow, context=f"expr:{char_name}:{expression}")
                if image_bytes:
                    expr_path.write_bytes(image_bytes)
                    logger.debug("Saved expression variant: %s", expr_path)
                    self.asset_sources["expressions"][str(expr_path)] = "generated"
            else:
                self.asset_sources["expressions"].setdefault(str(expr_path), "canonical")
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
        out_path = canonical_scene(scene["scene_id"], self.assets_root)
        library_scene = self._library.scene(scene["scene_id"])
        if library_scene:
            copy_asset_to_canonical(library_scene, out_path)
            self.asset_sources["scenes"][str(out_path)] = "library"

        if out_path.exists():
            self.asset_sources["scenes"].setdefault(str(out_path), "canonical")
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
        workflow = self._inject_loras(workflow, [self._library.style_lora()])
        image_bytes = await self._run_workflow(workflow, context=f"scene:{scene.get('scene_id', '')}")
        if image_bytes:
            out_path.write_bytes(image_bytes)
            logger.info("Saved scene background: %s", out_path)
            self.asset_sources["scenes"][str(out_path)] = "generated"
            return out_path

        return None

    async def _generate_shot(
        self,
        shot: dict[str, Any],
        char_cards: Optional[dict[str, dict[str, Any]]] = None,
    ) -> Optional[Path]:
        """Generate the storyboard image for *shot*.

        Args:
            shot: Shot dictionary containing ``shot_id`` and ``visual_prompt``.

        Returns:
            Path to the saved image, or ``None`` on failure.
        """
        shots_dir = self.assets_root / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)
        out_path = canonical_shot(shot["shot_id"], self.assets_root)
        library_shot = self._library.shot(shot["shot_id"])
        if library_shot:
            copy_asset_to_canonical(library_shot, out_path)
            self.asset_sources["shots"][str(out_path)] = "library"

        if out_path.exists():
            self.asset_sources["shots"].setdefault(str(out_path), "canonical")
            return out_path

        visual_prompt = shot.get("visual_prompt", "")
        if not visual_prompt:
            logger.warning("Shot %s has no visual_prompt — skipping", shot.get("shot_id"))
            return None

        prompt = self._enrich_shot_prompt(shot, visual_prompt, char_cards or {})
        reference_image = await self._select_shot_reference_image(shot)
        workflow = _load_workflow("shot_gen.json" if reference_image else "shot_gen_text.json")
        workflow = self._inject_prompt(workflow, prompt)
        if reference_image:
            uploaded_ref = await self._upload_image(reference_image)
            if uploaded_ref:
                workflow = self._inject_reference_image(workflow, uploaded_ref)
        loras = [self._library.style_lora()]
        for character in shot.get("characters", []):
            loras.append(self._library.character_lora(character))
        workflow = self._inject_loras(workflow, loras)
        image_bytes = await self._run_workflow(workflow, context=shot.get("shot_id", ""))
        if image_bytes:
            out_path.write_bytes(image_bytes)
            logger.info("Saved shot image: %s", out_path)
            self.asset_sources["shots"][str(out_path)] = "generated"
            return out_path

        return None

    def _enrich_shot_prompt(
        self,
        shot: dict[str, Any],
        visual_prompt: str,
        char_cards: dict[str, dict[str, Any]],
    ) -> str:
        triggers: list[str] = []
        descriptions: list[str] = []
        for character in shot.get("characters", []):
            lora = self._library.character_lora(character)
            if lora and lora.get("trigger"):
                triggers.append(str(lora["trigger"]))
            card = char_cards.get(character, {})
            if card.get("flux_prompt") and card["flux_prompt"] not in visual_prompt:
                descriptions.append(str(card["flux_prompt"]))
        prefix = ", ".join([*triggers, *descriptions])
        emotion = normalize_shot_emotion(shot)
        if prefix:
            return f"{prefix}, {emotion} expression, {visual_prompt}"
        return f"{emotion} expression, {visual_prompt}"

    async def _select_shot_reference_image(self, shot: dict[str, Any]) -> Optional[Path]:
        refs = self._asset_config.get("references", {})
        characters = [c for c in shot.get("characters", []) if c]
        if not characters:
            return None
        character = characters[0]
        emotion = normalize_shot_emotion(shot)
        if refs.get("use_expression_reference_for_shots", True):
            expr = canonical_character_expression(character, emotion, self.assets_root)
            if expr.exists():
                return expr
        if refs.get("use_character_reference_for_shots", True):
            ref = canonical_character_reference(character, self.assets_root)
            if ref.exists():
                return ref
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
        original_nodes = list(workflow.values())
        for node in original_nodes:
            if isinstance(node, dict):
                inputs = node.get("inputs", {})
                if "__PROMPT_PLACEHOLDER__" in inputs.get("text", ""):
                    inputs["text"] = prompt_text
        return workflow

    @staticmethod
    def _inject_reference_image(workflow: dict[str, Any], image_filename: str) -> dict[str, Any]:
        """Replace ``__REF_IMAGE_PLACEHOLDER__`` in *workflow* with *image_filename*.

        The placeholder is expected to appear as the ``image`` input inside a
        ``LoadImage`` node in the workflow.

        Args:
            workflow: ComfyUI workflow dictionary (modified in-place).
            image_filename: The ComfyUI-side filename of the uploaded reference image.

        Returns:
            The modified workflow dictionary.
        """
        for node in workflow.values():
            if isinstance(node, dict):
                inputs = node.get("inputs", {})
                if inputs.get("image") == "__REF_IMAGE_PLACEHOLDER__":
                    inputs["image"] = image_filename
        return workflow

    @staticmethod
    def _inject_loras(
        workflow: dict[str, Any],
        loras: list[Optional[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Inject optional LoRA nodes between model loaders and guider nodes."""
        active = [lora for lora in loras if lora and lora.get("name")]
        if not active:
            return workflow

        model_node_id: Optional[str] = None
        for node_id, node in workflow.items():
            if isinstance(node, dict) and node.get("class_type") == "UNETLoader":
                model_node_id = node_id
                break
        if model_node_id is None:
            return workflow

        current_model: list[Any] = [model_node_id, 0]
        next_id = max((int(k) for k in workflow.keys() if str(k).isdigit()), default=0) + 1
        for lora in active:
            node_id = str(next_id)
            next_id += 1
            workflow[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": current_model,
                    "lora_name": lora["name"],
                    "strength_model": float(lora.get("strength_model", 0.7)),
                },
            }
            current_model = [node_id, 0]

        original_nodes = [
            node
            for node in workflow.values()
            if isinstance(node, dict)
            and node.get("class_type") != "LoraLoaderModelOnly"
        ]
        for node in original_nodes:
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs", {})
            if inputs.get("model") == [model_node_id, 0]:
                inputs["model"] = current_model
        return workflow

    async def _upload_image(self, image_path: Path) -> Optional[str]:
        """Upload a local image to ComfyUI's ``/upload/image`` endpoint.

        Args:
            image_path: Path to the local image file.

        Returns:
            The filename assigned by ComfyUI, or ``None`` on failure.
        """
        session = self._get_session()
        try:
            data = aiohttp.FormData()
            with open(image_path, "rb") as fh:
                data.add_field(
                    "image",
                    fh,
                    filename=image_path.name,
                    content_type="image/png",
                )
            async with session.post(
                f"{self.comfyui_url}{_COMFYUI_UPLOAD_PATH}",
                data=data,
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result.get("name")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to upload image %s: %s", image_path, exc)
            return None

    async def _run_workflow(self, workflow: dict[str, Any], context: str = "") -> Optional[bytes]:
        """Submit *workflow* to ComfyUI and poll until the image is ready.

        Uses WebSocket for real-time progress, falls back to HTTP polling.

        Args:
            workflow: ComfyUI workflow dictionary.
            context: Asset identifier for progress reporting (e.g. shot_id).

        Returns:
            Raw image bytes, or ``None`` if the workflow failed.
        """
        client_id = str(uuid.uuid4())
        payload = {"prompt": workflow, "client_id": client_id}
        session = self._get_session()

        # Submit
        try:
            data = await _post_with_retry(
                session, f"{self.comfyui_url}{_COMFYUI_PROMPT_PATH}", payload,
            )
            prompt_id: str = data["prompt_id"]
        except (ConnectionError, TimeoutError) as exc:
            logger.error("ComfyUI prompt submission failed: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.error("ComfyUI prompt submission failed: %s", exc)
            return None

        # WebSocket progress tracking
        t0 = time.monotonic()
        def _on_ws_progress(val: int, mx: int, node: str) -> None:
            if mx > 0:
                pct = int(val / mx * 100)
                elapsed = time.monotonic() - t0
                eta = (elapsed / val * (mx - val)) if val > 0 else 0
                if self._progress_callback:
                    self._progress_callback(
                        f"生成中 {pct}% — 节点 {node or '...'} — ETA {int(eta)}s",
                        val, mx,
                    )
                if self._shot_progress_callback and context:
                    self._shot_progress_callback(context, pct, eta, node or "")

        ws_done = await _poll_comfyui_ws(
            self.comfyui_url, prompt_id, client_id,
            timeout=_POLL_TIMEOUT, on_progress=_on_ws_progress,
        )

        # Fetch result via HTTP
        deadline = time.monotonic() + (0 if ws_done else _POLL_TIMEOUT)
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
                continue

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

        # WS said done but HTTP didn't find it — retry once
        if ws_done:
            try:
                async with session.get(
                    f"{self.comfyui_url}{_COMFYUI_HISTORY_PATH.format(prompt_id=prompt_id)}"
                ) as resp:
                    resp.raise_for_status()
                    history = await resp.json()
                    if prompt_id in history:
                        for node_output in history[prompt_id].get("outputs", {}).values():
                            images = node_output.get("images", [])
                            if images:
                                image_info = images[0]
                                return await self._download_image(
                                    image_info["filename"],
                                    image_info.get("subfolder", ""),
                                    image_info.get("type", "output"),
                                )
            except Exception:  # noqa: BLE001
                pass

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
