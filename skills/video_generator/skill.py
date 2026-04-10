"""VideoGenerator Skill.

For each shot in the script this skill:

1. **Image → video**: Submits the shot's storyboard image to ComfyUI's Wan2.1 node
   and retrieves a short video clip (3–6 s).
2. **AI voiceover**: Calls ChatTTS to synthesise the shot's dialogue line.
   Different characters are mapped to different speaker IDs via ``voice_config.yaml``.
3. **Lip-sync** *(optional)*: Drives the character's mouth in the video using
   SadTalker.  If the service is unavailable the step is silently skipped and the
   raw generated video is used instead.

Outputs are written to::

    output/
    ├── videos/<shot_id>.mp4
    ├── audio/<shot_id>.wav
    └── lipsync/<shot_id>_lipsync.mp4
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import aiohttp
import yaml

from utils import slugify as _slugify
from utils.logger import get_logger

logger = get_logger(__name__)

_CONFIGS_DIR = Path(__file__).parent / "configs"

# Output directories
_OUTPUT_VIDEOS = Path("output/videos")
_OUTPUT_AUDIO = Path("output/audio")
_OUTPUT_LIPSYNC = Path("output/lipsync")

# Polling
_POLL_INTERVAL = 5.0
_POLL_TIMEOUT = 900.0  # 15 min for video generation


class VideoGenerator:
    """Generates video clips, voiceovers, and lip-synced videos for each script shot.

    Attributes:
        comfyui_url: Base URL of the ComfyUI server.
        chattts_url: Base URL of the ChatTTS API server.
        sadtalker_url: Base URL of the SadTalker API server.
    """

    def __init__(
        self,
        comfyui_url: str = "http://localhost:8188",
        chattts_url: str = "http://localhost:9966",
        sadtalker_url: str = "http://localhost:7860",
    ) -> None:
        """
        Args:
            comfyui_url: ComfyUI server base URL.
            chattts_url: ChatTTS API server base URL.
            sadtalker_url: SadTalker Gradio server base URL.
        """
        self.comfyui_url = comfyui_url.rstrip("/")
        self.chattts_url = chattts_url.rstrip("/")
        self.sadtalker_url = sadtalker_url.rstrip("/")

        self._video_config = self._load_yaml("video_config.yaml")
        self._voice_config = self._load_yaml("voice_config.yaml")
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "VideoGenerator":
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60, connect=10)
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate_all(self, script: dict[str, Any]) -> dict[str, list[Path]]:
        """Generate video clips for every shot in *script*.

        Args:
            script: Validated script dictionary.

        Returns:
            Dictionary with ``"videos"``, ``"audio"``, and ``"lipsync"`` keys mapping
            to lists of output file paths.
        """
        _OUTPUT_VIDEOS.mkdir(parents=True, exist_ok=True)
        _OUTPUT_AUDIO.mkdir(parents=True, exist_ok=True)
        _OUTPUT_LIPSYNC.mkdir(parents=True, exist_ok=True)

        chattts_available = await self._check_service(self.chattts_url)
        sadtalker_available = await self._check_service(self.sadtalker_url)

        if not chattts_available:
            logger.warning("ChatTTS service unavailable — voiceover will be skipped.")
        if not sadtalker_available:
            logger.warning("SadTalker service unavailable — lip-sync will be skipped.")

        video_paths: list[Path] = []
        audio_paths: list[Path] = []
        lipsync_paths: list[Path] = []

        shots = list(self._iter_shots(script))
        total = len(shots)
        for idx, (shot, shot_image_path) in enumerate(shots, 1):
            shot_id: str = shot["shot_id"]
            logger.info("[%d/%d] Processing shot %s", idx, total, shot_id)

            # --- Image → video ---
            video_path = _OUTPUT_VIDEOS / f"{shot_id}.mp4"
            if not video_path.exists():
                video_path = await self._generate_video(shot, shot_image_path)
            if video_path:
                video_paths.append(video_path)

            # --- Voiceover ---
            audio_path: Optional[Path] = None
            if chattts_available and shot.get("dialogue"):
                audio_path = _OUTPUT_AUDIO / f"{shot_id}.wav"
                if not audio_path.exists():
                    audio_path = await self._generate_audio(shot)
                if audio_path:
                    audio_paths.append(audio_path)

            # --- Lip-sync ---
            if sadtalker_available and video_path and audio_path:
                ls_path = _OUTPUT_LIPSYNC / f"{shot_id}_lipsync.mp4"
                if not ls_path.exists():
                    ls_path = await self._apply_lipsync(video_path, audio_path, shot_id)
                if ls_path:
                    lipsync_paths.append(ls_path)

        logger.info(
            "VideoGenerator complete: %d videos, %d audio files, %d lipsync clips",
            len(video_paths),
            len(audio_paths),
            len(lipsync_paths),
        )
        return {
            "videos": video_paths,
            "audio": audio_paths,
            "lipsync": lipsync_paths,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_shots(script: dict[str, Any]):
        """Yield ``(shot, shot_image_path)`` tuples for every shot in *script*."""
        for episode in script.get("episodes", []):
            for scene in episode.get("scenes", []):
                for shot in scene.get("shots", []):
                    shot_image_path = (
                        Path("assets/shots") / f"{_slugify(shot['shot_id'])}.png"
                    )
                    yield shot, shot_image_path

    async def _generate_video(
        self, shot: dict[str, Any], image_path: Path
    ) -> Optional[Path]:
        """Submit an image-to-video request to ComfyUI (Wan2.1 node).

        Args:
            shot: Shot dictionary.
            image_path: Path to the storyboard image.

        Returns:
            Path to the saved ``.mp4``, or ``None`` on failure.
        """
        shot_id = shot["shot_id"]
        out_path = _OUTPUT_VIDEOS / f"{shot_id}.mp4"

        if not image_path.exists():
            logger.warning("Shot image not found: %s — skipping video generation", image_path)
            return None

        cfg = self._video_config.get("video", {})
        workflow = self._build_wan21_workflow(image_path, shot, cfg)

        client_id = str(uuid.uuid4())
        payload = {"prompt": workflow, "client_id": client_id}
        session = self._get_session()

        try:
            async with session.post(
                f"{self.comfyui_url}/prompt", json=payload
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                prompt_id: str = data["prompt_id"]
        except Exception as exc:  # noqa: BLE001
            logger.error("ComfyUI video submission failed for shot %s: %s", shot_id, exc)
            return None

        video_bytes = await self._poll_video(prompt_id)
        if video_bytes:
            out_path.write_bytes(video_bytes)
            logger.info("Saved video: %s", out_path)
            return out_path

        logger.error("Video generation failed for shot %s", shot_id)
        return None

    def _build_wan21_workflow(
        self, image_path: Path, shot: dict[str, Any], cfg: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a ComfyUI Wan2.1 image-to-video workflow for *shot*.

        Args:
            image_path: Path to the source storyboard image.
            shot: Shot dictionary (for duration / camera move hints).
            cfg: Video configuration dictionary.

        Returns:
            ComfyUI workflow dictionary.
        """
        image_b64 = base64.b64encode(image_path.read_bytes()).decode()
        duration = shot.get("duration", 4)
        frames = int(duration * cfg.get("fps", 8))

        return {
            "1": {
                "class_type": "LoadImageBase64",
                "inputs": {"image": image_b64},
            },
            "2": {
                "class_type": "WanVideoSampler",
                "inputs": {
                    "model": cfg.get("model", "wan2.1-14b"),
                    "image": ["1", 0],
                    "num_frames": frames,
                    "steps": cfg.get("steps", 25),
                    "cfg": cfg.get("cfg", 6.0),
                    "width": cfg.get("width", 1280),
                    "height": cfg.get("height", 720),
                    "seed": 42,
                },
            },
            "3": {
                "class_type": "VHSVideoCombine",
                "inputs": {
                    "images": ["2", 0],
                    "frame_rate": cfg.get("fps", 8),
                    "format": "video/h264-mp4",
                    "filename_prefix": f"shot_{shot['shot_id']}",
                },
            },
        }

    async def _poll_video(self, prompt_id: str) -> Optional[bytes]:
        """Poll ComfyUI history until the video for *prompt_id* is ready.

        Args:
            prompt_id: ComfyUI prompt execution ID.

        Returns:
            Raw video bytes, or ``None`` on timeout.
        """
        deadline = time.monotonic() + _POLL_TIMEOUT
        session = self._get_session()
        while time.monotonic() < deadline:
            await asyncio.sleep(_POLL_INTERVAL)
            try:
                async with session.get(
                    f"{self.comfyui_url}/history/{prompt_id}"
                ) as resp:
                    resp.raise_for_status()
                    history = await resp.json()
            except Exception as exc:  # noqa: BLE001
                logger.warning("History poll error: %s", exc)
                continue

            if prompt_id not in history:
                continue

            outputs = history[prompt_id].get("outputs", {})
            for node_output in outputs.values():
                gifs = node_output.get("gifs", [])
                if gifs:
                    info = gifs[0]
                    params = {
                        "filename": info["filename"],
                        "subfolder": info.get("subfolder", ""),
                        "type": info.get("type", "output"),
                    }
                    try:
                        async with session.get(
                            f"{self.comfyui_url}/view", params=params
                        ) as resp:
                            resp.raise_for_status()
                            return await resp.read()
                    except Exception as exc:  # noqa: BLE001
                        logger.error("Failed to download video: %s", exc)
                        return None

        logger.error("Video poll timed out after %.0fs", _POLL_TIMEOUT)
        return None

    async def _generate_audio(self, shot: dict[str, Any]) -> Optional[Path]:
        """Call ChatTTS to generate a voiceover for *shot*'s dialogue.

        Args:
            shot: Shot dictionary with ``dialogue`` and ``characters`` fields.

        Returns:
            Path to the saved ``.wav`` file, or ``None`` on failure.
        """
        shot_id = shot["shot_id"]
        dialogue = shot.get("dialogue", "").strip()
        if not dialogue:
            return None

        # Determine speaker from first character
        speaker_map = self._voice_config.get("speaker_map", {})
        characters = shot.get("characters", [])
        speaker_id = speaker_map.get(characters[0], 0) if characters else 0

        payload = {
            "text": dialogue,
            "speaker_id": speaker_id,
            "speed": self._voice_config.get("speed", 1.0),
            "temperature": self._voice_config.get("temperature", 0.3),
        }
        session = self._get_session()
        out_path = _OUTPUT_AUDIO / f"{shot_id}.wav"

        for attempt in range(1, 4):
            try:
                async with session.post(
                    f"{self.chattts_url}/generate_audio", json=payload
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    audio_b64 = data.get("audio_base64", "")
                    if audio_b64:
                        out_path.write_bytes(base64.b64decode(audio_b64))
                        logger.info("Saved audio: %s", out_path)
                        return out_path
            except Exception as exc:  # noqa: BLE001
                logger.warning("ChatTTS attempt %d failed for shot %s: %s", attempt, shot_id, exc)
                if attempt < 3:
                    await asyncio.sleep(2.0 * attempt)

        logger.error("Audio generation failed for shot %s", shot_id)
        return None

    async def _apply_lipsync(
        self, video_path: Path, audio_path: Path, shot_id: str
    ) -> Optional[Path]:
        """Call SadTalker to apply lip-sync to *video_path* using *audio_path*.

        Args:
            video_path: Path to the generated video.
            audio_path: Path to the synthesised audio.
            shot_id: Shot identifier (used to name the output file).

        Returns:
            Path to the lip-synced video, or ``None`` on failure.
        """
        out_path = _OUTPUT_LIPSYNC / f"{shot_id}_lipsync.mp4"
        session = self._get_session()

        video_b64 = base64.b64encode(video_path.read_bytes()).decode()
        audio_b64 = base64.b64encode(audio_path.read_bytes()).decode()

        payload = {
            "video_base64": video_b64,
            "audio_base64": audio_b64,
        }

        for attempt in range(1, 4):
            try:
                async with session.post(
                    f"{self.sadtalker_url}/api/lipsync", json=payload,
                    timeout=aiohttp.ClientTimeout(total=300),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    result_b64 = data.get("video_base64", "")
                    if result_b64:
                        out_path.write_bytes(base64.b64decode(result_b64))
                        logger.info("Saved lipsync video: %s", out_path)
                        return out_path
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SadTalker attempt %d failed for shot %s: %s", attempt, shot_id, exc
                )
                if attempt < 3:
                    await asyncio.sleep(3.0 * attempt)

        logger.error("Lip-sync failed for shot %s", shot_id)
        return None

    async def _check_service(self, base_url: str) -> bool:
        """Return ``True`` if the service at *base_url* is reachable."""
        session = self._get_session()
        try:
            async with session.get(
                base_url, timeout=aiohttp.ClientTimeout(total=5)
            ) as resp:
                return resp.status < 500
        except Exception:  # noqa: BLE001
            return False

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60, connect=10)
            )
        return self._session

    @staticmethod
    def _load_yaml(filename: str) -> dict[str, Any]:
        """Load a YAML config file from the ``configs/`` directory."""
        path = _CONFIGS_DIR / filename
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
