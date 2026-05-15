"""VideoGenerator Skill.

For each shot in the script this skill:

1. **Image → video**: Submits the shot's storyboard image to ComfyUI's Wan 2.2 node
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
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import aiohttp
import yaml

from utils import slugify as _slugify
from utils.assets import load_asset_config
from utils.logger import get_logger
from utils.validators import MAX_SHOT_DURATION_SECONDS

logger = get_logger(__name__)

_CONFIGS_DIR = Path(__file__).parent / "configs"

# Output directories
_OUTPUT_VIDEOS = Path("output/videos")
_OUTPUT_AUDIO = Path("output/audio")
_OUTPUT_LIPSYNC = Path("output/lipsync")
_OUTPUT_CONTINUITY = Path("output/continuity")

# Polling
_POLL_INTERVAL = 5.0
_POLL_TIMEOUT = 1800.0  # 30 min for video generation (two-stage sampling ~17 min per shot)


def _bounded_duration(shot: dict[str, Any]) -> float:
    """Return a Wan-compatible shot duration, capped at 5 seconds."""
    raw_duration = shot.get("duration", 4)
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid duration for shot %s: %r — using 4 seconds",
            shot.get("shot_id", "?"),
            raw_duration,
        )
        return 4.0

    if duration <= 0:
        logger.warning(
            "Non-positive duration for shot %s: %r — using 4 seconds",
            shot.get("shot_id", "?"),
            raw_duration,
        )
        return 4.0

    if duration > MAX_SHOT_DURATION_SECONDS:
        logger.warning(
            "Shot %s duration %.1fs exceeds Wan 2.2 limit; capping to %.1fs",
            shot.get("shot_id", "?"),
            duration,
            MAX_SHOT_DURATION_SECONDS,
        )
        shot["duration"] = MAX_SHOT_DURATION_SECONDS
        return MAX_SHOT_DURATION_SECONDS

    return duration


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
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
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
        self._asset_config = load_asset_config()
        self._progress_callback = progress_callback
        self._session: Optional[aiohttp.ClientSession] = None

        # Resolve mode-specific config
        mode = self._video_config.get("mode", "5b_single")
        if mode == "14b_two_stage":
            self._cfg = self._video_config.get("video_14b", self._video_config.get("video", {}))
        else:
            self._cfg = self._video_config.get("video_5b", self._video_config.get("video", {}))
        self._mode = mode

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
        _OUTPUT_CONTINUITY.mkdir(parents=True, exist_ok=True)

        chattts_available = await self._check_service(self.chattts_url)
        sadtalker_available = await self._check_service(self.sadtalker_url)

        if not chattts_available:
            logger.warning("ChatTTS service unavailable — voiceover will be skipped.")
        if not sadtalker_available:
            logger.warning("SadTalker service unavailable — lip-sync will be skipped.")

        video_paths: list[Path] = []
        audio_paths: list[Path] = []
        lipsync_paths: list[Path] = []

        if chattts_available and self._asset_config.get("voice", {}).get("scene_level_tts", True):
            await self._generate_scene_audio(script)

        shots = list(self._iter_shots(script))
        total = len(shots)
        previous_scene_id: Optional[str] = None
        previous_video_path: Optional[Path] = None
        for idx, (shot, shot_image_path, scene) in enumerate(shots, 1):
            shot_id: str = shot["shot_id"]
            _bounded_duration(shot)
            scene_id = str(scene.get("scene_id", ""))
            logger.info("[%d/%d] Processing shot %s", idx, total, shot_id)
            self._report(f"Processing shot {shot_id}", idx, total)

            # --- Image → video ---
            video_path = _OUTPUT_VIDEOS / f"{shot_id}.mp4"
            if not video_path.exists():
                continuity_image = None
                if (
                    self._asset_config.get("continuity", {}).get("use_previous_tail_frame", True)
                    and previous_scene_id == scene_id
                    and previous_video_path
                    and previous_video_path.exists()
                ):
                    continuity_image = await self._extract_tail_frame(previous_video_path, shot_id)
                video_path = await self._generate_video(shot, shot_image_path, continuity_image)
            if video_path:
                video_paths.append(video_path)
                previous_video_path = video_path
                previous_scene_id = scene_id

            # --- Voiceover ---
            audio_path: Optional[Path] = _OUTPUT_AUDIO / f"{shot_id}.wav"
            if not audio_path.exists():
                audio_path = None
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
        """Yield ``(shot, shot_image_path, scene)`` tuples for every shot."""
        for episode in script.get("episodes", []):
            for scene in episode.get("scenes", []):
                for shot in scene.get("shots", []):
                    shot_image_path = (
                        Path("assets/shots") / f"{_slugify(shot['shot_id'])}.png"
                    )
                    yield shot, shot_image_path, scene

    async def _generate_video(
        self,
        shot: dict[str, Any],
        image_path: Path,
        continuity_image_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """Submit an image-to-video request to ComfyUI (Wan 2.2 node).

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

        start_image = continuity_image_path if continuity_image_path and continuity_image_path.exists() else image_path

        # Upload start image to ComfyUI
        comfyui_image_name = await self._upload_image(start_image)
        if not comfyui_image_name:
            logger.error("Failed to upload image for shot %s", shot_id)
            return None

        cfg = self._cfg
        if self._mode == "14b_two_stage":
            workflow = self._build_wan22_workflow(comfyui_image_name, shot, cfg)
        else:
            workflow = self._build_wan22_5b_workflow(comfyui_image_name, shot, cfg)

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

    async def _extract_tail_frame(self, video_path: Path, shot_id: str) -> Optional[Path]:
        """Extract the last frame of a previous clip for continuity."""
        out_path = _OUTPUT_CONTINUITY / f"{shot_id}_start.png"
        if out_path.exists():
            return out_path
        cmd = [
            "ffmpeg",
            "-y",
            "-sseof",
            "-0.1",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(out_path),
        ]
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: subprocess.run(cmd, capture_output=True, text=True),
            )
            if result.returncode == 0 and out_path.exists():
                logger.info("Extracted continuity frame: %s", out_path)
                return out_path
            logger.warning("Tail-frame extraction failed for %s: %s", video_path, result.stderr[-300:])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tail-frame extraction failed for %s: %s", video_path, exc)
        return None

    async def _upload_image(self, image_path: Path) -> Optional[str]:
        """Upload a local image to ComfyUI's ``/upload/image`` endpoint.

        Returns:
            The filename assigned by ComfyUI, or ``None`` on failure.
        """
        session = self._get_session()
        try:
            data = aiohttp.FormData()
            data.add_field(
                "image",
                open(image_path, "rb"),
                filename=image_path.name,
                content_type="image/png",
            )
            async with session.post(
                f"{self.comfyui_url}/upload/image", data=data,
            ) as resp:
                resp.raise_for_status()
                result = await resp.json()
                return result.get("name")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to upload image %s: %s", image_path, exc)
            return None

    def _build_wan22_workflow(
        self, image_name: str, shot: dict[str, Any], cfg: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a ComfyUI Wan 2.2 two-stage I2V workflow with LoRA acceleration.

        Pipeline:
        - Stage 1 (high_noise): UNETLoader → ModelSampling → LoRA → KSamplerAdvanced [0→2]
        - Stage 2 (low_noise):  UNETLoader → ModelSampling → LoRA → KSamplerAdvanced [2→4]
        - VAEDecode → CreateVideo → SaveVideo

        Args:
            image_name: ComfyUI filename of the uploaded start frame.
            shot: Shot dictionary (for duration / camera move hints).
            cfg: Video configuration dictionary.

        Returns:
            ComfyUI workflow dictionary.
        """
        duration = _bounded_duration(shot)
        fps = cfg.get("fps", 8)
        frames = int(duration * fps)
        frames = max(1, ((frames - 1) // 4) * 4 + 1)

        prompt_text = shot.get(
            "motion_prompt",
            shot.get("visual_description", shot.get("description", "")),
        )
        steps = cfg.get("steps", 4)
        half = steps // 2
        shift = cfg.get("shift", 5.0)
        sampler = cfg.get("sampler_name", "euler")
        scheduler = cfg.get("scheduler", "simple")
        cfg_val = cfg.get("cfg", 1.0)

        return {
            # --- Load high noise model + LoRA ---
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": cfg.get("model", "wan2.2_i2v_high_noise_14B_fp16.safetensors"),
                    "weight_dtype": "default",
                },
            },
            "12": {
                "class_type": "ModelSamplingSD3",
                "inputs": {
                    "model": ["1", 0],
                    "shift": shift,
                },
            },
            "13": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["12", 0],
                    "lora_name": cfg.get("lora_high_noise", "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"),
                    "strength_model": 1.0,
                },
            },
            # --- Load low noise model + LoRA ---
            "14": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": cfg.get("model_low_noise", "wan2.2_i2v_low_noise_14B_fp16.safetensors"),
                    "weight_dtype": "default",
                },
            },
            "15": {
                "class_type": "ModelSamplingSD3",
                "inputs": {
                    "model": ["14", 0],
                    "shift": shift,
                },
            },
            "16": {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ["15", 0],
                    "lora_name": cfg.get("lora_low_noise", "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"),
                    "strength_model": 1.0,
                },
            },
            # --- Text encoder + VAE ---
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": cfg.get("text_encoder", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
                    "type": "wan",
                    "device": "default",
                },
            },
            "3": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": cfg.get("vae", "wan2.1_vae.safetensors"),
                },
            },
            # --- Text conditioning ---
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt_text,
                    "clip": ["2", 0],
                },
            },
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "",
                    "clip": ["2", 0],
                },
            },
            # --- Load start frame ---
            "6": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": image_name,
                },
            },
            # --- WanImageToVideo: encodes image into conditioning + creates latent ---
            "7": {
                "class_type": "WanImageToVideo",
                "inputs": {
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "vae": ["3", 0],
                    "width": cfg.get("width", 1280),
                    "height": cfg.get("height", 704),
                    "length": frames,
                    "batch_size": 1,
                    "start_image": ["6", 0],
                },
            },
            # --- Stage 1: High noise sampling [0 → half] ---
            "8": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": ["13", 0],
                    "add_noise": "enable",
                    "noise_seed": 0,
                    "steps": steps,
                    "cfg": cfg_val,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "positive": ["7", 0],
                    "negative": ["7", 1],
                    "latent_image": ["7", 2],
                    "start_at_step": 0,
                    "end_at_step": half,
                    "return_with_leftover_noise": "enable",
                },
            },
            # --- Stage 2: Low noise sampling [half → steps] ---
            "9": {
                "class_type": "KSamplerAdvanced",
                "inputs": {
                    "model": ["16", 0],
                    "add_noise": "disable",
                    "noise_seed": 0,
                    "steps": steps,
                    "cfg": cfg_val,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "positive": ["7", 0],
                    "negative": ["7", 1],
                    "latent_image": ["8", 0],
                    "start_at_step": half,
                    "end_at_step": steps,
                    "return_with_leftover_noise": "disable",
                },
            },
            # --- Decode ---
            "10": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["9", 0],
                    "vae": ["3", 0],
                },
            },
            # --- Save video ---
            "11": {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": ["10", 0],
                    "fps": fps,
                },
            },
            "17": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["11", 0],
                    "filename_prefix": f"shot_{shot['shot_id']}",
                    "format": "mp4",
                    "codec": "h264",
                },
            },
        }

    def _build_wan22_5b_workflow(
        self, image_name: str, shot: dict[str, Any], cfg: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a ComfyUI Wan 2.2 5B TI2V single-stage workflow.

        Based on the official ComfyUI template for wan2.2_ti2v_5B_fp16.
        Uses Wan22ImageToVideoLatent (48-channel latent) + single KSampler.

        Args:
            image_name: ComfyUI filename of the uploaded start frame.
            shot: Shot dictionary (for duration / camera move hints).
            cfg: Video configuration dictionary.

        Returns:
            ComfyUI workflow dictionary.
        """
        duration = _bounded_duration(shot)
        fps = cfg.get("fps", 24)
        frames = int(duration * fps)
        # Wan 2.2 requires length ≡ 1 (mod 4)
        frames = max(1, ((frames - 1) // 4) * 4 + 1)

        prompt_text = shot.get(
            "motion_prompt",
            shot.get("visual_description", shot.get("description", "")),
        )
        steps = cfg.get("steps", 20)
        shift = cfg.get("shift", 8.0)
        sampler = cfg.get("sampler_name", "uni_pc")
        scheduler = cfg.get("scheduler", "simple")
        cfg_val = cfg.get("cfg", 5.0)

        return {
            # --- Load model ---
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": cfg.get("model", "wan2.2_ti2v_5B_fp16.safetensors"),
                    "weight_dtype": "default",
                },
            },
            # --- ModelSamplingSD3 ---
            "2": {
                "class_type": "ModelSamplingSD3",
                "inputs": {
                    "model": ["1", 0],
                    "shift": shift,
                },
            },
            # --- Text encoder ---
            "3": {
                "class_type": "CLIPLoader",
                "inputs": {
                    "clip_name": cfg.get("text_encoder", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
                    "type": "wan",
                    "device": "default",
                },
            },
            # --- VAE ---
            "4": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": cfg.get("vae", "wan2.2_vae.safetensors"),
                },
            },
            # --- Positive prompt ---
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt_text,
                    "clip": ["3", 0],
                },
            },
            # --- Negative prompt ---
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "",
                    "clip": ["3", 0],
                },
            },
            # --- Load start frame ---
            "7": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": image_name,
                },
            },
            # --- Wan22ImageToVideoLatent: 48-channel latent with image conditioning ---
            "8": {
                "class_type": "Wan22ImageToVideoLatent",
                "inputs": {
                    "vae": ["4", 0],
                    "start_image": ["7", 0],
                    "width": cfg.get("width", 848),
                    "height": cfg.get("height", 480),
                    "length": frames,
                    "batch_size": 1,
                },
            },
            # --- KSampler (single stage) ---
            "9": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["2", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["8", 0],
                    "seed": 0,
                    "steps": steps,
                    "cfg": cfg_val,
                    "sampler_name": sampler,
                    "scheduler": scheduler,
                    "denoise": 1.0,
                },
            },
            # --- Decode ---
            "10": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["9", 0],
                    "vae": ["4", 0],
                },
            },
            # --- Create video ---
            "11": {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": ["10", 0],
                    "fps": fps,
                },
            },
            # --- Save video ---
            "12": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["11", 0],
                    "filename_prefix": f"shot_{shot['shot_id']}",
                    "format": "mp4",
                    "codec": "h264",
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
                # SaveVideo uses "videos", VHSVideoCombine uses "gifs"
                gifs = node_output.get("videos", node_output.get("gifs", []))
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

    async def _generate_scene_audio(self, script: dict[str, Any]) -> None:
        """Generate scene-level audio files for review; shot TTS remains fallback."""
        if not self._asset_config.get("voice", {}).get("scene_level_tts", True):
            return
        for episode in script.get("episodes", []):
            for scene in episode.get("scenes", []):
                lines = [
                    str(shot.get("dialogue", "")).strip()
                    for shot in scene.get("shots", [])
                    if str(shot.get("dialogue", "")).strip()
                ]
                if not lines:
                    continue
                scene_id = str(scene.get("scene_id", "scene"))
                out_path = _OUTPUT_AUDIO / f"{_slugify(scene_id)}_scene.wav"
                if out_path.exists():
                    continue
                pseudo_shot = {
                    "shot_id": f"{_slugify(scene_id)}_scene",
                    "dialogue": "\n".join(lines),
                    "characters": [],
                }
                await self._generate_audio(pseudo_shot)

    def _report(self, message: str, current: int, total: int) -> None:
        if self._progress_callback:
            self._progress_callback(message, current, total)

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
