"""Editor Skill.

Assembles all generated video clips into a final polished episode video.

Pipeline:
1. Read the script JSON to determine shot order.
2. Locate each shot's video (prefer lipsync version when available).
3. Merge shot audio (dialogue + background music) via FFmpeg.
4. Apply configurable transitions between shots (crossfade, fade_black, cut).
5. Generate subtitles from the script dialogue (ASS format) and burn them in.
6. Add title card and end card.
7. Export the final ``output/final/<project_id>_ep<N>.mp4``.
"""

from __future__ import annotations

import asyncio
import gc
import json
import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import yaml

from utils import slugify as _slugify
from utils.logger import get_logger
from utils.paths import PROJECT_ROOT

logger = get_logger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_OUTPUT_FINAL = PROJECT_ROOT / "output" / "final"
_OUTPUT_VIDEOS = PROJECT_ROOT / "output" / "videos"
_OUTPUT_AUDIO = PROJECT_ROOT / "output" / "audio"
_OUTPUT_LIPSYNC = PROJECT_ROOT / "output" / "lipsync"
_ASSETS_SHOTS = PROJECT_ROOT / "assets" / "shots"

# Default FFmpeg subtitle burn-in filter
_ASS_STYLE_TEMPLATE = _TEMPLATES_DIR / "subtitle_style.ass"


def _run_ffmpeg(args: list[str], description: str = "") -> bool:
    """Run an FFmpeg command and return ``True`` on success."""
    cmd = ["ffmpeg", "-y"] + args
    logger.debug("FFmpeg %s: %s", description, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("FFmpeg failed (%s): %s", description, result.stderr[-500:])
        return False
    return True


def _probe_duration(path: Path) -> float:
    """Probe video/audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.warning("ffprobe failed for %s: %s", path, result.stderr[-200:])
            return 0.0
        data = json.loads(result.stdout or "{}")
        return float(data.get("format", {}).get("duration", 0))
    except Exception as exc:  # noqa: BLE001
        logger.warning("ffprobe error for %s: %s", path, exc)
        return 0.0


class Editor:
    """Assembles shot videos into a final episode using FFmpeg."""

    def __init__(self, project_id: str = "project") -> None:
        self.project_id = project_id
        self._transitions = self._load_transitions()
        self._bgm_config = self._load_bgm_config()
        self._whisper_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def edit(self, script: dict[str, Any]) -> list[Path]:
        """Build the final video for every episode in *script*.

        Returns:
            List of paths to final episode videos.
        """
        _OUTPUT_FINAL.mkdir(parents=True, exist_ok=True)
        final_paths: list[Path] = []
        for episode in script.get("episodes", []):
            ep_num = episode.get("episode", 1)
            path = await self._edit_episode(script, episode, ep_num)
            if path:
                final_paths.append(path)
        return final_paths

    # ------------------------------------------------------------------
    # Episode-level editing
    # ------------------------------------------------------------------

    async def _edit_episode(
        self, script: dict[str, Any], episode: dict[str, Any], ep_num: int
    ) -> Optional[Path]:
        out_path = _OUTPUT_FINAL / f"{self.project_id}_ep{ep_num:02d}.mp4"
        if out_path.exists():
            logger.info("Episode already exists, skipping: %s", out_path)
            return out_path

        shots = list(self._iter_shots(episode))
        if not shots:
            logger.warning("Episode %d has no shots — skipping", ep_num)
            return None

        temp_files: list[Path] = []

        clip_paths: list[Path] = []
        transition_names: list[str] = []
        for shot in shots:
            clip = self._find_clip(shot)
            if clip:
                clip_paths.append(clip)
                t_name = shot.get("transition", self._transitions.get("default", "crossfade"))
                transition_names.append(t_name)
            else:
                logger.warning("No video found for shot %s — skipping", shot.get("shot_id"))

        if not clip_paths:
            logger.error("No clips found for episode %d", ep_num)
            return None

        # Concatenate clips with xfade transitions
        concat_path = await self._concat_clips(clip_paths, ep_num, transition_names)
        if not concat_path:
            return None

        source = concat_path
        temp_files.append(concat_path)

        # Mix background music
        bgm_path = self._select_bgm(episode)
        if bgm_path and bgm_path.exists():
            mixed_path = Path(tempfile.gettempdir()) / f"mixed_ep{ep_num:02d}.mp4"
            if await self._mix_bgm(source, bgm_path, mixed_path):
                source = mixed_path
                temp_files.append(mixed_path)

        # Whisper-based subtitle timing (optional)
        whisper_segments: list[dict[str, Any]] | None = None
        try:
            audio_wav = Path(tempfile.gettempdir()) / f"whisper_ep{ep_num:02d}.wav"
            extract_ok = await asyncio.get_running_loop().run_in_executor(
                None,
                _run_ffmpeg,
                ["-i", str(source), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio_wav)],
                "extract audio for whisper",
            )
            if extract_ok and audio_wav.exists():
                whisper_segments = await self._transcribe_with_whisper(audio_wav)
                audio_wav.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Whisper transcription failed, falling back to duration estimation: %s", exc)

        # Generate subtitle file
        ass_path = await self._generate_subtitles(shots, ep_num, whisper_segments)

        # Add title card
        title_path = await self._add_title_card(
            source, title=script.get("title", ""), ep_num=ep_num,
        )
        if title_path:
            source = title_path
            temp_files.append(title_path)

        # Burn subtitles
        if ass_path and ass_path.exists():
            subtitled = await self._burn_subtitles(source, ass_path, ep_num)
            if subtitled:
                source = subtitled
                temp_files.append(subtitled)

        # Copy to final path
        if source != out_path:
            shutil.copy2(source, out_path)
            logger.info("Episode %d saved: %s", ep_num, out_path)

        # Cleanup temp files
        for tmp in temp_files:
            try:
                if tmp.exists() and tmp != out_path:
                    tmp.unlink()
            except OSError:
                pass

        return out_path

    # ------------------------------------------------------------------
    # Subtitle generation
    # ------------------------------------------------------------------

    async def _generate_subtitles(
        self,
        shots: list[dict[str, Any]],
        ep_num: int,
        whisper_segments: list[dict[str, Any]] | None = None,
    ) -> Optional[Path]:
        """Generate an ASS subtitle file from shot dialogue.

        If *whisper_segments* is provided, uses actual speech timestamps.
        Otherwise falls back to duration estimation.
        """
        ass_path = Path(tempfile.gettempdir()) / f"subs_ep{ep_num:02d}.ass"
        style = self._load_subtitle_style()
        events: list[str] = []

        if whisper_segments:
            # Map whisper segments to shots by text overlap
            for shot in shots:
                dialogue = shot.get("dialogue", "").strip()
                if not dialogue:
                    continue
                text = re.sub(r"^[^：:]+[：:]", "", dialogue).strip()
                if not text:
                    continue
                # Find best matching whisper segment
                best_seg = None
                for seg in whisper_segments:
                    seg_text = str(seg.get("text", "")).strip()
                    if text in seg_text or seg_text in text:
                        best_seg = seg
                        break
                if best_seg:
                    start = _seconds_to_ass(float(best_seg.get("start", 0)))
                    end = _seconds_to_ass(float(best_seg.get("end", 0)))
                else:
                    # Fallback: use duration estimation for this shot
                    duration = float(shot.get("duration", 4))
                    idx = shots.index(shot)
                    cursor = sum(float(s.get("duration", 4)) for s in shots[:idx])
                    start = _seconds_to_ass(cursor)
                    end = _seconds_to_ass(cursor + duration)
                events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
        else:
            # Duration-based estimation (original logic)
            cursor = 0.0
            for shot in shots:
                duration = float(shot.get("duration", 4))
                dialogue = shot.get("dialogue", "").strip()
                if dialogue:
                    text = re.sub(r"^[^：:]+[：:]", "", dialogue).strip()
                    if text:
                        start = _seconds_to_ass(cursor)
                        end = _seconds_to_ass(cursor + duration)
                        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
                cursor += duration

        if not events:
            return None

        ass_content = f"{style}\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        ass_content += "\n".join(events) + "\n"
        ass_path.write_text(ass_content, encoding="utf-8")
        return ass_path

    # ------------------------------------------------------------------
    # Whisper transcription
    # ------------------------------------------------------------------

    async def _transcribe_with_whisper(self, audio_path: Path) -> list[dict[str, Any]] | None:
        """Transcribe audio using Whisper and return segments with timestamps.

        Caches the loaded model for reuse across multiple episodes.
        """
        def _do_transcribe() -> list[dict[str, Any]] | None:
            try:
                import whisper  # type: ignore[import-untyped]
            except ImportError:
                logger.warning("openai-whisper not installed, skipping transcription")
                return None
            try:
                if self._whisper_model is None:
                    self._whisper_model = whisper.load_model("base")
                result = self._whisper_model.transcribe(str(audio_path), word_timestamps=False)
                segments = []
                for seg in result.get("segments", []):
                    segments.append({
                        "start": float(seg.get("start", 0)),
                        "end": float(seg.get("end", 0)),
                        "text": str(seg.get("text", "")).strip(),
                    })
                return segments
            except Exception as exc:  # noqa: BLE001
                logger.warning("Whisper transcription error: %s", exc)
                return None

        return await asyncio.get_running_loop().run_in_executor(None, _do_transcribe)

    def unload_whisper(self) -> None:
        """Release the cached Whisper model to free memory."""
        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
            gc.collect()

    # ------------------------------------------------------------------
    # Background music
    # ------------------------------------------------------------------

    @staticmethod
    def _load_bgm_config() -> dict[str, Any]:
        path = Path("configs/assets.yaml")
        try:
            with path.open(encoding="utf-8") as fh:
                config = yaml.safe_load(fh) or {}
            return config.get("bgm", {})
        except Exception:  # noqa: BLE001
            return {}

    def _select_bgm(self, episode: dict[str, Any]) -> Optional[Path]:
        """Select a BGM track based on the dominant mood of the episode."""
        if not self._bgm_config.get("enabled"):
            return None
        bgm_dir = Path(self._bgm_config.get("directory", "assets/bgm"))
        mood_mapping: dict[str, str] = self._bgm_config.get("mood_mapping", {})
        default_track = self._bgm_config.get("default_track", "default.mp3")

        # Determine dominant mood from shots
        moods: list[str] = []
        for shot in self._iter_shots(episode):
            mood = shot.get("emotion") or shot.get("mood", "")
            if mood:
                moods.append(mood.lower())
        dominant_mood = Counter(moods).most_common(1)[0][0] if moods else "neutral"

        track_name = mood_mapping.get(dominant_mood, default_track)
        track_path = bgm_dir / track_name
        if track_path.exists():
            return track_path
        fallback = bgm_dir / default_track
        return fallback if fallback.exists() else None

    async def _mix_bgm(
        self, dialogue_path: Path, bgm_path: Path, output_path: Path, volume: float = 0.25
    ) -> bool:
        """Mix dialogue video with background music using FFmpeg amix."""
        fade_in = float(self._bgm_config.get("fade_in", 1.0))
        fade_out = float(self._bgm_config.get("fade_out", 2.0))
        duration = _probe_duration(dialogue_path)
        if duration <= 0:
            return False
        fade_out_start = max(0, duration - fade_out)

        filter_complex = (
            f"[1:a]volume={volume},"
            f"afade=t=in:d={fade_in},"
            f"afade=t=out:st={fade_out_start}:d={fade_out}[bgm];"
            f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
        )

        args = [
            "-i", str(dialogue_path),
            "-i", str(bgm_path),
            "-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[aout]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ]
        return await asyncio.get_running_loop().run_in_executor(
            None, _run_ffmpeg, args, "mix bgm"
        )

    # ------------------------------------------------------------------
    # FFmpeg operations
    # ------------------------------------------------------------------

    async def _concat_clips(
        self,
        clip_paths: list[Path],
        ep_num: int,
        transition_names: list[str] | None = None,
    ) -> Optional[Path]:
        """Concatenate clips with xfade transitions.

        Falls back to simple concat if transitions config is empty.
        Single clip is returned directly.
        """
        out = Path(tempfile.gettempdir()) / f"concat_ep{ep_num:02d}.mp4"

        if len(clip_paths) == 1:
            shutil.copy2(clip_paths[0], out)
            return out

        # Fallback to simple concat if no transitions loaded
        if not self._transitions:
            return await self._concat_clips_simple(clip_paths, ep_num)

        # Probe durations
        durations = []
        for p in clip_paths:
            d = _probe_duration(p)
            if d <= 0:
                logger.warning("Could not probe duration for %s, using 4s fallback", p)
                d = 4.0
            durations.append(d)

        # Resolve transition configs
        t_configs = self._transitions.get("transitions", {})
        default_t = self._transitions.get("default", "crossfade")
        if transition_names is None:
            transition_names = [default_t] * (len(clip_paths) - 1)

        # Check if all transitions are "cut" — use simple concat
        all_cut = all(
            t_configs.get(t, {}).get("filter") is None or t_configs.get(t, {}).get("duration", 0) == 0
            for t in transition_names
        )
        if all_cut:
            return await self._concat_clips_simple(clip_paths, ep_num)

        # Build filter_complex
        n = len(clip_paths)
        inputs: list[str] = []
        for p in clip_paths:
            inputs.extend(["-i", str(p)])

        video_filters: list[str] = []
        audio_filters: list[str] = []
        v_label = "[0:v]"
        a_label = "[0:a]"
        cumulative_duration = durations[0]

        for i in range(n - 1):
            t_name = transition_names[i] if i < len(transition_names) else default_t
            t_conf = t_configs.get(t_name, {})
            t_filter = t_conf.get("filter")
            t_dur = float(t_conf.get("duration", 0))

            # Clamp transition duration to half the shorter clip
            max_dur = min(durations[i], durations[i + 1]) / 2
            t_dur = min(t_dur, max_dur) if max_dur > 0 else 0

            in_v = f"[{i + 1}:v]"
            in_a = f"[{i + 1}:a]"
            out_v = f"[v{i}]"
            out_a = f"[a{i}]"

            if t_filter and t_dur > 0:
                offset = cumulative_duration - t_dur
                if offset < 0:
                    offset = 0
                transition_type = t_conf.get("transition", "fade")
                video_filters.append(
                    f"{v_label}{in_v}xfade=transition={transition_type}"
                    f":duration={t_dur}:offset={offset}{out_v}"
                )
                audio_filters.append(
                    f"{a_label}{in_a}acrossfade=d={t_dur}{out_a}"
                )
                cumulative_duration += durations[i + 1] - t_dur
            else:
                # Hard cut — no overlap
                cumulative_duration += durations[i + 1]
                # Use concat filter for hard cuts
                out_v = f"[v{i}]"
                out_a = f"[a{i}]"
                video_filters.append(f"{v_label}{in_v}concat=n=2:v=1:a=0{out_v}")
                audio_filters.append(f"{a_label}{in_a}concat=n=2:v=0:a=1{out_a}")

            v_label = out_v
            a_label = out_a

        filter_complex = ";".join(video_filters + audio_filters)

        args = inputs + [
            "-filter_complex", filter_complex,
            "-map", v_label, "-map", a_label,
            "-c:v", "libx264", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            str(out),
        ]
        success = await asyncio.get_running_loop().run_in_executor(
            None, _run_ffmpeg, args, f"xfade concat ep{ep_num}"
        )
        return out if success else None

    async def _concat_clips_simple(
        self, clip_paths: list[Path], ep_num: int
    ) -> Optional[Path]:
        """Simple concat demuxer (no transitions). Used as fallback."""
        out = Path(tempfile.gettempdir()) / f"concat_ep{ep_num:02d}.mp4"
        list_file = Path(tempfile.gettempdir()) / f"concat_list_ep{ep_num:02d}.txt"
        with list_file.open("w", encoding="utf-8") as fh:
            for p in clip_paths:
                fh.write(f"file '{p.resolve()}'\n")
        args = [
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out),
        ]
        success = await asyncio.get_running_loop().run_in_executor(
            None, _run_ffmpeg, args, f"simple concat ep{ep_num}"
        )
        return out if success else None

    async def _add_title_card(
        self, source: Path, title: str, ep_num: int
    ) -> Optional[Path]:
        if not title:
            return None
        out = Path(tempfile.gettempdir()) / f"titled_ep{ep_num:02d}.mp4"
        escaped_title = title.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
        title_text = f"{escaped_title} — Episode {ep_num}"
        title_clip = Path(tempfile.gettempdir()) / f"title_card_ep{ep_num:02d}.mp4"
        args_title = [
            "-f", "lavfi", "-i", "color=c=black:size=1280x720:rate=24:duration=3",
            "-vf",
            f"drawtext=text='{title_text}':fontsize=60:fontcolor=white:x=(w-text_w)/2:y=(h-text_h)/2",
            "-c:v", "libx264", "-crf", "18",
            str(title_clip),
        ]
        ok = await asyncio.get_running_loop().run_in_executor(
            None, _run_ffmpeg, args_title, "title card"
        )
        if not ok:
            return None
        list_file = Path(tempfile.gettempdir()) / f"title_concat_ep{ep_num:02d}.txt"
        with list_file.open("w", encoding="utf-8") as fh:
            fh.write(f"file '{title_clip.resolve()}'\n")
            fh.write(f"file '{source.resolve()}'\n")
        args_concat = [
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            str(out),
        ]
        ok = await asyncio.get_running_loop().run_in_executor(
            None, _run_ffmpeg, args_concat, "title concat"
        )
        return out if ok else None

    async def _burn_subtitles(
        self, source: Path, ass_path: Path, ep_num: int
    ) -> Optional[Path]:
        out = Path(tempfile.gettempdir()) / f"subtitled_ep{ep_num:02d}.mp4"
        escaped_ass = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
        args = [
            "-i", str(source),
            "-vf", f"ass={escaped_ass}",
            "-c:a", "copy",
            str(out),
        ]
        ok = await asyncio.get_running_loop().run_in_executor(
            None, _run_ffmpeg, args, f"burn subtitles ep{ep_num}"
        )
        return out if ok else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_shots(episode: dict[str, Any]):
        for scene in episode.get("scenes", []):
            for shot in scene.get("shots", []):
                yield shot

    @staticmethod
    def _find_clip(shot: dict[str, Any]) -> Optional[Path]:
        shot_id = shot.get("shot_id", "")
        lipsync = _OUTPUT_LIPSYNC / f"{shot_id}_lipsync.mp4"
        video = _OUTPUT_VIDEOS / f"{shot_id}.mp4"
        if lipsync.exists():
            return lipsync
        if video.exists():
            return video
        return None

    @staticmethod
    def _load_transitions() -> dict[str, Any]:
        path = _TEMPLATES_DIR / "transitions.yaml"
        try:
            with path.open(encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except Exception:  # noqa: BLE001
            return {}

    @staticmethod
    def _load_subtitle_style() -> str:
        try:
            return _ASS_STYLE_TEMPLATE.read_text(encoding="utf-8")
        except FileNotFoundError:
            return (
                "[Script Info]\nScriptType: v4.00+\n\n"
                "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, "
                "SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, "
                "StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
                "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
                "Style: Default,Arial,36,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,"
                "0,0,0,0,100,100,0,0,1,2,1,2,10,10,20,1\n"
            )


def _seconds_to_ass(seconds: float) -> str:
    """Convert seconds to ASS timecode ``H:MM:SS.cc``."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    cs = int((s % 1) * 100)
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"
