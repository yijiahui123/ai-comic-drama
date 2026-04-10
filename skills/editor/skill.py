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
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import yaml

from utils import slugify as _slugify
from utils.logger import get_logger

logger = get_logger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_OUTPUT_FINAL = Path("output/final")
_OUTPUT_VIDEOS = Path("output/videos")
_OUTPUT_AUDIO = Path("output/audio")
_OUTPUT_LIPSYNC = Path("output/lipsync")
_ASSETS_SHOTS = Path("assets/shots")

# Default FFmpeg subtitle burn-in filter
_ASS_STYLE_TEMPLATE = _TEMPLATES_DIR / "subtitle_style.ass"


def _run_ffmpeg(args: list[str], description: str = "") -> bool:
    """Run an FFmpeg command and return ``True`` on success.

    Args:
        args: Full argument list (without leading ``ffmpeg``).
        description: Human-readable description for log messages.

    Returns:
        ``True`` if FFmpeg exits with code 0, ``False`` otherwise.
    """
    cmd = ["ffmpeg", "-y"] + args
    logger.debug("FFmpeg %s: %s", description, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("FFmpeg failed (%s): %s", description, result.stderr[-500:])
        return False
    return True


class Editor:
    """Assembles shot videos into a final episode using FFmpeg.

    Attributes:
        project_id: Unique identifier for the current pipeline run.
    """

    def __init__(self, project_id: str = "project") -> None:
        """
        Args:
            project_id: Identifier used to name the output file.
        """
        self.project_id = project_id
        self._transitions = self._load_transitions()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def edit(self, script: dict[str, Any]) -> Optional[Path]:
        """Build the final video for every episode in *script*.

        Args:
            script: Validated script dictionary.

        Returns:
            Path to the last exported episode file, or ``None`` if no clips
            could be assembled.
        """
        _OUTPUT_FINAL.mkdir(parents=True, exist_ok=True)

        final_path: Optional[Path] = None
        for episode in script.get("episodes", []):
            ep_num = episode.get("episode", 1)
            path = await self._edit_episode(script, episode, ep_num)
            if path:
                final_path = path

        return final_path

    # ------------------------------------------------------------------
    # Episode-level editing
    # ------------------------------------------------------------------

    async def _edit_episode(
        self, script: dict[str, Any], episode: dict[str, Any], ep_num: int
    ) -> Optional[Path]:
        """Edit a single episode and export it to disk.

        Args:
            script: Full script (for title / style).
            episode: Episode dictionary.
            ep_num: Episode number.

        Returns:
            Path to the exported ``.mp4``, or ``None`` on failure.
        """
        out_path = _OUTPUT_FINAL / f"{self.project_id}_ep{ep_num:02d}.mp4"
        if out_path.exists():
            logger.info("Episode already exists, skipping: %s", out_path)
            return out_path

        shots = list(self._iter_shots(episode))
        if not shots:
            logger.warning("Episode %d has no shots — skipping", ep_num)
            return None

        # Collect clip paths
        clip_paths: list[Path] = []
        for shot in shots:
            clip = self._find_clip(shot)
            if clip:
                clip_paths.append(clip)
            else:
                logger.warning("No video found for shot %s — skipping", shot.get("shot_id"))

        if not clip_paths:
            logger.error("No clips found for episode %d", ep_num)
            return None

        # Generate subtitle file
        ass_path = await self._generate_subtitles(shots, ep_num)

        # Concatenate clips with transitions
        concat_path = await self._concat_clips(clip_paths, ep_num)
        if not concat_path:
            return None

        # Add title card
        title_path = await self._add_title_card(
            concat_path,
            title=script.get("title", ""),
            ep_num=ep_num,
        )
        source = title_path or concat_path

        # Burn subtitles
        if ass_path and ass_path.exists():
            subtitled = await self._burn_subtitles(source, ass_path, ep_num)
            source = subtitled or source

        # Rename/copy to final path
        if source != out_path:
            out_path.write_bytes(source.read_bytes())
            logger.info("Episode %d saved: %s", ep_num, out_path)

        return out_path

    # ------------------------------------------------------------------
    # Subtitle generation
    # ------------------------------------------------------------------

    async def _generate_subtitles(
        self, shots: list[dict[str, Any]], ep_num: int
    ) -> Optional[Path]:
        """Generate an ASS subtitle file from shot dialogue.

        Timecodes are estimated from the ``duration`` field of each shot.

        Args:
            shots: Ordered list of shot dictionaries.
            ep_num: Episode number (used to name the file).

        Returns:
            Path to the generated ``.ass`` file, or ``None`` if no dialogue.
        """
        ass_path = Path(tempfile.gettempdir()) / f"subs_ep{ep_num:02d}.ass"
        style = self._load_subtitle_style()
        events: list[str] = []

        cursor = 0.0
        for shot in shots:
            duration = float(shot.get("duration", 4))
            dialogue = shot.get("dialogue", "").strip()
            if dialogue:
                # Remove character name prefix (e.g. "凯：...")
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
    # FFmpeg operations (run in thread pool to stay async)
    # ------------------------------------------------------------------

    async def _concat_clips(
        self, clip_paths: list[Path], ep_num: int
    ) -> Optional[Path]:
        """Concatenate *clip_paths* into a single video using FFmpeg concat.

        Args:
            clip_paths: Ordered list of input clip paths.
            ep_num: Episode number (for naming temp files).

        Returns:
            Path to the concatenated video, or ``None`` on failure.
        """
        out = Path(tempfile.gettempdir()) / f"concat_ep{ep_num:02d}.mp4"

        # Write concat list file
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
        success = await asyncio.get_event_loop().run_in_executor(
            None, _run_ffmpeg, args, f"concat ep{ep_num}"
        )
        return out if success else None

    async def _add_title_card(
        self, source: Path, title: str, ep_num: int
    ) -> Optional[Path]:
        """Prepend a title card to *source* using FFmpeg drawtext.

        Args:
            source: Input video path.
            title: Script title string.
            ep_num: Episode number.

        Returns:
            Path to the video with title card prepended, or ``None`` on failure.
        """
        if not title:
            return None

        out = Path(tempfile.gettempdir()) / f"titled_ep{ep_num:02d}.mp4"
        # Escape single quotes and backslashes for FFmpeg drawtext
        escaped_title = title.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
        title_text = f"{escaped_title} — Episode {ep_num}"

        # Generate 3-second black title card with text
        title_clip = Path(tempfile.gettempdir()) / f"title_card_ep{ep_num:02d}.mp4"
        args_title = [
            "-f", "lavfi", "-i", "color=c=black:size=1280x720:rate=24:duration=3",
            "-vf",
            (
                f"drawtext=text='{title_text}':"
                "fontsize=60:fontcolor=white:"
                "x=(w-text_w)/2:y=(h-text_h)/2"
            ),
            "-c:v", "libx264", "-crf", "18",
            str(title_clip),
        ]
        ok = await asyncio.get_event_loop().run_in_executor(
            None, _run_ffmpeg, args_title, "title card"
        )
        if not ok:
            return None

        # Concatenate title card + main video
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
        ok = await asyncio.get_event_loop().run_in_executor(
            None, _run_ffmpeg, args_concat, "title concat"
        )
        return out if ok else None

    async def _burn_subtitles(
        self, source: Path, ass_path: Path, ep_num: int
    ) -> Optional[Path]:
        """Burn *ass_path* subtitles into *source*.

        Args:
            source: Input video path.
            ass_path: ASS subtitle file path.
            ep_num: Episode number.

        Returns:
            Path to the video with subtitles burned in, or ``None`` on failure.
        """
        out = Path(tempfile.gettempdir()) / f"subtitled_ep{ep_num:02d}.mp4"
        # Escape the path for FFmpeg's ass filter (backslashes and colons)
        escaped_ass = str(ass_path.resolve()).replace("\\", "/").replace(":", "\\:")
        args = [
            "-i", str(source),
            "-vf", f"ass={escaped_ass}",
            "-c:a", "copy",
            str(out),
        ]
        ok = await asyncio.get_event_loop().run_in_executor(
            None, _run_ffmpeg, args, f"burn subtitles ep{ep_num}"
        )
        return out if ok else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_shots(episode: dict[str, Any]):
        """Yield all shots from *episode* in order."""
        for scene in episode.get("scenes", []):
            for shot in scene.get("shots", []):
                yield shot

    @staticmethod
    def _find_clip(shot: dict[str, Any]) -> Optional[Path]:
        """Return the best available video clip for *shot*.

        Preference order: lipsync → generated video.

        Args:
            shot: Shot dictionary.

        Returns:
            Path to the clip file, or ``None`` if none found.
        """
        shot_id = shot.get("shot_id", "")
        slug = _slugify(shot_id)
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
    """Convert a float number of seconds to ASS timecode ``H:MM:SS.cc``.

    Args:
        seconds: Time in seconds.

    Returns:
        ASS-format timecode string.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    cs = int((s % 1) * 100)
    return f"{h}:{m:02d}:{int(s):02d}.{cs:02d}"
