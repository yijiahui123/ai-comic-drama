"""Title card generator for the Editor skill.

This module provides a helper to produce a short title-card clip (black
background with white text) using FFmpeg's ``lavfi`` colour source and
``drawtext`` filter.  It is invoked by :class:`skills.editor.skill.Editor`
when assembling an episode.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def generate_title_card(
    title: str,
    episode: int,
    output_path: Path,
    duration: float = 3.0,
    width: int = 1280,
    height: int = 720,
    font_size: int = 60,
    font_color: str = "white",
    bg_color: str = "black",
) -> bool:
    """Generate a title-card video clip via FFmpeg.

    Args:
        title: Main title text (e.g. the script title).
        episode: Episode number displayed beneath the title.
        output_path: Where to save the resulting ``.mp4``.
        duration: Duration of the card in seconds.
        width: Output video width.
        height: Output video height.
        font_size: Title font size in pixels.
        font_color: Title text colour (FFmpeg colour string).
        bg_color: Background colour (FFmpeg colour string).

    Returns:
        ``True`` if FFmpeg succeeded, ``False`` otherwise.
    """
    line1 = title.replace("'", "\\'")
    line2 = f"Episode {episode}".replace("'", "\\'")

    drawtext = (
        f"drawtext=text='{line1}':fontsize={font_size}:fontcolor={font_color}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2-40,"
        f"drawtext=text='{line2}':fontsize={font_size // 2}:fontcolor={font_color}:"
        f"x=(w-text_w)/2:y=(h-text_h)/2+40"
    )

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c={bg_color}:size={width}x{height}:rate=24:duration={duration}",
        "-vf", drawtext,
        "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0
