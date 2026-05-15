"""Standalone video generation script.

Runs VideoGenerator independently without the full pipeline.
Useful for testing video generation with a single image or a full script.

Usage::

    # Single image → video (quickest test)
    python run_video_gen.py --image path/to/image.png --prompt "camera slowly pans right"

    # Full script JSON → all videos
    python run_video_gen.py --script path/to/script.json

    # Specify custom duration (seconds)
    python run_video_gen.py --image shot.png --prompt "subtle breathing" --duration 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

from skills.video_generator.skill import VideoGenerator
from utils.logger import get_logger

logger = get_logger("run_video_gen")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Wan 2.2 video generation tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        metavar="PATH",
        help="Path to a single storyboard image (generates one video)",
    )
    group.add_argument(
        "--script",
        metavar="PATH",
        help="Path to a validated script JSON (generates videos for all shots)",
    )

    parser.add_argument(
        "--prompt",
        default="",
        help="Motion prompt for single-image mode (default: empty)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=4.0,
        help="Video duration in seconds for single-image mode (default: 4)",
    )
    parser.add_argument(
        "--comfyui",
        default="http://localhost:8188",
        help="ComfyUI server URL (default: http://localhost:8188)",
    )

    return parser.parse_args(argv)


def _make_single_shot(image_path: str, prompt: str, duration: float) -> dict:
    """Build a minimal script dict for a single shot."""
    shot_id = Path(image_path).stem
    return {
        "title": "single_test",
        "style": "anime",
        "characters": [],
        "episodes": [
            {
                "episode": 1,
                "scenes": [
                    {
                        "scene_id": "S01",
                        "location": "test",
                        "shots": [
                            {
                                "shot_id": shot_id,
                                "type": "中景",
                                "characters": [],
                                "dialogue": "",
                                "visual_prompt": "",
                                "motion_prompt": prompt,
                                "camera_move": "static",
                                "duration": duration,
                                "mood": "neutral",
                            }
                        ],
                    }
                ],
            }
        ],
    }


async def _run_single(args: argparse.Namespace) -> int:
    """Generate a video from a single image."""
    image_path = Path(args.image)
    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        return 1

    # Copy image to assets/shots/ where VideoGenerator expects it
    assets_dir = Path("assets/shots")
    assets_dir.mkdir(parents=True, exist_ok=True)
    dest = assets_dir / image_path.name
    if not dest.exists():
        import shutil
        shutil.copy2(image_path, dest)

    script = _make_single_shot(image_path.name, args.prompt, args.duration)

    async with VideoGenerator(comfyui_url=args.comfyui) as gen:
        logger.info("Mode: %s", gen._mode)
        logger.info("Generating video for %s (%.1fs)...", image_path.name, args.duration)
        result = await gen.generate_all(script)

    videos = result.get("videos", [])
    if videos:
        logger.info("Done! Video saved: %s", videos[0])
        return 0
    else:
        logger.error("No video produced.")
        return 1


async def _run_script(args: argparse.Namespace) -> int:
    """Generate videos for all shots in a script JSON."""
    script_path = Path(args.script)
    if not script_path.exists():
        logger.error("Script not found: %s", script_path)
        return 1

    with script_path.open(encoding="utf-8") as f:
        script = json.load(f)

    shots = []
    for ep in script.get("episodes", []):
        for sc in ep.get("scenes", []):
            shots.extend(sc.get("shots", []))
    logger.info("Loaded script: %d shots", len(shots))

    async with VideoGenerator(comfyui_url=args.comfyui) as gen:
        logger.info("Mode: %s", gen._mode)
        result = await gen.generate_all(script)

    videos = result.get("videos", [])
    audio = result.get("audio", [])
    logger.info("Done! %d videos, %d audio files", len(videos), len(audio))
    return 0


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.image:
        return await _run_single(args)
    else:
        return await _run_script(args)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
