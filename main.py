"""CLI entry point for the AI Comic Drama pipeline.

Usage examples::

    # Start a new run
    python main.py --prompt "写一个赛博朋克风格的3分钟漫剧，主角是黑客少女"

    # Resume an interrupted run
    python main.py --resume <project_id>

    # Check the status of a run
    python main.py --status <project_id>

    # Batch mode: one prompt per line in a text file
    python main.py --batch prompts.txt
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.state import PipelineState
from utils.logger import get_logger

logger = get_logger("main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Comic Drama — fully automated comic-drama generation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--prompt",
        metavar="DESCRIPTION",
        help="Natural-language description of the desired comic drama",
    )
    group.add_argument(
        "--resume",
        metavar="PROJECT_ID",
        help="Resume an interrupted pipeline run by project ID",
    )
    group.add_argument(
        "--status",
        metavar="PROJECT_ID",
        help="Print the current status of a pipeline run and exit",
    )
    group.add_argument(
        "--batch",
        metavar="FILE",
        help="Text file with one prompt per line; runs each sequentially",
    )

    return parser.parse_args(argv)


async def _run_single(prompt: str) -> tuple[str, bool]:
    orchestrator = PipelineOrchestrator.new(prompt)
    project_id = orchestrator.state.project_id
    logger.info("Starting project '%s'…", project_id)
    try:
        final_state = await orchestrator.run()
        success = bool(final_state.final_video)
    except Exception as exc:  # noqa: BLE001
        logger.error("Project '%s' failed: %s", project_id, exc)
        success = False
    return project_id, success


async def _run_batch(file_path: str) -> int:
    path = Path(file_path)
    if not path.exists():
        logger.error("Batch file not found: %s", file_path)
        return 1
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        logger.error("Batch file is empty: %s", file_path)
        return 1
    total = len(prompts)
    succeeded = 0
    failed_projects: list[str] = []
    logger.info("Batch mode: %d prompts to process", total)
    for idx, prompt in enumerate(prompts, 1):
        logger.info("[%d/%d] %s", idx, total, prompt[:80])
        project_id, success = await _run_single(prompt)
        if success:
            succeeded += 1
            logger.info("[%d/%d] ✅ %s", idx, total, project_id)
        else:
            failed_projects.append(project_id)
            logger.error("[%d/%d] ❌ %s", idx, total, project_id)
    print(f"\n{'='*60}")
    print(f"Batch complete: {succeeded}/{total} succeeded")
    if failed_projects:
        print(f"Failed projects: {', '.join(failed_projects)}")
    return 0 if not failed_projects else 1


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # --- Batch mode ---
    if args.batch:
        return await _run_batch(args.batch)

    # --- Status query ---
    if args.status:
        try:
            state = PipelineState.load(args.status)
            print(state.status_summary())
            return 0
        except FileNotFoundError:
            logger.error("No project found with ID '%s'", args.status)
            return 1

    # --- Resume ---
    if args.resume:
        try:
            orchestrator = PipelineOrchestrator.resume(args.resume)
            logger.info("Resuming project '%s'…", args.resume)
        except FileNotFoundError:
            logger.error("No project found with ID '%s'", args.resume)
            return 1

    # --- New run ---
    else:
        orchestrator = PipelineOrchestrator.new(args.prompt)
        logger.info(
            "Starting new project '%s'…", orchestrator.state.project_id
        )

    # --- Execute pipeline ---
    try:
        final_state = await orchestrator.run()
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
        return 1

    if final_state.final_video:
        print(f"\n✅ Done!  Final video: {final_state.final_video}")
        return 0
    else:
        logger.error(
            "Pipeline did not produce a final video.  Stage: %s",
            final_state.current_stage.value,
        )
        print(f"\n❌ Pipeline ended with status: {final_state.current_stage.value}")
        print(final_state.status_summary())
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
