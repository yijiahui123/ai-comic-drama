"""CLI entry point for the AI Comic Drama pipeline.

Usage examples::

    # Start a new run
    python main.py --prompt "写一个赛博朋克风格的3分钟漫剧，主角是黑客少女"

    # Resume an interrupted run
    python main.py --resume <project_id>

    # Check the status of a run
    python main.py --status <project_id>
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.state import PipelineState
from utils.logger import get_logger

logger = get_logger("main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Parsed :class:`argparse.Namespace`.
    """
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

    return parser.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    """Async entry point.

    Args:
        argv: Argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success, 1 = error).
    """
    args = parse_args(argv)

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
