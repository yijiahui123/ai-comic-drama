"""Model unloading utilities for inter-stage memory management.

Provides async helpers to release GPU/unified-memory held by oMLX, ComfyUI,
ChatTTS, and SadTalker between pipeline stages.  All operations are
*best-effort*: failures are logged as warnings so they never interrupt
the pipeline.
"""

from __future__ import annotations

import asyncio
import gc
import os
import signal
import subprocess
from typing import Optional

import aiohttp

from utils.logger import get_logger

logger = get_logger(__name__)
_UNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=30, connect=10)


async def kill_omlx_server() -> None:
    """Kill the oMLX server process to free LLM memory (~20-25GB).

    oMLX is only used in the SCRIPTING stage.  After that, killing the
    process is the most reliable way to reclaim memory on macOS, since
    oMLX doesn't expose an explicit model-unload API.

    Finds the process by name and sends SIGTERM, then SIGKILL if needed.
    """
    logger.info("🧹 Killing oMLX server process to free LLM memory…")
    try:
        # Find oMLX processes (exclude our own PID)
        result = subprocess.run(
            ["pgrep", "-f", "omlx"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        pids = [int(pid) for pid in result.stdout.strip().split("\n") if pid.strip()]

        if not pids:
            logger.info("ℹ️  No oMLX process found (already stopped?).")
            return

        for pid in pids:
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info("  → Sent SIGTERM to oMLX process %d", pid)
            except ProcessLookupError:
                pass

        # Wait briefly for graceful shutdown, then force kill
        await asyncio.sleep(2)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
                logger.info("  → Sent SIGKILL to oMLX process %d", pid)
            except ProcessLookupError:
                pass

        logger.info("✅ oMLX server killed. ~20-25GB memory freed.")
    except Exception as exc:  # noqa: BLE001
        logger.warning("⚠️  Could not kill oMLX server (non-fatal): %s", exc)


async def unload_comfyui_models(comfyui_url: str) -> None:
    """Ask ComfyUI to free all loaded models and GPU/RAM caches.

    Calls ``POST /free`` with ``{"unload_models": true, "free_memory": true}``.

    Args:
        comfyui_url: Base URL of the ComfyUI service (e.g. ``http://localhost:8188``).
    """
    logger.info("🧹 Releasing ComfyUI model cache…")
    url = f"{comfyui_url.rstrip('/')}/free"
    payload = {"unload_models": True, "free_memory": True}

    try:
        async with aiohttp.ClientSession(timeout=_UNLOAD_TIMEOUT) as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
        logger.info("✅ ComfyUI models unloaded successfully.")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "⚠️  Could not unload ComfyUI models (non-fatal): %s", exc
        )


def force_gc() -> None:
    """Run Python's cyclic garbage collector to reclaim freed objects.

    Logs the number of objects collected at each generation.
    """
    collected = gc.collect()
    logger.debug("🗑️  gc.collect() freed %d objects.", collected)

