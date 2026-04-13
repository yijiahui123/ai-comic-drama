"""Model unloading utilities for inter-stage memory management.

Provides async helpers to release GPU/unified-memory held by Ollama and
ComfyUI between pipeline stages.  All operations are *best-effort*: failures
are logged as warnings so they never interrupt the pipeline.
"""

from __future__ import annotations

import asyncio
import gc
from typing import Optional

import aiohttp

from utils.logger import get_logger

logger = get_logger(__name__)

_UNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=30, connect=10)


async def unload_ollama_model(ollama_url: str, model_name: str) -> None:
    """Ask Ollama to evict *model_name* from memory.

    Sends ``POST /api/generate`` with ``keep_alive=0``, which instructs
    Ollama to immediately unload the model rather than keeping it warm.

    Args:
        ollama_url: Base URL of the Ollama service (e.g. ``http://localhost:11434``).
        model_name: Exact model tag as returned by ``ollama list``
            (e.g. ``llama3.1:70b-instruct-q4_K_M``).
    """
    logger.info("🧹 Unloading Ollama model '%s'…", model_name)
    url = f"{ollama_url.rstrip('/')}/api/generate"
    payload = {"model": model_name, "keep_alive": 0}

    try:
        async with aiohttp.ClientSession(timeout=_UNLOAD_TIMEOUT) as session:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
        logger.info("✅ Ollama model '%s' unloaded successfully.", model_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "⚠️  Could not unload Ollama model '%s' (non-fatal): %s", model_name, exc
        )


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

