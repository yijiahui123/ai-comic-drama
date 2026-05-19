"""Shared ComfyUI client helpers — retry POST and WebSocket progress polling."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Optional

import aiohttp

from utils.logger import get_logger

logger = get_logger(__name__)

# Default timeouts
_WS_CONNECT_TIMEOUT = 10.0


async def post_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    payload: dict[str, Any],
    retries: int = 3,
    delay: float = 2.0,
) -> dict[str, Any]:
    """POST JSON to *url* with exponential-backoff retry on transient errors."""
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return await resp.json()
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_exc = exc
            if attempt < retries:
                logger.warning(
                    "POST %s failed (attempt %d/%d): %s — retrying in %.1fs",
                    url, attempt, retries, exc, delay,
                )
                await asyncio.sleep(delay)
                delay *= 2
            else:
                logger.error("POST %s failed after %d attempts: %s", url, retries, exc)
    if isinstance(last_exc, ConnectionRefusedError):
        raise ConnectionError(f"ComfyUI 服务未启动，请检查 {url} 是否可达") from last_exc
    if isinstance(last_exc, asyncio.TimeoutError):
        raise TimeoutError("ComfyUI 响应超时，可能正在处理其他任务") from last_exc
    raise last_exc  # type: ignore[misc]


async def poll_comfyui_ws(
    comfyui_url: str,
    prompt_id: str,
    client_id: str,
    timeout: float = 1800.0,
    on_progress: Optional[Callable[[int, int, str], None]] = None,
) -> bool:
    """Connect to ComfyUI WebSocket and wait for *prompt_id* to finish.

    Calls *on_progress(value, max, node_id)* on each progress event.
    Returns ``True`` if execution completed, ``False`` on timeout or error.
    """
    ws_url = comfyui_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws?clientId={client_id}"
    deadline = time.monotonic() + timeout
    try:
        async with asyncio.timeout(_WS_CONNECT_TIMEOUT):
            async with aiohttp.ClientSession() as ws_session:
                async with ws_session.ws_connect(ws_url, heartbeat=30) as ws:
                    while time.monotonic() < deadline:
                        try:
                            msg = await asyncio.wait_for(
                                ws.receive(), timeout=min(30, deadline - time.monotonic())
                            )
                        except asyncio.TimeoutError:
                            continue
                        if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        try:
                            data = json.loads(msg.data)
                        except (json.JSONDecodeError, TypeError):
                            continue
                        evt_type = data.get("type", "")
                        evt_data = data.get("data", {})
                        if evt_type == "progress" and evt_data.get("prompt_id") == prompt_id:
                            val = evt_data.get("value", 0)
                            mx = evt_data.get("max", 1)
                            if on_progress:
                                on_progress(val, mx, "")
                        elif evt_type == "executing" and evt_data.get("prompt_id") == prompt_id:
                            node = evt_data.get("node")
                            if node is None:
                                return True  # Execution complete
                            if on_progress:
                                on_progress(0, 0, str(node))
                        elif evt_type == "execution_error" and evt_data.get("prompt_id") == prompt_id:
                            logger.error("ComfyUI execution error: %s", evt_data.get("exception_message", ""))
                            return False
    except Exception as exc:  # noqa: BLE001
        logger.debug("WebSocket progress tracking unavailable: %s", exc)
    return False
