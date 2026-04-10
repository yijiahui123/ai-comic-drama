"""Async HTTP client with retry, timeout, and health-check support."""

from __future__ import annotations

import asyncio
from typing import Any, Optional

import aiohttp

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = aiohttp.ClientTimeout(total=120, connect=10)
DEFAULT_RETRY_COUNT = 3
DEFAULT_RETRY_DELAY = 2.0  # seconds


class HTTPClient:
    """Thin async wrapper around :class:`aiohttp.ClientSession`.

    Features:
    - Configurable connect / total timeouts.
    - Automatic retry with exponential back-off on network errors.
    - ``health_check`` convenience method.
    """

    def __init__(
        self,
        base_url: str,
        timeout: aiohttp.ClientTimeout = DEFAULT_TIMEOUT,
        retry_count: int = DEFAULT_RETRY_COUNT,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> None:
        """
        Args:
            base_url: Root URL for all requests (e.g. ``http://localhost:8188``).
            timeout: aiohttp timeout configuration.
            retry_count: Number of retries on transient network failures.
            retry_delay: Base delay (seconds) between retries; doubled each attempt.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "HTTPClient":
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the underlying aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        """Execute an HTTP request with retry logic.

        Args:
            method: HTTP verb (``GET``, ``POST``, etc.).
            path: URL path relative to ``base_url``.
            **kwargs: Extra arguments forwarded to :meth:`aiohttp.ClientSession.request`.

        Returns:
            Parsed JSON response body.

        Raises:
            aiohttp.ClientError: After all retries are exhausted.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        session = self._get_session()
        delay = self.retry_delay

        last_exc: Exception = RuntimeError("No attempts made")
        for attempt in range(1, self.retry_count + 2):
            try:
                async with session.request(method, url, **kwargs) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_exc = exc
                if attempt <= self.retry_count:
                    logger.warning(
                        "Request %s %s failed (attempt %d/%d): %s — retrying in %.1fs",
                        method,
                        url,
                        attempt,
                        self.retry_count + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2
                else:
                    logger.error(
                        "Request %s %s failed after %d attempts: %s",
                        method,
                        url,
                        attempt,
                        exc,
                    )
        raise last_exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get(self, path: str, **kwargs: Any) -> Any:
        """HTTP GET.

        Args:
            path: URL path.
            **kwargs: Forwarded to :meth:`aiohttp.ClientSession.request`.
        """
        return await self._request("GET", path, **kwargs)

    async def post(self, path: str, **kwargs: Any) -> Any:
        """HTTP POST.

        Args:
            path: URL path.
            **kwargs: Forwarded to :meth:`aiohttp.ClientSession.request`.
        """
        return await self._request("POST", path, **kwargs)

    async def health_check(self, path: str = "/") -> bool:
        """Check whether the remote service is reachable.

        Args:
            path: Health-check endpoint (default ``/``).

        Returns:
            ``True`` if the service responds with a 2xx status, ``False`` otherwise.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        session = self._get_session()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                ok = resp.status < 400
                if ok:
                    logger.debug("Health check OK: %s", url)
                else:
                    logger.warning("Health check failed (HTTP %d): %s", resp.status, url)
                return ok
        except Exception as exc:  # noqa: BLE001
            logger.warning("Health check unreachable: %s — %s", url, exc)
            return False
