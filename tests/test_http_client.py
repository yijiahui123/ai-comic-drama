import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp

from utils.http_client import HTTPClient


class HTTPClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_returns_json(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value={"ok": True})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.request = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        client = HTTPClient("http://localhost:9999")
        client._session = mock_session
        result = await client.get("/test")
        self.assertEqual(result, {"ok": True})

    async def test_post_returns_json(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json = AsyncMock(return_value={"created": True})
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = AsyncMock()
        mock_session.request = MagicMock(return_value=mock_resp)
        mock_session.closed = False

        client = HTTPClient("http://localhost:9999")
        client._session = mock_session
        result = await client.post("/create", json={"name": "test"})
        self.assertEqual(result, {"created": True})

    async def test_retry_on_client_error(self):
        call_count = 0

        class FailingThenSucceed:
            def __init__(self):
                self.closed = False

            def request(self, method, url, **kwargs):
                nonlocal call_count
                call_count += 1

                if call_count <= 2:
                    resp = AsyncMock()
                    resp.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("fail"))
                    resp.__aexit__ = AsyncMock(return_value=False)
                    return resp
                resp = AsyncMock()
                resp.raise_for_status = MagicMock()
                resp.json = AsyncMock(return_value={"ok": True})
                resp.__aenter__ = AsyncMock(return_value=resp)
                resp.__aexit__ = AsyncMock(return_value=False)
                return resp

        client = HTTPClient("http://localhost:9999", retry_count=3, retry_delay=0.01)
        client._session = FailingThenSucceed()
        result = await client.get("/test")
        self.assertEqual(result, {"ok": True})
        self.assertEqual(call_count, 3)

    async def test_raises_after_all_retries_exhausted(self):
        mock_session = MagicMock()
        mock_session.closed = False

        def always_fail(method, url, **kwargs):
            resp = AsyncMock()
            resp.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("always fail"))
            resp.__aexit__ = AsyncMock(return_value=False)
            return resp

        mock_session.request = always_fail

        client = HTTPClient("http://localhost:9999", retry_count=2, retry_delay=0.01)
        client._session = mock_session
        with self.assertRaises(aiohttp.ClientError):
            await client.get("/test")

    async def test_health_check_returns_true_on_success(self):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(return_value=mock_resp)

        client = HTTPClient("http://localhost:9999")
        client._session = mock_session
        result = await client.health_check()
        self.assertTrue(result)

    async def test_health_check_returns_false_on_error(self):
        mock_session = MagicMock()
        mock_session.closed = False
        mock_session.get = MagicMock(side_effect=ConnectionError("refused"))

        client = HTTPClient("http://localhost:9999")
        client._session = mock_session
        result = await client.health_check()
        self.assertFalse(result)

    async def test_context_manager_creates_and_closes_session(self):
        async with HTTPClient("http://localhost:9999") as client:
            self.assertIsNotNone(client._session)
            self.assertFalse(client._session.closed)
        # After exit, session should be closed
        self.assertTrue(client._session is None or client._session.closed)

    async def test_close_sets_session_to_none(self):
        client = HTTPClient("http://localhost:9999")
        client._session = aiohttp.ClientSession()
        await client.close()
        self.assertIsNone(client._session)


if __name__ == "__main__":
    unittest.main()
