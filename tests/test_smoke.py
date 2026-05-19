"""Smoke tests: verify core endpoints are reachable and return expected shapes."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_server
from pipeline.state import PipelineState, Stage


class SmokeTests(unittest.TestCase):
    """Fast, model-free checks that the web server starts and serves data."""

    def test_projects_endpoint_returns_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            state = PipelineState(project_id="smoke1", user_prompt="smoke test")
            state.save(state_dir=state_dir)
            with patch.object(web_server, "STATE_DIR", state_dir):
                client = TestClient(web_server.app)
                res = client.get("/api/projects")
                self.assertEqual(res.status_code, 200)
                self.assertIsInstance(res.json(), list)
                self.assertEqual(res.json()[0]["project_id"], "smoke1")

    def test_project_detail_returns_expected_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            state = PipelineState(project_id="smoke2", user_prompt="detail test", current_stage=Stage.SCRIPTING)
            state.save(state_dir=state_dir)
            with (
                patch.object(web_server, "STATE_DIR", state_dir),
                patch.object(web_server, "_load_state", lambda project_id: state),
            ):
                client = TestClient(web_server.app)
                res = client.get("/api/projects/smoke2")
                self.assertEqual(res.status_code, 200)
                data = res.json()
                for key in ("project_id", "current_stage", "user_prompt", "stages", "queue_status"):
                    self.assertIn(key, data)

    def test_create_project_returns_project_with_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            with patch.object(web_server, "STATE_DIR", state_dir):
                client = TestClient(web_server.app)
                res = client.post("/api/projects", json={"prompt": "smoke create", "profile": "default"})
                self.assertEqual(res.status_code, 200)
                self.assertIn("project_id", res.json())

    def test_queue_endpoint_returns_tasks_key(self):
        client = TestClient(web_server.app)
        res = client.get("/api/queue")
        self.assertEqual(res.status_code, 200)
        self.assertIn("tasks", res.json())

    def test_health_endpoint_returns_checks(self):
        with patch.object(web_server, "_local_http_check", lambda url: {"ok": False, "url": url, "message": "skipped"}):
            client = TestClient(web_server.app)
            res = client.get("/api/health")
            self.assertEqual(res.status_code, 200)
            self.assertIn("checks", res.json())


if __name__ == "__main__":
    unittest.main()
