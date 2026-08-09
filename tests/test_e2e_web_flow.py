"""End-to-end web smoke flow with fake model outputs and no real services."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_server
from pipeline.state import PipelineState


def fake_script():
    return {
        "episodes": [
            {
                "episode": 1,
                "scenes": [
                    {
                        "scene_id": "S01",
                        "shots": [
                            {
                                "shot_id": "S01-001",
                                "duration": 4,
                                "visual_prompt": "small midnight store, warm light",
                                "motion_prompt": "slow push-in",
                                "dialogue": "欢迎光临。",
                                "emotion": "neutral",
                            }
                        ],
                    }
                ],
            }
        ]
    }


class WebE2ESmokeTests(unittest.TestCase):
    def test_full_web_control_loop_without_model_services(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state_store: dict[str, PipelineState] = {}

            def fake_save(self, state_dir=Path("output/state"), force=False):
                state_store[self.project_id] = self.model_copy(deep=True)
                return root / "output" / "state" / f"{self.project_id}.json"

            def fake_load(project_id: str) -> PipelineState:
                return state_store[project_id]

            web_server._queue_records.clear()
            web_server._queue_order.clear()
            web_server._pending_queue_ids.clear()
            web_server._active_queue_task_id = None
            web_server._queue_runner = None

            with (
                patch.object(web_server, "ROOT", root),
                patch.object(web_server, "QUEUE_STATE_PATH", root / "output/queue_state.json"),
                patch.object(web_server, "_load_state", fake_load),
                patch.object(PipelineState, "save", fake_save),
                patch("utils.production.ROOT", root),
                patch("utils.production.SUBTITLE_ROOT", root / "output/subtitles"),
                patch("utils.production.load_asset_manifest", return_value={"characters": {}, "scenes": {}, "shots": {}, "scene_audio": {}}),
                patch.object(web_server, "validate_asset_setup", lambda: {"ok": True, "errors": [], "warnings": []}),
                patch.object(web_server, "_local_http_check", lambda url: {"ok": False, "url": url, "message": "skipped"}),
            ):
                client = TestClient(web_server.app)

                created = client.post("/api/projects", json={"prompt": "深夜便利店", "profile": "default"})
                self.assertEqual(created.status_code, 200)
                project_id = created.json()["project_id"]
                state_store[project_id].script = fake_script()
                state_store[project_id].save()

                shots = client.get(f"/api/projects/{project_id}/shots")
                self.assertEqual(shots.status_code, 200)
                self.assertEqual(shots.json()[0]["shot_id"], "S01-001")

                edited = client.put(
                    f"/api/projects/{project_id}/shots/S01-001",
                    json={"data": {"dialogue": "欢迎光临，今晚只有一盏灯。"}},
                )
                self.assertEqual(edited.status_code, 200)
                self.assertIn("今晚", edited.json()["script"]["dialogue"])

                quality = client.post(f"/api/projects/{project_id}/quality-check")
                self.assertEqual(quality.status_code, 200)
                self.assertEqual(quality.json()["failed"], 1)

                review = client.post(
                    f"/api/projects/{project_id}/shots/S01-001/review",
                    json={"status": "needs_retry", "note": "fake service found missing video"},
                )
                self.assertEqual(review.status_code, 200)
                self.assertEqual(review.json()["review_status"], "needs_retry")

                retry = client.post(f"/api/projects/{project_id}/retry-failed")
                self.assertEqual(retry.status_code, 200)
                self.assertEqual(retry.json()["shot_ids"], ["S01-001"])

                export_task = client.post(f"/api/projects/{project_id}/export")
                self.assertEqual(export_task.status_code, 200)
                if web_server._pending_queue_ids:
                    asyncio.run(web_server._queue_worker())
                export_info = client.get(f"/api/projects/{project_id}/export")
                self.assertEqual(export_info.status_code, 200)
                self.assertTrue(Path(export_info.json()["zip"]).exists())


if __name__ == "__main__":
    unittest.main()
