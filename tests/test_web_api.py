import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import web_server
from pipeline.state import PipelineState, Stage
from utils.assets import load_asset_config, load_asset_manifest, save_asset_config, save_asset_manifest


class WebApiTests(unittest.TestCase):
    def _script(self):
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
                                    "visual_prompt": "room",
                                    "motion_prompt": "walk",
                                    "dialogue": "",
                                }
                            ],
                        }
                    ],
                }
            ]
        }

    def test_asset_config_and_manifest_endpoints_use_mapping_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "configs/assets.yaml"
            manifest_path = root / "assets/library/manifest.yaml"
            save_asset_config({}, config_path)
            save_asset_manifest({}, manifest_path)

            with (
                patch.object(web_server, "ASSET_CONFIG_PATH", config_path),
                patch.object(web_server, "ASSET_MANIFEST_PATH", manifest_path),
                patch.object(web_server, "load_asset_config", lambda: load_asset_config(config_path)),
                patch.object(web_server, "load_asset_manifest", lambda: load_asset_manifest(manifest_path)),
                patch.object(web_server, "save_asset_config", lambda data: save_asset_config(data, config_path)),
                patch.object(web_server, "save_asset_manifest", lambda data: save_asset_manifest(data, manifest_path)),
            ):
                client = TestClient(web_server.app)
                res = client.get("/api/assets")
                self.assertEqual(res.status_code, 200)
                self.assertIn("config", res.json())

                res = client.put("/api/assets/config-json", json={"data": {"continuity": {"max_shot_duration_seconds": 5}}})
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.json()["config"]["continuity"]["max_shot_duration_seconds"], 5)

                manifest_yaml = "characters: {}\nscenes: {}\nshots:\n  S01: assets/library/shots/s01.png\n"
                res = client.put("/api/assets/manifest", json={"content": manifest_yaml})
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.json()["manifest"]["shots"]["S01"], "assets/library/shots/s01.png")

    def test_upload_and_delete_asset_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "assets/library/manifest.yaml"
            save_asset_manifest({}, manifest_path)

            with (
                patch.object(web_server, "ROOT", root),
                patch.object(web_server, "ASSET_MANIFEST_PATH", manifest_path),
                patch.object(web_server, "load_asset_manifest", lambda: load_asset_manifest(manifest_path)),
                patch.object(web_server, "save_asset_manifest", lambda data: save_asset_manifest(data, manifest_path)),
            ):
                client = TestClient(web_server.app)
                res = client.post(
                    "/api/assets/upload",
                    data={"asset_type": "shot", "key": "S01E01-001", "emotion": ""},
                    files={"file": ("shot.png", b"fake-png", "image/png")},
                )
                self.assertEqual(res.status_code, 200)
                uploaded = root / res.json()["path"]
                self.assertTrue(uploaded.exists())

                res = client.delete(
                    "/api/assets/binding",
                    params={"asset_type": "shot", "key": "S01E01-001", "delete_file": "true"},
                )
                self.assertEqual(res.status_code, 200)
                self.assertFalse(uploaded.exists())
                self.assertNotIn("S01E01-001", res.json()["manifest"]["shots"])

    def test_scene_audio_endpoint_stores_alignment_structure(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.yaml"
            save_asset_manifest({}, manifest_path)

            with (
                patch.object(web_server, "load_asset_manifest", lambda: load_asset_manifest(manifest_path)),
                patch.object(web_server, "save_asset_manifest", lambda data: save_asset_manifest(data, manifest_path)),
            ):
                client = TestClient(web_server.app)
                payload = {
                    "scene_id": "S01E01",
                    "audio_path": "output/audio/s01e01_scene.wav",
                    "segments": [{"shot_id": "S01E01-001", "start": 0, "end": 4.2, "text": "hello"}],
                }
                res = client.put("/api/assets/scene-audio", json=payload)
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.json()["manifest"]["scene_audio"]["S01E01"]["segments"][0]["shot_id"], "S01E01-001")

    def test_project_list_reads_state_without_starting_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            state = PipelineState(project_id="abc123", user_prompt="测试项目", current_stage=Stage.INIT)
            state.save(state_dir=state_dir)
            with patch.object(web_server, "STATE_DIR", state_dir):
                client = TestClient(web_server.app)
                res = client.get("/api/projects")
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.json()[0]["project_id"], "abc123")

    def test_shot_cleanup_deletes_files_and_marks_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root / "output/videos/S01E01-001.mp4"
            audio = root / "output/audio/S01E01-001.wav"
            shot = root / "assets/shots/s01e01-001.png"
            for path in (video, audio, shot):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x")
            state = PipelineState(
                project_id="abc123",
                user_prompt="测试项目",
                current_stage=Stage.DONE,
                asset_manifest={"shots": ["assets/shots/s01e01-001.png"]},
                video_manifest={"videos": ["output/videos/S01E01-001.mp4"], "audio": ["output/audio/S01E01-001.wav"]},
            )

            with (
                patch.object(web_server, "ROOT", root),
                patch.object(web_server, "_load_state", lambda project_id: state),
                patch.object(PipelineState, "save", lambda self, state_dir=Path("output/state"), force=False: Path("state.json")),
            ):
                client = TestClient(web_server.app)
                res = client.post(
                    "/api/projects/abc123/shots/S01E01-001/cleanup",
                    json={"include_asset": True, "delete_files": True},
                )
                self.assertEqual(res.status_code, 200)
                self.assertFalse(video.exists())
                self.assertFalse(audio.exists())
                self.assertFalse(shot.exists())
                self.assertEqual(res.json()["current_stage"], "ASSET_GEN")

    def test_validate_assets_endpoint_is_callable_without_model_services(self):
        expected = {"ok": True, "errors": [], "warnings": [], "config": {}, "manifest": {}}
        with patch.object(web_server, "validate_asset_setup", lambda: expected):
            client = TestClient(web_server.app)
            res = client.post("/api/assets/validate")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(json.loads(res.content), expected)

    def test_queue_and_health_endpoints_are_callable_without_model_services(self):
        with (
            patch.object(web_server, "validate_asset_setup", lambda: {"ok": True, "errors": [], "warnings": []}),
            patch.object(web_server, "_local_http_check", lambda url: {"ok": False, "url": url, "message": "skipped"}),
        ):
            client = TestClient(web_server.app)
            self.assertEqual(client.get("/api/queue").status_code, 200)
            res = client.get("/api/health")
            self.assertEqual(res.status_code, 200)
            self.assertIn("ffmpeg", res.json()["checks"])

    def test_cancel_queue_task_returns_canceled_status(self):
        web_server._queue_records.clear()
        web_server._queue_order.clear()
        web_server._queue_records["task1"] = {
            "task_id": "task1",
            "kind": "test",
            "project_id": None,
            "payload": {},
            "status": "queued",
            "message": "Queued",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "runner": None,
        }
        web_server._queue_order.append("task1")
        client = TestClient(web_server.app)
        res = client.post("/api/queue/task1/cancel")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "canceled")

    def test_shot_edit_lock_review_quality_and_retry_apis(self):
        state = PipelineState(project_id="abc123", user_prompt="测试项目", script=self._script())

        with (
            patch.object(web_server, "_load_state", lambda project_id: state),
            patch.object(PipelineState, "save", lambda self, state_dir=Path("output/state"), force=False: Path("state.json")),
            patch("utils.production.load_asset_manifest", return_value={"shots": {}, "scenes": {}, "characters": {}, "scene_audio": {}}),
        ):
            client = TestClient(web_server.app)
            res = client.get("/api/projects/abc123/shots")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()[0]["shot_id"], "S01-001")

            res = client.put("/api/projects/abc123/shots/S01-001", json={"data": {"dialogue": "updated"}})
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["script"]["dialogue"], "updated")

            res = client.post("/api/projects/abc123/shots/S01-001/lock")
            self.assertEqual(res.status_code, 200)
            self.assertTrue(res.json()["locked"])

            res = client.post("/api/projects/abc123/shots/S01-001/review", json={"status": "needs_retry", "note": "bad"})
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["review_status"], "needs_retry")

            res = client.post("/api/projects/abc123/shots/S01-001/quality-check")
            self.assertEqual(res.status_code, 200)
            self.assertIn(res.json()["status"], {"failed", "needs_review"})

            res = client.post("/api/projects/abc123/retry-failed")
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["shot_ids"], [])


    def test_batch_create_returns_multiple_projects(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            with patch.object(web_server, "STATE_DIR", state_dir):
                client = TestClient(web_server.app)
                res = client.post("/api/batch", json={"prompts": ["prompt A", "prompt B"], "profile": "default"})
                self.assertEqual(res.status_code, 200)
                self.assertEqual(res.json()["count"], 2)

    def test_delete_project_removes_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            state = PipelineState(project_id="delme", user_prompt="delete me", current_stage=Stage.INIT)
            state.save(state_dir=state_dir)
            with patch.object(web_server, "STATE_DIR", state_dir):
                client = TestClient(web_server.app)
                # Confirm exists
                res = client.get("/api/projects")
                self.assertEqual(len(res.json()), 1)
                # Delete
                res = client.delete("/api/projects/delme")
                self.assertEqual(res.status_code, 200)
                self.assertTrue(res.json()["ok"])
                # Confirm gone
                res = client.get("/api/projects")
                self.assertEqual(len(res.json()), 0)


if __name__ == "__main__":
    unittest.main()
