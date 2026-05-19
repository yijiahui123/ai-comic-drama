import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline.state import PipelineState, ShotState
from utils.production import (
    build_shot_states,
    export_project,
    generate_shot_subtitles,
    mark_failed_for_retry,
    quality_check_project,
    quality_check_shot,
    review_shot,
    scene_audio_segments,
    set_shot_lock,
    update_script_shot,
    validate_scene_segments,
)


def sample_script():
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
                                "dialogue": "hello",
                                "emotion": "neutral",
                            }
                        ],
                    }
                ],
            }
        ]
    }


class ProductionUtilityTests(unittest.TestCase):
    def test_build_shot_states_from_script_and_manifests(self):
        state = PipelineState(project_id="p1", user_prompt="x", script=sample_script())
        with patch("utils.production.load_asset_manifest", return_value={"shots": {}, "scenes": {}, "characters": {}, "scene_audio": {}}):
            shots = build_shot_states(state)
        self.assertIn("S01-001", shots)
        self.assertEqual(shots["S01-001"].scene_id, "S01")
        self.assertEqual(shots["S01-001"].script["visual_prompt"], "room")

    def test_locked_shot_is_not_marked_for_retry(self):
        state = PipelineState(project_id="p1", user_prompt="x", script=sample_script())
        state.shot_states = {
            "S01-001": ShotState(
                shot_id="S01-001",
                scene_id="S01",
                status="failed",
                review_status="needs_retry",
                locked=True,
            )
        }
        with patch("utils.production.load_asset_manifest", return_value={"shots": {}, "scenes": {}, "characters": {}, "scene_audio": {}}):
            queued = mark_failed_for_retry(state)
        self.assertEqual(queued, [])
        self.assertEqual(state.shot_states["S01-001"].retry_count, 0)

    def test_quality_check_marks_missing_outputs(self):
        state = PipelineState(project_id="p1", user_prompt="x", script=sample_script())
        with patch("utils.production.load_asset_manifest", return_value={"shots": {}, "scenes": {}, "characters": {}, "scene_audio": {}}):
            report = quality_check_project(state)
        self.assertEqual(report["failed"], 1)
        self.assertIn("S01-001", state.review_queue)

    def test_scene_audio_segments_and_subtitle_generation(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = PipelineState(project_id="p1", user_prompt="x", script=sample_script())
            manifest = {
                "shots": {},
                "scenes": {},
                "characters": {},
                "scene_audio": {
                    "S01": {
                        "audio_path": "output/audio/s01_scene.wav",
                        "segments": [{"shot_id": "S01-001", "start": 0, "end": 4, "text": "from segment"}],
                    }
                },
            }
            with (
                patch("utils.production.SUBTITLE_ROOT", Path(tmp) / "subtitles"),
                patch("utils.production.load_asset_manifest", return_value=manifest),
            ):
                self.assertEqual(scene_audio_segments(manifest)["S01"][0]["text"], "from segment")
                generated = generate_shot_subtitles(state)
            self.assertEqual(len(generated), 1)
            self.assertIn("from segment", Path(generated[0]).read_text(encoding="utf-8"))

    def test_export_project_creates_manifest_and_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = root / "output/final/p1.mp4"
            final.parent.mkdir(parents=True)
            final.write_bytes(b"video")
            state = PipelineState(project_id="p1", user_prompt="x", script=sample_script(), final_video=str(final))
            with (
                patch("utils.production.SUBTITLE_ROOT", root / "output/subtitles"),
                patch("utils.production.load_asset_manifest", return_value={"shots": {}, "scenes": {}, "characters": {}, "scene_audio": {}}),
            ):
                manifest = export_project(state, root)
            self.assertTrue(Path(manifest["zip"]).exists())
            self.assertTrue(Path(manifest["files"]["script"]).exists())

    def test_review_shot_updates_review_queue(self):
        state = PipelineState(project_id="p1", user_prompt="x", script=sample_script())
        with patch("utils.production.load_asset_manifest", return_value={"shots": {}, "scenes": {}, "characters": {}, "scene_audio": {}}):
            shot = review_shot(state, "S01-001", "needs_retry", "bad frame")
        self.assertEqual(shot.review_status, "needs_retry")
        self.assertIn("S01-001", state.review_queue)

    def test_validate_scene_segments_detects_overlap_and_gap(self):
        state = PipelineState(project_id="p1", user_prompt="x", script=sample_script())
        state.shot_states = {
            "S01-001": ShotState(shot_id="S01-001", scene_id="S01", script={"duration": 4}),
            "S01-002": ShotState(shot_id="S01-002", scene_id="S01", script={"duration": 3}),
        }
        manifest = {
            "scene_audio": {
                "S01": {
                    "segments": [
                        {"shot_id": "S01-001", "start": 0, "end": 5, "text": "a"},
                        {"shot_id": "S01-002", "start": 4, "end": 7, "text": "b"},
                    ]
                }
            }
        }
        checks = validate_scene_segments(state, manifest)
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].status, "fail")
        self.assertIn("overlap", checks[0].message)

    def test_validate_scene_segments_passes_clean_timeline(self):
        state = PipelineState(project_id="p1", user_prompt="x", script=sample_script())
        state.shot_states = {
            "S01-001": ShotState(shot_id="S01-001", scene_id="S01", script={"duration": 4}),
            "S01-002": ShotState(shot_id="S01-002", scene_id="S01", script={"duration": 3}),
        }
        manifest = {
            "scene_audio": {
                "S01": {
                    "segments": [
                        {"shot_id": "S01-001", "start": 0, "end": 4, "text": "a"},
                        {"shot_id": "S01-002", "start": 4, "end": 7, "text": "b"},
                    ]
                }
            }
        }
        checks = validate_scene_segments(state, manifest)
        self.assertEqual(checks[0].status, "pass")

    def test_export_includes_subtitles_and_asset_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            final = root / "output/final/p1.mp4"
            final.parent.mkdir(parents=True)
            final.write_bytes(b"video")
            sub = root / "output/subtitles/S01-001.ass"
            sub.parent.mkdir(parents=True)
            sub.write_text("[Script Info]\nTitle: test", encoding="utf-8")
            state = PipelineState(project_id="p1", user_prompt="x", script=sample_script(), final_video=str(final))
            state.shot_states = {
                "S01-001": ShotState(
                    shot_id="S01-001",
                    scene_id="S01",
                    script={"dialogue": "hello", "duration": 4},
                    outputs={"subtitle": str(sub)},
                )
            }
            with (
                patch("utils.production.SUBTITLE_ROOT", root / "output/subtitles"),
                patch("utils.production.load_asset_manifest", return_value={"shots": {}, "scenes": {}, "characters": {}, "scene_audio": {}}),
            ):
                manifest = export_project(state, root)
            export_dir = Path(manifest["directory"])
            self.assertTrue((export_dir / "subtitles" / "S01-001.ass").exists())
            self.assertTrue((export_dir / "asset_list.txt").exists())
            self.assertTrue((export_dir / "log_index.txt").exists())

    def test_cleanup_preserves_library_assets(self):
        """Verify that cleanup only deletes generated assets, not library assets."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library_asset = root / "assets/library/characters/hero/reference.png"
            library_asset.parent.mkdir(parents=True)
            library_asset.write_bytes(b"library")
            generated_asset = root / "assets/shots/s01-001.png"
            generated_asset.parent.mkdir(parents=True)
            generated_asset.write_bytes(b"generated")
            state = PipelineState(
                project_id="p1",
                user_prompt="x",
                script=sample_script(),
                asset_manifest={
                    "characters": ["assets/library/characters/hero/reference.png"],
                    "shots": ["assets/shots/s01-001.png"],
                    "scenes": [],
                },
                asset_sources={
                    "characters": {"assets/library/characters/hero/reference.png": "library"},
                    "shots": {"assets/shots/s01-001.png": "canonical"},
                },
            )
            with patch("utils.production.ROOT", root):
                from web_server import _generated_asset_paths
                paths = _generated_asset_paths(state)
            self.assertNotIn("assets/library/characters/hero/reference.png", paths)
            self.assertIn("assets/shots/s01-001.png", paths)

    def test_end_to_end_production_flow(self):
        """Full integration: build states -> edit -> quality fail -> review -> retry -> lock -> export."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            state = PipelineState(project_id="e2e", user_prompt="test", script=sample_script())
            manifest = {"shots": {}, "scenes": {}, "characters": {}, "scene_audio": {}}

            with patch("utils.production.load_asset_manifest", return_value=manifest):
                # 1. Build shot states
                shots = build_shot_states(state)
                self.assertIn("S01-001", shots)
                self.assertEqual(shots["S01-001"].status, "pending")

                # 2. Edit shot
                shot = update_script_shot(state, "S01-001", {"dialogue": "updated dialogue"})
                self.assertEqual(shot.script["dialogue"], "updated dialogue")

                # 3. Quality check should fail (missing outputs)
                shot = quality_check_shot(state, "S01-001")
                self.assertEqual(shot.status, "failed")
                self.assertIn("S01-001", state.review_queue)

                # 4. Review as needs_retry
                shot = review_shot(state, "S01-001", "needs_retry", "bad frame")
                self.assertEqual(shot.review_status, "needs_retry")

                # 5. Mark for retry
                queued = mark_failed_for_retry(state)
                self.assertIn("S01-001", queued)
                self.assertEqual(state.shot_states["S01-001"].retry_count, 1)
                self.assertEqual(state.shot_states["S01-001"].status, "pending")

                # 6. Lock the shot
                shot = set_shot_lock(state, "S01-001", True)
                self.assertTrue(shot.locked)

                # 7. Verify locked shot is skipped on retry
                state.shot_states["S01-001"].status = "failed"
                queued = mark_failed_for_retry(state)
                self.assertEqual(queued, [])

                # 8. Export (no video, but should still produce package)
                final = root / "output/final/e2e.mp4"
                final.parent.mkdir(parents=True)
                final.write_bytes(b"video")
                state.final_video = str(final)
                with (
                    patch("utils.production.SUBTITLE_ROOT", root / "output/subtitles"),
                    patch("utils.production.ROOT", root),
                ):
                    export = export_project(state, root)
                self.assertTrue(Path(export["zip"]).exists())


if __name__ == "__main__":
    unittest.main()
