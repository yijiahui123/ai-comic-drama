import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.assets import (
    AssetLibrary,
    canonical_shot,
    normalize_emotion,
    normalize_script_for_generation,
    validate_asset_setup,
)


class AssetUtilityTests(unittest.TestCase):
    def test_normalize_emotion_known_values(self):
        self.assertEqual(normalize_emotion("happy"), "happy")
        self.assertEqual(normalize_emotion("悲伤"), "sad")
        self.assertEqual(normalize_emotion("unknown"), "neutral")

    def test_split_long_shot_duration(self):
        script = {
            "episodes": [
                {
                    "episode": 1,
                    "scenes": [
                        {
                            "scene_id": "S01",
                            "shots": [
                                {
                                    "shot_id": "S01-001",
                                    "duration": 12,
                                    "mood": "happy",
                                    "dialogue": "A：hello",
                                    "motion_prompt": "walk",
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        normalize_script_for_generation(script, max_duration=5, split_long_actions=True)
        shots = script["episodes"][0]["scenes"][0]["shots"]
        self.assertEqual([s["duration"] for s in shots], [5, 5, 2])
        self.assertEqual(shots[0]["shot_id"], "S01-001-01")
        self.assertEqual(shots[1]["dialogue"], "")
        self.assertEqual(shots[0]["emotion"], "happy")

    def test_asset_library_resolves_existing_manifest_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "assets/library/shots/S01.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"png")
            with patch("pathlib.Path.cwd", return_value=root):
                lib = AssetLibrary(
                    manifest={"characters": {}, "scenes": {}, "shots": {"S01": str(image)}}
                )
                self.assertEqual(lib.shot("S01"), image)
                from utils.paths import PROJECT_ROOT
                self.assertEqual(canonical_shot("S01"), PROJECT_ROOT / "assets" / "shots" / "s01.png")

    def test_validate_asset_setup_reports_missing_manifest_asset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "assets.yaml"
            manifest = root / "manifest.yaml"
            config.write_text("style_lora:\n  enabled: false\n", encoding="utf-8")
            manifest.write_text(
                "characters: {}\nscenes: {}\nshots:\n  S01: missing.png\n",
                encoding="utf-8",
            )
            with patch("pathlib.Path.cwd", return_value=root):
                result = validate_asset_setup(config, manifest)
            self.assertFalse(result["ok"])
            self.assertTrue(result["errors"])


if __name__ == "__main__":
    unittest.main()
