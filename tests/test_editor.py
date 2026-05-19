import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from skills.editor.skill import Editor, _seconds_to_ass, _probe_duration


def _make_editor():
    with patch.object(Editor, "_load_transitions", return_value={
        "default": "crossfade",
        "transitions": {
            "crossfade": {"filter": "xfade", "transition": "fade", "duration": 1.0},
            "fade_black": {"filter": "xfade", "transition": "fadeblack", "duration": 1.5},
            "cut": {"filter": None, "duration": 0},
        },
    }), patch.object(Editor, "_load_bgm_config", return_value={
        "enabled": True,
        "directory": "assets/bgm",
        "mood_mapping": {"happy": "upbeat.mp3", "sad": "melancholy.mp3"},
        "default_track": "default.mp3",
    }):
        return Editor(project_id="test")


class EditorHelperTests(unittest.TestCase):
    def test_load_transitions_reads_yaml(self):
        editor = _make_editor()
        self.assertIn("crossfade", editor._transitions.get("transitions", {}))

    def test_iter_shots_flattens_scenes(self):
        episode = {
            "scenes": [
                {"shots": [{"shot_id": "A"}, {"shot_id": "B"}]},
                {"shots": [{"shot_id": "C"}]},
            ]
        }
        shots = list(Editor._iter_shots(episode))
        self.assertEqual(len(shots), 3)
        self.assertEqual(shots[0]["shot_id"], "A")
        self.assertEqual(shots[2]["shot_id"], "C")

    def test_find_clip_prefers_lipsync(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("skills.editor.skill._OUTPUT_LIPSYNC", Path(tmp) / "lipsync"), \
                 patch("skills.editor.skill._OUTPUT_VIDEOS", Path(tmp) / "videos"):
                (Path(tmp) / "lipsync").mkdir()
                (Path(tmp) / "videos").mkdir()
                (Path(tmp) / "lipsync" / "S01_lipsync.mp4").write_bytes(b"lip")
                (Path(tmp) / "videos" / "S01.mp4").write_bytes(b"vid")
                result = Editor._find_clip({"shot_id": "S01"})
                self.assertIn("lipsync", str(result))

    def test_find_clip_returns_video_when_no_lipsync(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("skills.editor.skill._OUTPUT_LIPSYNC", Path(tmp) / "lipsync"), \
                 patch("skills.editor.skill._OUTPUT_VIDEOS", Path(tmp) / "videos"):
                (Path(tmp) / "lipsync").mkdir()
                (Path(tmp) / "videos").mkdir()
                (Path(tmp) / "videos" / "S02.mp4").write_bytes(b"vid")
                result = Editor._find_clip({"shot_id": "S02"})
                self.assertIn("videos", str(result))

    def test_find_clip_returns_none_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("skills.editor.skill._OUTPUT_LIPSYNC", Path(tmp) / "lipsync"), \
                 patch("skills.editor.skill._OUTPUT_VIDEOS", Path(tmp) / "videos"):
                (Path(tmp) / "lipsync").mkdir()
                (Path(tmp) / "videos").mkdir()
                result = Editor._find_clip({"shot_id": "MISSING"})
                self.assertIsNone(result)

    def test_generate_subtitles_produces_ass(self):
        editor = _make_editor()
        shots = [
            {"duration": 4, "dialogue": "Narrator: Hello world"},
            {"duration": 3, "dialogue": "Goodbye"},
        ]
        import asyncio
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(editor._generate_subtitles(shots, 1))
        loop.close()
        self.assertIsNotNone(result)
        content = result.read_text(encoding="utf-8")
        self.assertIn("Hello world", content)
        self.assertIn("Goodbye", content)
        self.assertIn("Dialogue:", content)
        result.unlink(missing_ok=True)

    def test_generate_subtitles_skips_empty_dialogue(self):
        editor = _make_editor()
        shots = [{"duration": 4, "dialogue": ""}, {"duration": 3, "dialogue": "  "}]
        import asyncio
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(editor._generate_subtitles(shots, 1))
        loop.close()
        self.assertIsNone(result)

    def test_seconds_to_ass_format(self):
        self.assertEqual(_seconds_to_ass(0), "0:00:00.00")
        self.assertEqual(_seconds_to_ass(61.5), "0:01:01.50")
        self.assertEqual(_seconds_to_ass(3661.25), "1:01:01.25")

    def test_probe_duration_parses_ffprobe_output(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"format": {"duration": "12.345"}})
        with patch("skills.editor.skill.subprocess.run", return_value=mock_result):
            d = _probe_duration(Path("test.mp4"))
            self.assertAlmostEqual(d, 12.345, places=2)

    def test_probe_duration_returns_zero_on_error(self):
        mock_result = MagicMock()
        mock_result.returncode = 1
        with patch("skills.editor.skill.subprocess.run", return_value=mock_result):
            d = _probe_duration(Path("bad.mp4"))
            self.assertEqual(d, 0.0)


class EditorBGMTests(unittest.TestCase):
    def test_select_bgm_from_mood_mapping(self):
        editor = _make_editor()
        episode = {"scenes": [{"shots": [
            {"shot_id": "A", "emotion": "happy"},
            {"shot_id": "B", "emotion": "happy"},
            {"shot_id": "C", "emotion": "sad"},
        ]}]}
        with patch("skills.editor.skill.Path.exists", return_value=True):
            result = editor._select_bgm(episode)
            self.assertIn("upbeat", str(result))

    def test_select_bgm_returns_none_when_disabled(self):
        editor = _make_editor()
        editor._bgm_config = {"enabled": False}
        result = editor._select_bgm({"scenes": []})
        self.assertIsNone(result)


class EditorConcatTests(unittest.TestCase):
    def test_concat_clips_single_clip_passthrough(self):
        editor = _make_editor()
        import asyncio
        with tempfile.TemporaryDirectory() as tmp:
            clip = Path(tmp) / "clip.mp4"
            clip.write_bytes(b"fake video data")
            loop = asyncio.new_event_loop()
            result = loop.run_until_complete(editor._concat_clips([clip], 1))
            loop.close()
            self.assertIsNotNone(result)
            self.assertTrue(result.exists())

    def test_concat_clips_builds_xfade_filter(self):
        editor = _make_editor()
        import asyncio
        mock_ffmpeg = MagicMock(return_value=True)
        mock_probe = MagicMock(return_value=5.0)
        with tempfile.TemporaryDirectory() as tmp:
            clip1 = Path(tmp) / "c1.mp4"
            clip2 = Path(tmp) / "c2.mp4"
            clip1.write_bytes(b"a")
            clip2.write_bytes(b"b")
            with patch("skills.editor.skill._run_ffmpeg", mock_ffmpeg), \
                 patch("skills.editor.skill._probe_duration", mock_probe):
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    editor._concat_clips([clip1, clip2], 1, ["crossfade"])
                )
                loop.close()
                self.assertTrue(mock_ffmpeg.called)
                args = mock_ffmpeg.call_args[0][0]
                self.assertIn("-filter_complex", args)
                fc = args[args.index("-filter_complex") + 1]
                self.assertIn("xfade", fc)


    def test_concat_clips_three_clips_builds_two_xfade_nodes(self):
        editor = _make_editor()
        import asyncio
        mock_ffmpeg = MagicMock(return_value=True)
        mock_probe = MagicMock(return_value=5.0)
        with tempfile.TemporaryDirectory() as tmp:
            clips = []
            for i in range(3):
                clip = Path(tmp) / f"c{i}.mp4"
                clip.write_bytes(b"x")
                clips.append(clip)
            with patch("skills.editor.skill._run_ffmpeg", mock_ffmpeg), \
                 patch("skills.editor.skill._probe_duration", mock_probe):
                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    editor._concat_clips(clips, 1, ["crossfade"])
                )
                loop.close()
                self.assertTrue(mock_ffmpeg.called)
                args = mock_ffmpeg.call_args[0][0]
                fc = args[args.index("-filter_complex") + 1]
                # Two xfade nodes for three clips
                self.assertEqual(fc.count("xfade"), 2)


class EditorWhisperTests(unittest.TestCase):
    def test_whisper_fallback_on_import_error(self):
        import asyncio
        editor = _make_editor()

        async def _run():
            with patch.dict("sys.modules", {"whisper": None}):
                result = await editor._transcribe_with_whisper(Path("fake.wav"))
                return result

        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(_run())
        loop.close()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
