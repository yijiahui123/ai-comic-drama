import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.state import PipelineState, Stage, StageResult


class OrchestratorTests(unittest.TestCase):
    def test_new_creates_fresh_state(self):
        orch = PipelineOrchestrator.new("test prompt")
        self.assertEqual(orch.state.user_prompt, "test prompt")
        self.assertEqual(orch.state.current_stage, Stage.INIT)
        self.assertTrue(len(orch.state.project_id) >= 4)

    def test_resume_loads_existing_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp)
            state = PipelineState(project_id="abc123", user_prompt="p", current_stage=Stage.SCRIPTING)
            state.save(state_dir)
            loaded = PipelineState.load("abc123", state_dir)
            self.assertEqual(loaded.project_id, "abc123")
            self.assertEqual(loaded.current_stage, Stage.SCRIPTING)

    def test_stage_progression_skips_done_stages(self):
        orch = PipelineOrchestrator.new("test")
        orch.state.stages[Stage.SCRIPTING.value] = StageResult(
            stage=Stage.SCRIPTING, status="done"
        )
        orch.state.current_stage = Stage.SCRIPTING
        self.assertTrue(orch.state.is_stage_done(Stage.SCRIPTING))
        self.assertFalse(orch.state.is_stage_done(Stage.ASSET_GEN))

    def test_run_stage_handles_exception(self):
        orch = PipelineOrchestrator.new("test")
        orch.state.current_stage = Stage.SCRIPTING
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(PipelineState, "save"):
                import asyncio

                async def _fail(**kw):
                    raise RuntimeError("boom")

                loop = asyncio.new_event_loop()
                result = loop.run_until_complete(
                    orch._run_stage(Stage.SCRIPTING, _fail)
                )
                loop.close()
                self.assertIsNone(result)
                self.assertEqual(orch.state.current_stage, Stage.ERROR)

    def test_reset_from_stage_clears_downstream(self):
        orch = PipelineOrchestrator.new("test")
        orch.state.script = {"episodes": []}
        orch.state.asset_manifest = {"characters": ["a.png"]}
        orch.state.stages[Stage.SCRIPTING.value] = StageResult(stage=Stage.SCRIPTING, status="done")
        orch.state.stages[Stage.ASSET_GEN.value] = StageResult(stage=Stage.ASSET_GEN, status="done")
        with patch.object(PipelineState, "save"):
            orch.reset_from_stage(Stage.ASSET_GEN)
        # SCRIPTING is upstream — stays
        self.assertIn(Stage.SCRIPTING.value, orch.state.stages)
        # ASSET_GEN and downstream — cleared
        self.assertNotIn(Stage.ASSET_GEN.value, orch.state.stages)
        self.assertIsNotNone(orch.state.script)
        self.assertEqual(orch.state.asset_manifest, {})

    def test_reset_from_scripting_clears_everything(self):
        orch = PipelineOrchestrator.new("test")
        orch.state.script = {"episodes": []}
        orch.state.asset_manifest = {"x": ["y"]}
        orch.state.video_manifest = {"v": ["z"]}
        orch.state.final_video = "out.mp4"
        with patch.object(PipelineState, "save"):
            orch.reset_from_stage(Stage.SCRIPTING)
        self.assertIsNone(orch.state.script)
        self.assertEqual(orch.state.asset_manifest, {})
        self.assertEqual(orch.state.video_manifest, {})
        self.assertIsNone(orch.state.final_video)

    def test_status_summary_includes_all_stages(self):
        orch = PipelineOrchestrator.new("test prompt for summary")
        orch.state.stages[Stage.SCRIPTING.value] = StageResult(
            stage=Stage.SCRIPTING, status="done", elapsed_seconds=12.3
        )
        summary = orch.status()
        self.assertIn("SCRIPTING", summary)
        self.assertIn("DONE", summary)
        self.assertIn("12.3", summary)


if __name__ == "__main__":
    unittest.main()
