"""ScriptWriter Skill.

Calls the Ollama HTTP API to produce a structured JSON script from a natural-language
user description.  Generation is split into two phases:

1. **Outline phase** – produce a high-level story outline (episodes, scenes).
2. **Scene expansion phase** – expand each scene into detailed shot descriptions.

The final output is a validated script dictionary conforming to the schema defined
in ``utils/validators.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import aiohttp

from utils.logger import get_logger
from utils.validators import validate_script

logger = get_logger(__name__)

# Directory containing system prompt templates
_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Ollama endpoint paths
_OLLAMA_CHAT_PATH = "/api/chat"

# Retry configuration
_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds


def _load_prompt(filename: str) -> str:
    """Load a prompt template from the ``prompts/`` directory.

    Args:
        filename: Name of the template file (e.g. ``system_outline.txt``).

    Returns:
        File contents as a string.
    """
    path = _PROMPTS_DIR / filename
    return path.read_text(encoding="utf-8")


def _extract_json(text: str) -> dict[str, Any]:
    """Extract and parse the first JSON object found in *text*.

    Args:
        text: Raw text that may contain prose followed by a JSON block.

    Returns:
        Parsed Python dictionary.

    Raises:
        ValueError: If no valid JSON object can be found.
    """
    # Try to find a JSON block (```json ... ``` or bare { ... })
    json_block = re.search(r"```(?:json)?\s*(\{.*?})\s*```", text, re.DOTALL)
    if json_block:
        return json.loads(json_block.group(1))

    # Fallback: find the first '{' and attempt to parse from there
    start = text.find("{")
    if start != -1:
        # Walk forward to find matching closing brace
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(text[start : i + 1])

    raise ValueError("No JSON object found in LLM response.")


class ScriptWriter:
    """Generates structured comic-drama scripts using an Ollama-hosted LLM.

    Attributes:
        ollama_url: Base URL of the Ollama service.
        model: Ollama model identifier.
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.1:70b-instruct-q4_K_M",
    ) -> None:
        """
        Args:
            ollama_url: Base URL of the Ollama API server.
            model: Model name as registered in Ollama (e.g. ``qwen2.5:72b-q4_K_M``).
        """
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self._outline_prompt = _load_prompt("system_outline.txt")
        self._scene_prompt = _load_prompt("system_scene.txt")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(self, user_description: str) -> dict[str, Any]:
        """Generate a complete, validated script from a user description.

        This is the main entry point.  It executes the two-phase generation
        (outline → scene expansion) and validates the result.

        Args:
            user_description: Natural-language description of the desired comic drama
                (e.g. "写一个赛博朋克风格的3分钟漫剧，主角是黑客少女").

        Returns:
            A validated script dictionary.

        Raises:
            RuntimeError: If Ollama is unreachable or the script cannot be validated
                after ``_MAX_RETRIES`` attempts.
        """
        logger.info("ScriptWriter: starting generation for prompt: %r", user_description[:80])

        # Phase 1 – outline
        outline = await self._generate_outline(user_description)
        logger.info("ScriptWriter: outline generated — %d episode(s)", len(outline.get("episodes", [])))

        # Phase 2 – expand each scene
        script = await self._expand_scenes(outline, user_description)

        # Validate
        is_valid, errors = validate_script(script)
        if not is_valid:
            logger.warning("Script validation failed; attempting auto-repair…")
            script = self._auto_repair(script, errors)
            is_valid, errors = validate_script(script)
            if not is_valid:
                raise RuntimeError(
                    f"Script validation failed after repair: {errors[:5]}"
                )

        logger.info("ScriptWriter: generation complete — title=%r", script.get("title"))
        return script

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _chat(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
    ) -> str:
        """Send a chat request to Ollama and return the assistant's reply.

        Args:
            system_prompt: System-level instruction.
            user_message: User turn content.
            temperature: Sampling temperature.

        Returns:
            Assistant reply text.

        Raises:
            RuntimeError: After all retries are exhausted.
        """
        import asyncio  # noqa: PLC0415

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }

        last_exc: Exception = RuntimeError("No attempts made")
        delay = _RETRY_DELAY
        timeout = aiohttp.ClientTimeout(total=300, connect=10)

        for attempt in range(1, _MAX_RETRIES + 2):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        f"{self.ollama_url}{_OLLAMA_CHAT_PATH}",
                        json=payload,
                    ) as resp:
                        resp.raise_for_status()
                        data = await resp.json()
                        return data["message"]["content"]
            except (aiohttp.ClientError, asyncio.TimeoutError, KeyError) as exc:
                last_exc = exc
                if attempt <= _MAX_RETRIES:
                    logger.warning(
                        "Ollama request failed (attempt %d/%d): %s — retrying in %.1fs",
                        attempt,
                        _MAX_RETRIES + 1,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2

        raise RuntimeError(f"Ollama unreachable after {_MAX_RETRIES + 1} attempts: {last_exc}") from last_exc

    async def _generate_outline(self, user_description: str) -> dict[str, Any]:
        """Phase 1: generate a high-level story outline.

        Args:
            user_description: Raw user prompt.

        Returns:
            Outline dictionary with ``title``, ``style``, and ``episodes`` list
            (each episode contains scene stubs without individual shots).
        """
        response = await self._chat(self._outline_prompt, user_description)
        try:
            outline = _extract_json(response)
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Failed to parse outline JSON: {exc}\nRaw response:\n{response[:500]}") from exc
        return outline

    async def _expand_scenes(
        self, outline: dict[str, Any], original_prompt: str
    ) -> dict[str, Any]:
        """Phase 2: expand each scene in the outline into detailed shot lists.

        Args:
            outline: Outline dictionary from phase 1.
            original_prompt: Original user description (for context).

        Returns:
            Complete script dictionary with all shots filled in.
        """
        script = dict(outline)
        script.setdefault("style", "anime")
        expanded_episodes = []

        for episode in outline.get("episodes", []):
            expanded_scenes = []
            for scene in episode.get("scenes", []):
                user_msg = (
                    f"Original story request: {original_prompt}\n\n"
                    f"Script style: {script.get('style', 'anime')}\n\n"
                    f"Please expand the following scene into detailed shots:\n"
                    f"{json.dumps(scene, ensure_ascii=False, indent=2)}"
                )
                response = await self._chat(self._scene_prompt, user_msg, temperature=0.8)
                try:
                    expanded_scene = _extract_json(response)
                except (ValueError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "Failed to parse scene JSON for scene %s: %s — keeping stub",
                        scene.get("scene_id", "?"),
                        exc,
                    )
                    expanded_scene = scene
                expanded_scenes.append(expanded_scene)

            ep_copy = dict(episode)
            ep_copy["scenes"] = expanded_scenes
            expanded_episodes.append(ep_copy)

        script["episodes"] = expanded_episodes
        return script

    @staticmethod
    def _auto_repair(script: dict[str, Any], errors: list[str]) -> dict[str, Any]:
        """Attempt lightweight structural repairs on an invalid script.

        Handles common issues such as missing ``style`` field or shots with
        empty ``visual_prompt``.

        Args:
            script: Script dictionary to repair in-place.
            errors: List of validation error messages.

        Returns:
            Repaired script dictionary.
        """
        # Ensure top-level style field exists
        if "style" not in script:
            script["style"] = "anime"

        for episode in script.get("episodes", []):
            for scene in episode.get("scenes", []):
                scene.setdefault("location", "未知场景")
                for shot in scene.get("shots", []):
                    shot.setdefault("type", "中景")
                    shot.setdefault("characters", [])
                    shot.setdefault("dialogue", "")
                    shot.setdefault("camera_move", "固定")
                    shot.setdefault("duration", 4)
                    shot.setdefault("mood", "neutral")
                    if not shot.get("visual_prompt"):
                        shot["visual_prompt"] = (
                            f"{shot.get('type', 'medium shot')}, anime style, "
                            f"{scene.get('location', 'interior')}"
                        )

        return script
