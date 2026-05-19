"""Configuration loader with environment-variable overrides.

Usage::

    from utils.config import load_services_config
    cfg = load_services_config()
    # cfg["llm"]["url"] may come from YAML or AI_COMIC_LLM_URL env var
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from utils.paths import PROJECT_ROOT

_ENV_PREFIX = "AI_COMIC"

# Mapping of env var suffixes to nested YAML keys
_ENV_MAP: dict[str, list[str]] = {
    "LLM_URL": ["llm", "url"],
    "LLM_MODEL": ["llm", "model"],
    "LLM_API_KEY": ["llm", "api_key"],
    "COMFYUI_URL": ["comfyui", "url"],
    "CHATTTS_URL": ["chattts", "url"],
    "SADTALKER_URL": ["sadtalker", "url"],
}


def load_services_config(path: Path | None = None) -> dict[str, Any]:
    """Load ``configs/services.yaml`` with env-var overrides.

    Environment variables take precedence over YAML values::

        AI_COMIC_LLM_URL=http://localhost:9000/v1
        AI_COMIC_LLM_API_KEY=my-secret-key
    """
    if path is None:
        path = PROJECT_ROOT / "configs" / "services.yaml"
    data: dict[str, Any] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

    for env_suffix, keys in _ENV_MAP.items():
        value = os.environ.get(f"{_ENV_PREFIX}_{env_suffix}")
        if value is not None:
            node = data
            for key in keys[:-1]:
                node = node.setdefault(key, {})
            node[keys[-1]] = value

    return data
