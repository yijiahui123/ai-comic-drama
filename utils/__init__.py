"""Utility modules for the AI Comic Drama pipeline."""

import re


def slugify(text: str) -> str:
    """Convert *text* into a safe file-system-friendly slug.

    Args:
        text: Input string.

    Returns:
        Lower-cased, alphanumeric-and-hyphen-only slug (max 64 chars).
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text)
    return text[:64]
