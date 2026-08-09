#!/usr/bin/env python3
"""
Storage Cleanup Script.

Cleans up intermediate assets and generated files older than a specified TTL.
Intended to be run periodically (e.g., via cron) in production environments.
"""

import argparse
import shutil
import time
from pathlib import Path

from utils.logger import get_logger
from utils.paths import PROJECT_ROOT

logger = get_logger(__name__)

# Directories that typically contain intermediate files safe to delete after TTL
CLEANUP_TARGETS = [
    PROJECT_ROOT / "output" / "audio",
    PROJECT_ROOT / "output" / "continuity",
    PROJECT_ROOT / "output" / "lipsync",
    PROJECT_ROOT / "output" / "videos",
    PROJECT_ROOT / "assets" / "shots",
]

def cleanup_storage(ttl_days: int = 7, dry_run: bool = False) -> None:
    """Delete files in target directories older than `ttl_days`.

    Args:
        ttl_days: Time-to-live in days.
        dry_run: If True, do not actually delete anything.
    """
    now = time.time()
    ttl_seconds = ttl_days * 86400

    deleted_count = 0
    freed_bytes = 0

    for target_dir in CLEANUP_TARGETS:
        if not target_dir.exists() or not target_dir.is_dir():
            continue

        for path in target_dir.rglob("*"):
            if path.is_file():
                # Check file age (mtime)
                mtime = path.stat().st_mtime
                age_seconds = now - mtime
                if age_seconds > ttl_seconds:
                    size = path.stat().st_size
                    if not dry_run:
                        try:
                            path.unlink()
                            deleted_count += 1
                            freed_bytes += size
                        except OSError as e:
                            logger.error("Failed to delete %s: %s", path, e)
                    else:
                        logger.info("[DRY RUN] Would delete: %s (%.1f days old)", path, age_seconds / 86400)
                        deleted_count += 1
                        freed_bytes += size

    if dry_run:
        logger.info("[DRY RUN] Summary: Would delete %d files, freeing %.2f MB.", deleted_count, freed_bytes / (1024 * 1024))
    else:
        logger.info("Cleanup complete: Deleted %d files, freed %.2f MB.", deleted_count, freed_bytes / (1024 * 1024))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleanup old intermediate pipeline files.")
    parser.add_argument("--ttl", type=int, default=7, help="Time to live in days (default: 7)")
    parser.add_argument("--dry-run", action="store_true", help="Perform a dry run without deleting files")
    args = parser.parse_args()

    cleanup_storage(ttl_days=args.ttl, dry_run=args.dry_run)
