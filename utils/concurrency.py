import asyncio
from typing import Any, Callable, Coroutine, TypeVar

T = TypeVar("T")

# Global semaphore to restrict pipeline runs across the entire application.
# Ensures that only one memory/VRAM-heavy pipeline stage runs at a time.
_pipeline_semaphore = asyncio.Semaphore(1)


async def with_pipeline_lock(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine while holding the global pipeline lock.
    
    This ensures that multiple projects do not run the pipeline concurrently,
    preventing Out-Of-Memory (OOM) errors on systems with limited VRAM.
    """
    async with _pipeline_semaphore:
        return await coro
