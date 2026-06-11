"""Utility functions for event loop cleanup in task workers."""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


def cleanup_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Clean up a manually managed event loop."""
    try:
        tasks_to_cancel = asyncio.all_tasks(loop)
        if tasks_to_cancel:
            for task in tasks_to_cancel:
                task.cancel()
            try:
                loop.run_until_complete(asyncio.gather(*tasks_to_cancel, return_exceptions=True))
            except RuntimeError as error:
                logger.warning("Could not wait for task cancellation: %s", error)

        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except RuntimeError as error:
            logger.warning("Could not shut down async generators: %s", error)

        try:
            loop.run_until_complete(asyncio.sleep(0))
        except RuntimeError as error:
            logger.warning("Could not process scheduled callbacks: %s", error)

        pending = asyncio.all_tasks(loop)
        if pending:
            try:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except RuntimeError as error:
                logger.warning("Could not wait for remaining tasks: %s", error)

    finally:
        if not loop.is_closed():
            loop.close()

