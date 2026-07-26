"""Durable session event stream: storage contracts, recording, and projection.

This package is the storage-agnostic half of the session event spine. Host
applications embed this package and persist sessions in their own databases, so
nothing here may assume local disk, an in-process sequence counter, or a single
writer. The filesystem-backed implementation lives in ``kolega_code.cli``.
"""

from .inmemory import InMemoryArtifactStore, InMemorySessionEventStore
from .recording import RecordingConnectionManager, RetentionPolicy
from .store import (
    ArtifactStore,
    SessionEventMeta,
    SessionEventStore,
    SessionStoreError,
)

__all__ = [
    "ArtifactStore",
    "InMemoryArtifactStore",
    "InMemorySessionEventStore",
    "RecordingConnectionManager",
    "RetentionPolicy",
    "SessionEventMeta",
    "SessionEventStore",
    "SessionStoreError",
]
