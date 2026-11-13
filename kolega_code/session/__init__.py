"""Durable session event stream: storage contracts, recording, and projection.

This package is the storage-agnostic half of the session event spine. Host
applications embed this package and persist sessions in their own databases, so
nothing here may assume local disk, an in-process sequence counter, or a single
writer. The filesystem-backed implementation lives in ``kolega_code.cli``.
"""

from .control import ControlChannel, ControlLeaseError, ControlRequest
from .inmemory import InMemoryArtifactStore, InMemorySessionEventStore
from .recording import RecordingConnectionManager, RetentionPolicy
from .runtime import SessionRuntime, SessionRuntimeError, control_channel_for
from .turns import TurnBusyError, TurnController, TurnHooks, TurnOutcome, TurnQueue, TurnResult
from .store import (
    ArtifactStore,
    SessionEventMeta,
    SessionEventStore,
    SessionStoreError,
)

__all__ = [
    "ArtifactStore",
    "ControlChannel",
    "ControlLeaseError",
    "ControlRequest",
    "InMemoryArtifactStore",
    "InMemorySessionEventStore",
    "RecordingConnectionManager",
    "RetentionPolicy",
    "SessionEventMeta",
    "SessionEventStore",
    "SessionRuntime",
    "SessionRuntimeError",
    "SessionStoreError",
    "TurnBusyError",
    "TurnController",
    "TurnHooks",
    "TurnOutcome",
    "TurnQueue",
    "TurnResult",
    "control_channel_for",
]
