"""Owner-private discovery registry for active browser runtimes."""

from __future__ import annotations

import errno
import contextlib
import hmac
import json
import os
import re
import stat
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from .protocol import MAX_SAFE_INTEGER, PROTOCOL_VERSION

MAX_DESCRIPTOR_BYTES = 65_536
RuntimeTransport = Literal["unix"]

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{32,256}$")
_ORIGIN_RE = re.compile(r"^chrome-extension://[a-p]{32}/$")
_DESCRIPTOR_KEYS = frozenset(
    {
        "protocol_version",
        "runtime_id",
        "session_id",
        "transport",
        "endpoint",
        "token",
        "pid",
        "created_at_ms",
        "expires_at_ms",
        "extension_origin",
    }
)


class DescriptorError(RuntimeError):
    """A safe-to-report runtime descriptor failure."""


class DescriptorSecurityError(DescriptorError):
    """Descriptor ownership, type, or permissions are unsafe."""


class DescriptorNotFoundError(DescriptorError):
    """No descriptor exists for the requested runtime."""


class StaleDescriptorError(DescriptorError):
    """A descriptor no longer identifies a live runtime."""


def validate_extension_origin(origin: object) -> str:
    """Return one exact Chrome extension origin."""
    if not isinstance(origin, str) or not _ORIGIN_RE.fullmatch(origin):
        raise DescriptorError("Chrome extension origin is invalid")
    return origin


def validate_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise DescriptorError(f"{label} is invalid")
    return value


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_endpoint(transport: object, endpoint: object) -> tuple[RuntimeTransport, str]:
    if transport != "unix":
        raise DescriptorError("Runtime descriptor transport is unsupported")
    if not isinstance(endpoint, str) or not endpoint or "\0" in endpoint:
        raise DescriptorError("Runtime descriptor endpoint is invalid")
    if not Path(endpoint).is_absolute():
        raise DescriptorError("Runtime descriptor endpoint is invalid")
    return cast(RuntimeTransport, transport), endpoint


@dataclass(frozen=True, slots=True)
class RuntimeDescriptor:
    """Private connection data for one active runtime."""

    runtime_id: str
    session_id: str
    transport: RuntimeTransport
    endpoint: str
    token: str
    pid: int
    created_at_ms: int
    expires_at_ms: int
    extension_origin: str
    protocol_version: int = PROTOCOL_VERSION

    def __post_init__(self) -> None:
        RuntimeDescriptor.from_mapping(self.to_mapping())

    @classmethod
    def from_mapping(cls, value: object) -> RuntimeDescriptor:
        if not isinstance(value, dict) or set(value) != _DESCRIPTOR_KEYS:
            raise DescriptorError("Runtime descriptor has an invalid schema")
        raw = cast(dict[str, Any], value)
        if not _is_int(raw["protocol_version"]) or raw["protocol_version"] != PROTOCOL_VERSION:
            raise DescriptorError("Runtime descriptor protocol is unsupported")
        runtime_id = validate_identifier(raw["runtime_id"], "runtime ID")
        session_id = validate_identifier(raw["session_id"], "session ID")
        transport, endpoint = _validate_endpoint(raw["transport"], raw["endpoint"])
        token = raw["token"]
        if not isinstance(token, str) or not _TOKEN_RE.fullmatch(token):
            raise DescriptorError("Runtime descriptor credential is invalid")
        pid = raw["pid"]
        created = raw["created_at_ms"]
        expires = raw["expires_at_ms"]
        if not _is_int(pid) or pid <= 0:
            raise DescriptorError("Runtime descriptor process ID is invalid")
        if (
            not _is_int(created)
            or not _is_int(expires)
            or created <= 0
            or expires <= created
            or expires > MAX_SAFE_INTEGER
        ):
            raise DescriptorError("Runtime descriptor lifetime is invalid")
        origin = validate_extension_origin(raw["extension_origin"])

        instance = object.__new__(cls)
        object.__setattr__(instance, "runtime_id", runtime_id)
        object.__setattr__(instance, "session_id", session_id)
        object.__setattr__(instance, "transport", transport)
        object.__setattr__(instance, "endpoint", endpoint)
        object.__setattr__(instance, "token", token)
        object.__setattr__(instance, "pid", pid)
        object.__setattr__(instance, "created_at_ms", created)
        object.__setattr__(instance, "expires_at_ms", expires)
        object.__setattr__(instance, "extension_origin", origin)
        object.__setattr__(instance, "protocol_version", PROTOCOL_VERSION)
        return instance

    def to_mapping(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "runtime_id": self.runtime_id,
            "session_id": self.session_id,
            "transport": self.transport,
            "endpoint": self.endpoint,
            "token": self.token,
            "pid": self.pid,
            "created_at_ms": self.created_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "extension_origin": self.extension_origin,
        }

    def accepts_origin(self, origin: str) -> bool:
        return hmac.compare_digest(self.extension_origin, origin)


def _owner_uid() -> int:
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        raise DescriptorSecurityError("POSIX descriptor ownership checks are unavailable")
    return cast(int, getuid())


def _verify_private_stat(file_stat: os.stat_result, *, directory: bool) -> None:
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(file_stat.st_mode):
        raise DescriptorSecurityError("Runtime registry object has an unsafe type")
    if file_stat.st_uid != _owner_uid() or stat.S_IMODE(file_stat.st_mode) & 0o077:
        raise DescriptorSecurityError("Runtime registry ownership or permissions are unsafe")


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno == errno.EPERM
    return True


class RuntimeDescriptorRegistry:
    """Atomic per-user runtime descriptor storage."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self._ensure_private_root()

    def _ensure_private_root(self) -> None:
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=False)
        except FileExistsError:
            pass
        _verify_private_stat(self.root.lstat(), directory=True)
        os.chmod(self.root, 0o700)

    def _path_for(self, runtime_id: str) -> Path:
        return self.root / f"{validate_identifier(runtime_id, 'runtime ID')}.json"

    def register(self, descriptor: RuntimeDescriptor) -> Path:
        if descriptor.expires_at_ms <= int(time.time() * 1000):
            raise StaleDescriptorError("Runtime descriptor is stale")
        payload = json.dumps(
            descriptor.to_mapping(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(payload) > MAX_DESCRIPTOR_BYTES:
            raise DescriptorError("Runtime descriptor exceeds the size limit")
        destination = self._path_for(descriptor.runtime_id)
        fd, temporary_name = tempfile.mkstemp(prefix=".runtime-", dir=self.root)
        temporary = Path(temporary_name)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb", closefd=True) as file:
                fd = -1
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination)
            self._fsync_root()
        finally:
            if fd >= 0:
                os.close(fd)
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()
        return destination

    def read(
        self,
        runtime_id: str,
        *,
        check_stale: bool = True,
        now_ms: int | None = None,
    ) -> RuntimeDescriptor:
        path = self._path_for(runtime_id)
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            fd = os.open(path, flags)
        except FileNotFoundError:
            raise DescriptorNotFoundError("Runtime descriptor was not found") from None
        except OSError:
            raise DescriptorSecurityError("Runtime descriptor could not be opened safely") from None
        try:
            file_stat = os.fstat(fd)
            _verify_private_stat(file_stat, directory=False)
            with os.fdopen(fd, "rb", closefd=True) as file:
                fd = -1
                payload = file.read(MAX_DESCRIPTOR_BYTES + 1)
        finally:
            if fd >= 0:
                os.close(fd)
        if len(payload) > MAX_DESCRIPTOR_BYTES:
            raise DescriptorError("Runtime descriptor exceeds the size limit")
        try:
            raw = json.loads(payload.decode("utf-8"), object_pairs_hook=self._reject_duplicate_keys)
        except DescriptorError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
            raise DescriptorError("Runtime descriptor is malformed") from None
        descriptor = RuntimeDescriptor.from_mapping(raw)
        if descriptor.runtime_id != runtime_id:
            raise DescriptorError("Runtime descriptor identity does not match its filename")
        if check_stale and self._is_stale(descriptor, now_ms=now_ms):
            self._unlink_if_same(path, file_stat)
            raise StaleDescriptorError("Runtime descriptor is stale")
        return descriptor

    def unregister(self, runtime_id: str, *, token: str) -> bool:
        try:
            descriptor = self.read(runtime_id, check_stale=False)
        except DescriptorNotFoundError:
            return False
        if not hmac.compare_digest(descriptor.token, token):
            return False
        path = self._path_for(runtime_id)
        try:
            file_stat = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return False
        self._unlink_if_same(path, file_stat)
        return True

    def list_active(self) -> list[RuntimeDescriptor]:
        active: list[RuntimeDescriptor] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                active.append(self.read(path.stem))
            except DescriptorError:
                continue
        return active

    @staticmethod
    def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in pairs:
            if key in value:
                raise DescriptorError("Runtime descriptor is malformed")
            value[key] = item
        return value

    def _is_stale(self, descriptor: RuntimeDescriptor, *, now_ms: int | None = None) -> bool:
        current = int(time.time() * 1000) if now_ms is None else now_ms
        if descriptor.expires_at_ms <= current or not _pid_is_alive(descriptor.pid):
            return True
        try:
            endpoint_stat = Path(descriptor.endpoint).stat(follow_symlinks=False)
        except OSError:
            return True
        return (
            not stat.S_ISSOCK(endpoint_stat.st_mode)
            or endpoint_stat.st_uid != _owner_uid()
            or bool(stat.S_IMODE(endpoint_stat.st_mode) & 0o077)
        )

    @staticmethod
    def _unlink_if_same(path: Path, expected_stat: os.stat_result) -> None:
        try:
            current = path.stat(follow_symlinks=False)
            if current.st_dev == expected_stat.st_dev and current.st_ino == expected_stat.st_ino:
                path.unlink()
        except FileNotFoundError:
            return

    def _fsync_root(self) -> None:
        try:
            directory_fd = os.open(self.root, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        except OSError:
            pass
        finally:
            os.close(directory_fd)
