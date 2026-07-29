"""Safe per-user Chrome Native Messaging host installation."""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Mapping, get_args

from .native_host import (
    DEFAULT_MAX_RELAY_PENDING,
    DEFAULT_MAX_RUNTIMES,
    NativeHostConfigurationError,
    default_host_config_path,
    load_host_config,
)
from .registry import validate_extension_origin
from .runtime import UnsupportedRuntimeTransportError, ensure_runtime_transport_supported

NATIVE_HOST_NAME = "ai.kolega.browser"
CHROME_EXTENSION_ID = "edihigldhbmimflgjkohkgnjefmhngdn"
CHROME_WEB_STORE_URL = f"https://chromewebstore.google.com/detail/{CHROME_EXTENSION_ID}"
BrowserExtensionChannel = Literal["production", "beta", "dev"]
BROWSER_EXTENSION_CHANNELS: Final[tuple[BrowserExtensionChannel, ...]] = get_args(BrowserExtensionChannel)
DEFAULT_BROWSER_EXTENSION_CHANNEL: Final[BrowserExtensionChannel] = "production"
CHANNEL_EXTENSION_IDS: Final[Mapping[BrowserExtensionChannel, str | None]] = {
    "production": CHROME_EXTENSION_ID,
    "beta": None,
    "dev": None,
}
_EXTENSION_ID_RE = re.compile(r"^[a-p]{32}$")
_MANIFEST_KEYS = frozenset({"name", "description", "path", "type", "allowed_origins"})


class NativeHostInstallError(RuntimeError):
    """The native host cannot be installed or removed safely."""


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    path: Path
    content: bytes | None
    mode: int | None


@dataclass(frozen=True, slots=True)
class NativeHostStatus:
    manifest_path: Path
    installed: bool
    valid: bool
    host_path: Path | None
    extension_id: str
    detail: str
    channel: BrowserExtensionChannel = DEFAULT_BROWSER_EXTENSION_CHANNEL
    configured_extension_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "installed": self.installed,
            "valid": self.valid,
            "host_path": str(self.host_path) if self.host_path is not None else None,
            "extension_id": self.extension_id,
            "configured_extension_id": self.configured_extension_id,
            "channel": self.channel,
            "detail": self.detail,
        }


def validate_extension_id(extension_id: str) -> str:
    if not _EXTENSION_ID_RE.fullmatch(extension_id):
        raise NativeHostInstallError("Chrome extension ID must be 32 lowercase letters in the range a-p.")
    return extension_id


def resolve_channel_extension_id(
    *,
    channel: str = DEFAULT_BROWSER_EXTENSION_CHANNEL,
    extension_id: str | None = None,
) -> tuple[BrowserExtensionChannel, str]:
    if channel not in BROWSER_EXTENSION_CHANNELS:
        raise NativeHostInstallError(
            f"Browser extension channel must be one of: {', '.join(BROWSER_EXTENSION_CHANNELS)}."
        )
    selected: BrowserExtensionChannel = channel
    compiled = CHANNEL_EXTENSION_IDS[selected]
    if compiled is not None:
        if extension_id is not None and validate_extension_id(extension_id) != compiled:
            raise NativeHostInstallError(
                f"The {selected} channel uses extension ID {compiled}; a conflicting ID is not allowed."
            )
        return selected, compiled
    if extension_id is None:
        raise NativeHostInstallError(f"The {selected} channel requires an explicit extension ID.")
    return selected, validate_extension_id(extension_id)


def channel_web_store_url(channel: BrowserExtensionChannel) -> str | None:
    return CHROME_WEB_STORE_URL if channel == "production" else None


def chrome_extension_origin(extension_id: str = CHROME_EXTENSION_ID) -> str:
    return validate_extension_origin(f"chrome-extension://{validate_extension_id(extension_id)}/")


def default_state_dir(*, platform: str | None = None, env: Mapping[str, str] | None = None) -> Path:
    selected = platform or sys.platform
    if selected != "darwin":
        raise UnsupportedRuntimeTransportError("Chrome browser integration is supported only on macOS.")
    values = os.environ if env is None else env
    configured = values.get("KOLEGA_CODE_STATE_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Library" / "Application Support" / "kolega-code"


def chrome_native_host_manifest_path(
    *,
    platform: str | None = None,
    home: Path | None = None,
) -> Path:
    selected = platform or sys.platform
    if selected != "darwin":
        raise UnsupportedRuntimeTransportError("Chrome browser integration is supported only on macOS.")
    user_home = (home or Path.home()).expanduser().resolve()
    root = user_home / "Library" / "Application Support" / "Google" / "Chrome"
    return root / "NativeMessagingHosts" / f"{NATIVE_HOST_NAME}.json"


def _owner_id() -> int | None:
    getuid = getattr(os, "getuid", None)
    return getuid() if getuid is not None else None


def _safe_owned_stat(file_stat: os.stat_result, *, directory: bool, require_private: bool) -> bool:
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(file_stat.st_mode):
        return False
    owner = _owner_id()
    if owner is not None and file_stat.st_uid != owner:
        return False
    mask = 0o077 if require_private else 0o022
    return not stat.S_IMODE(file_stat.st_mode) & mask


def _ensure_safe_parent(path: Path) -> None:
    parent = path.parent
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=False)
    except FileExistsError:
        pass
    try:
        parent_stat = parent.lstat()
    except OSError:
        raise NativeHostInstallError("Native host destination directory is unavailable.") from None
    if not _safe_owned_stat(parent_stat, directory=True, require_private=False):
        raise NativeHostInstallError("Native host destination directory has unsafe ownership or permissions.")
    if _owner_id() is not None:
        os.chmod(parent, 0o700)


def _assert_replaceable(path: Path) -> None:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return
    if not _safe_owned_stat(file_stat, directory=False, require_private=False):
        raise NativeHostInstallError("Refusing to replace a native host file with unsafe ownership or permissions.")


def _write_private_json(path: Path, value: Mapping[str, object]) -> None:
    destination = path.expanduser()
    _ensure_safe_parent(destination)
    _assert_replaceable(destination)
    payload = json.dumps(dict(value), indent=2, allow_nan=False, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        if _owner_id() is not None:
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as file:
            fd = -1
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _snapshot_file(path: Path) -> _FileSnapshot:
    if not path.exists():
        return _FileSnapshot(path=path, content=None, mode=None)
    file_stat = path.lstat()
    if not _safe_owned_stat(file_stat, directory=False, require_private=False):
        raise NativeHostInstallError("Native host file has unsafe ownership or permissions.")
    try:
        content = path.read_bytes()
    except OSError:
        raise NativeHostInstallError("Native host file could not be read safely.") from None
    return _FileSnapshot(path=path, content=content, mode=stat.S_IMODE(file_stat.st_mode))


def _restore_snapshot(snapshot: _FileSnapshot) -> None:
    if snapshot.content is None:
        snapshot.path.unlink(missing_ok=True)
        return
    _ensure_safe_parent(snapshot.path)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{snapshot.path.name}-rollback-", dir=snapshot.path.parent)
    temporary = Path(temporary_name)
    try:
        if snapshot.mode is not None:
            os.fchmod(fd, snapshot.mode)
        with os.fdopen(fd, "wb", closefd=True) as file:
            fd = -1
            file.write(snapshot.content)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, snapshot.path)
    finally:
        if fd >= 0:
            os.close(fd)
        temporary.unlink(missing_ok=True)


def _is_trusted_host_executable(path: Path) -> bool:
    try:
        file_stat = path.stat()
    except OSError:
        return False
    owner = _owner_id()
    return (
        stat.S_ISREG(file_stat.st_mode)
        and os.access(path, os.X_OK)
        and not file_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        and (owner is None or file_stat.st_uid == owner)
    )


def resolve_native_host_executable(explicit_path: Path | None = None) -> Path:
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    else:
        candidates.append(Path(sys.executable).parent / "kolega-code-browser-host")
        discovered = shutil.which("kolega-code-browser-host")
        if discovered:
            candidates.append(Path(discovered))
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if _is_trusted_host_executable(resolved):
            return resolved
    raise NativeHostInstallError(
        "Native host executable must be a regular executable owned by the current user and not writable by others."
    )


def build_native_host_manifest(
    host_path: Path,
    *,
    extension_id: str = CHROME_EXTENSION_ID,
) -> dict[str, Any]:
    origin = chrome_extension_origin(extension_id)
    resolved = host_path.expanduser().resolve()
    if not resolved.is_absolute():
        raise NativeHostInstallError("Native host path must be absolute.")
    return {
        "name": NATIVE_HOST_NAME,
        "description": "Kolega Code Chrome browser bridge",
        "path": str(resolved),
        "type": "stdio",
        "allowed_origins": [origin],
    }


def install_native_host(
    *,
    host_path: Path | None = None,
    channel: str = DEFAULT_BROWSER_EXTENSION_CHANNEL,
    extension_id: str | None = None,
    platform: str | None = None,
    home: Path | None = None,
    state_dir: Path | None = None,
    host_config_path: Path | None = None,
) -> NativeHostStatus:
    selected_channel, selected_id = resolve_channel_extension_id(channel=channel, extension_id=extension_id)
    ensure_runtime_transport_supported(platform=platform)
    executable = resolve_native_host_executable(host_path)
    manifest_path = chrome_native_host_manifest_path(platform=platform, home=home)
    origin = chrome_extension_origin(selected_id)
    registry_dir = (
        (state_dir or default_state_dir(platform=platform)).expanduser().resolve() / "browser-extension" / "runtimes"
    )
    config_path = host_config_path or default_host_config_path(platform=platform)
    if manifest_path.exists() or manifest_path.is_symlink():
        existing_manifest = _read_private_mapping(manifest_path)
        if existing_manifest is None or set(existing_manifest) != _MANIFEST_KEYS:
            raise NativeHostInstallError("Refusing to replace an unsafe or foreign native host manifest.")
        if existing_manifest.get("name") != NATIVE_HOST_NAME or existing_manifest.get("allowed_origins") != [origin]:
            raise NativeHostInstallError("Refusing to replace an unsafe or foreign native host manifest.")
    if config_path.exists() or config_path.is_symlink():
        try:
            existing_config = load_host_config(config_path)
        except NativeHostConfigurationError:
            raise NativeHostInstallError("Refusing to replace an unsafe or malformed host configuration.") from None
        if (
            existing_config.extension_origin != origin
            or existing_config.registry_dir.expanduser().resolve() != registry_dir
        ):
            raise NativeHostInstallError(
                "Refusing to replace host configuration owned by another runtime or extension."
            )
    config_snapshot = _snapshot_file(config_path)
    manifest_snapshot = _snapshot_file(manifest_path)
    try:
        _write_private_json(
            config_path,
            {
                "extension_origin": origin,
                "registry_dir": str(registry_dir),
                "max_runtimes": DEFAULT_MAX_RUNTIMES,
                "max_pending_requests": DEFAULT_MAX_RELAY_PENDING,
            },
        )
        _write_private_json(manifest_path, build_native_host_manifest(executable, extension_id=selected_id))
        status = native_host_status(
            channel=selected_channel,
            extension_id=selected_id,
            platform=platform,
            home=home,
            state_dir=state_dir,
            host_config_path=host_config_path,
            expected_host_path=executable,
        )
        if not status.valid:
            raise NativeHostInstallError(status.detail)
        return status
    except BaseException:
        _restore_snapshot(manifest_snapshot)
        _restore_snapshot(config_snapshot)
        raise


def _read_private_mapping(path: Path, *, require_private: bool = True) -> Mapping[str, object] | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError:
        return None
    try:
        file_stat = os.fstat(fd)
        if not _safe_owned_stat(file_stat, directory=False, require_private=require_private):
            return None
        with os.fdopen(fd, "r", encoding="utf-8", closefd=True) as file:
            fd = -1
            raw = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    finally:
        if fd >= 0:
            os.close(fd)
    return raw if isinstance(raw, dict) else None


def _extension_id_from_manifest(manifest: Mapping[str, object]) -> str | None:
    origins = manifest.get("allowed_origins")
    if not isinstance(origins, list) or len(origins) != 1 or not isinstance(origins[0], str):
        return None
    prefix = "chrome-extension://"
    origin = origins[0]
    if not origin.startswith(prefix) or not origin.endswith("/"):
        return None
    extension_id = origin[len(prefix) : -1]
    return extension_id if _EXTENSION_ID_RE.fullmatch(extension_id) else None


def native_host_status(
    *,
    channel: str = DEFAULT_BROWSER_EXTENSION_CHANNEL,
    extension_id: str | None = None,
    platform: str | None = None,
    home: Path | None = None,
    state_dir: Path | None = None,
    host_config_path: Path | None = None,
    expected_host_path: Path | None = None,
) -> NativeHostStatus:
    selected_channel, selected_id = resolve_channel_extension_id(channel=channel, extension_id=extension_id)
    manifest_path = chrome_native_host_manifest_path(platform=platform, home=home)
    if not manifest_path.is_file():
        return NativeHostStatus(
            manifest_path=manifest_path,
            installed=False,
            valid=False,
            host_path=None,
            extension_id=selected_id,
            detail=f"Native host manifest is not installed for the {selected_channel} channel.",
            channel=selected_channel,
        )
    manifest = _read_private_mapping(manifest_path)
    if manifest is None:
        return NativeHostStatus(
            manifest_path=manifest_path,
            installed=True,
            valid=False,
            host_path=None,
            extension_id=selected_id,
            detail="Native host manifest is unsafe or malformed.",
            channel=selected_channel,
        )
    host_value = manifest.get("path")
    host_path = Path(host_value).expanduser() if isinstance(host_value, str) and host_value else None
    origin = chrome_extension_origin(selected_id)
    configured_id = _extension_id_from_manifest(manifest)
    shape_valid = (
        set(manifest) == _MANIFEST_KEYS
        and manifest.get("name") == NATIVE_HOST_NAME
        and manifest.get("type") == "stdio"
        and manifest.get("allowed_origins") == [origin]
        and host_path is not None
        and host_path.is_absolute()
    )
    try:
        executable_valid = bool(
            host_path and host_path == host_path.resolve() and _is_trusted_host_executable(host_path)
        )
    except (OSError, RuntimeError):
        executable_valid = False
    try:
        expected_executable = resolve_native_host_executable(expected_host_path)
        executable_valid = executable_valid and host_path == expected_executable
    except NativeHostInstallError:
        executable_valid = False
    expected_registry = (
        (state_dir or default_state_dir(platform=platform)).expanduser().resolve() / "browser-extension" / "runtimes"
    )
    try:
        config = load_host_config(host_config_path or default_host_config_path(platform=platform))
        config_valid = (
            config.extension_origin == origin and config.registry_dir.expanduser().resolve() == expected_registry
        )
    except NativeHostConfigurationError:
        config_valid = False
    valid = shape_valid and executable_valid and config_valid
    detail = (
        f"Native host manifest is installed and valid for the {selected_channel} channel."
        if valid
        else "Native host files do not match the executable, extension origin, registry, or safety requirements."
    )
    return NativeHostStatus(
        manifest_path=manifest_path,
        installed=True,
        valid=valid,
        host_path=host_path,
        extension_id=selected_id,
        detail=detail,
        channel=selected_channel,
        configured_extension_id=configured_id,
    )


def configured_extension_origin(
    *,
    state_dir: Path | None = None,
    host_config_path: Path | None = None,
) -> str:
    config = load_host_config(host_config_path or default_host_config_path())
    expected_registry = (state_dir or default_state_dir()).expanduser().resolve() / "browser-extension" / "runtimes"
    if config.registry_dir.expanduser().resolve() != expected_registry:
        raise NativeHostConfigurationError("Native-host configuration points at a different runtime registry")
    return config.extension_origin


def uninstall_native_host(
    *,
    channel: str = DEFAULT_BROWSER_EXTENSION_CHANNEL,
    extension_id: str | None = None,
    platform: str | None = None,
    home: Path | None = None,
    host_config_path: Path | None = None,
) -> bool:
    _, selected_id = resolve_channel_extension_id(channel=channel, extension_id=extension_id)
    origin = chrome_extension_origin(selected_id)
    manifest_path = chrome_native_host_manifest_path(platform=platform, home=home)
    config_path = host_config_path or default_host_config_path(platform=platform)
    manifest_present = manifest_path.exists() or manifest_path.is_symlink()
    config_present = config_path.exists() or config_path.is_symlink()
    if manifest_present:
        manifest = _read_private_mapping(manifest_path)
        if (
            manifest is None
            or set(manifest) != _MANIFEST_KEYS
            or manifest.get("name") != NATIVE_HOST_NAME
            or manifest.get("allowed_origins") != [origin]
        ):
            raise NativeHostInstallError("Refusing to remove an unsafe or foreign native host manifest.")
    if config_present:
        try:
            config = load_host_config(config_path)
        except NativeHostConfigurationError:
            raise NativeHostInstallError("Refusing to remove an unsafe or malformed host configuration.") from None
        if config.extension_origin != origin:
            raise NativeHostInstallError("Refusing to remove host configuration for a different extension.")
    if not manifest_present and not config_present:
        return False

    manifest_snapshot = _snapshot_file(manifest_path)
    config_snapshot = _snapshot_file(config_path)
    try:
        if manifest_present:
            manifest_path.unlink()
        if config_present:
            config_path.unlink()
    except BaseException:
        _restore_snapshot(manifest_snapshot)
        _restore_snapshot(config_snapshot)
        raise
    return True
