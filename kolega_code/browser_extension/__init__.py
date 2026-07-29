"""Chrome Native Messaging browser backend."""

from .framing import (
    MAX_NATIVE_MESSAGE_BYTES,
    FramingError,
    MessageTooLargeError,
    TruncatedMessageError,
    read_message,
    read_message_async,
    write_message,
    write_message_async,
)
from .manager import (
    CHROME_EXTENSION_CAPABILITIES,
    CHROME_EXTENSION_SUPPORTED_TOOLS,
    ChromeExtensionBrowserManager,
    ChromeExtensionProtocolError,
    ChromeExtensionUnavailableError,
)
from .multiplex import (
    ConnectionClosedError,
    MultiplexedPeer,
    PendingRequestLimitError,
    RemoteRequestError,
    RequestTimeoutError,
)
from .protocol import (
    ALLOWED_OPERATIONS,
    PROTOCOL_VERSION,
    Envelope,
    MessageDirection,
    MessageType,
    ProtocolValidationError,
)
from .registry import RuntimeDescriptor, RuntimeDescriptorRegistry
from .runtime import (
    RuntimeAuthenticationError,
    RuntimeChannel,
    RuntimeServer,
    UnsupportedRuntimeTransportError,
    connect_runtime_channel,
    selected_runtime_transport,
)

__all__ = [
    "ALLOWED_OPERATIONS",
    "CHROME_EXTENSION_CAPABILITIES",
    "CHROME_EXTENSION_SUPPORTED_TOOLS",
    "MAX_NATIVE_MESSAGE_BYTES",
    "PROTOCOL_VERSION",
    "ChromeExtensionBrowserManager",
    "ChromeExtensionProtocolError",
    "ChromeExtensionUnavailableError",
    "ConnectionClosedError",
    "Envelope",
    "FramingError",
    "MessageDirection",
    "MessageTooLargeError",
    "MessageType",
    "MultiplexedPeer",
    "PendingRequestLimitError",
    "ProtocolValidationError",
    "RemoteRequestError",
    "RequestTimeoutError",
    "RuntimeAuthenticationError",
    "RuntimeChannel",
    "RuntimeDescriptor",
    "RuntimeDescriptorRegistry",
    "RuntimeServer",
    "TruncatedMessageError",
    "UnsupportedRuntimeTransportError",
    "connect_runtime_channel",
    "read_message",
    "read_message_async",
    "selected_runtime_transport",
    "write_message",
    "write_message_async",
]
