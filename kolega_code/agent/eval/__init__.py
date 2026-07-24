"""Persistent eval kernels (Python/JavaScript) with a loopback tool bridge.

Internal package — not part of the supported ``kolega_code`` public API.
"""

from .bridge import BridgeRegistration, ToolBridge
from .env import EvalEnvironmentManager, KernelEnvInfo
from .kernel import (
    BaseKernel,
    EvalCellResult,
    EvalKernelManager,
    KernelErrorInfo,
    KernelUnavailableError,
)

__all__ = [
    "BaseKernel",
    "BridgeRegistration",
    "EvalCellResult",
    "EvalEnvironmentManager",
    "EvalKernelManager",
    "KernelEnvInfo",
    "KernelErrorInfo",
    "KernelUnavailableError",
    "ToolBridge",
]
