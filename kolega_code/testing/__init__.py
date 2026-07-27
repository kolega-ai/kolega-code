"""Reusable test support for code that implements this package's contracts.

Shipped rather than kept in the test tree so that host applications validating
their own backend implementations run exactly the checks the first-party
implementations are held to.
"""

from .store_conformance import CONFORMANCE_CHECKS, StoreFactory, make_event, run_conformance

__all__ = [
    "CONFORMANCE_CHECKS",
    "StoreFactory",
    "make_event",
    "run_conformance",
]
