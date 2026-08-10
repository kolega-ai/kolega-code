"""Guard the package's import-time footprint.

``import kolega_code`` must never load the optional native Tinker stack:
``providers/tinker.py`` imports tinker-cookbook (and therefore torch) at module
load whenever the ``[tinker]`` extra is installed, torch takes ~0.6s to import,
and torch.package monkey-patches ``inspect.getfile`` process-wide — which taxes
every Textual widget mount in the TUI. Providers are loaded lazily per session;
only the dependency-free ``tinker_trace`` leaf module is exported eagerly.

Run in a subprocess so the check is independent of what this test process has
already imported. In environments without the ``[tinker]`` extra the assertion
is trivially true; with it installed (the dev venv), this catches any eager
import creeping back into the package root.
"""

import subprocess
import sys

_CHECK = """
import sys
import kolega_code

heavy = [name for name in ("torch", "tinker", "tinker_cookbook") if name in sys.modules]
assert not heavy, f"import kolega_code eagerly loaded: {heavy}"
assert kolega_code.TinkerTraceRecord is not None
"""


def test_importing_the_package_does_not_load_the_native_tinker_stack() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _CHECK],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
