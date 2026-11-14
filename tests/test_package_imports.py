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

The same guard covers the web-fetch HTML extraction stack and the Firecrawl SDK,
which only a fetch or a keyed search needs but which used to load on every CLI
start.
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

#: The HTML extraction stack (trafilatura plus its htmldate/dateparser/courlan
#: tree, readability, markdownify) costs ~215 ms of import work. It is only
#: needed once a fetch extracts HTML, so it must not load on a CLI start.
_WEB_FETCH_CHECK = """
import sys
import kolega_code.cli.main

heavy = [name for name in ("trafilatura", "readability", "markdownify") if name in sys.modules]
assert not heavy, f"import kolega_code.cli.main eagerly loaded: {heavy}"
"""

#: bs4 pulls in soupsieve, chardet and lxml's builder (~55 ms). Only decoding a
#: fetched body or extracting HTML needs it, so it may not load on a CLI start.
_BS4_CHECK = """
import sys
import kolega_code.cli.main
from kolega_code.agent.tool_backend.web_fetch import extractors, pipeline

heavy = [name for name in ("bs4", "lxml") if name in sys.modules]
assert not heavy, f"import kolega_code.cli.main eagerly loaded: {heavy}"
assert extractors.DEFAULT_EXTRACTOR and pipeline.WebContent
"""

#: aiohttp backs the messaging gateway's Telegram adapter (~110 ms of import
#: work). The LLM error mapper only needs it to classify an exception that must
#: already have imported it, so it may not load on a CLI start.
_AIOHTTP_CHECK = """
import sys
import kolega_code.cli.main
import kolega_code.llm.exceptions

assert "aiohttp" not in sys.modules, "import kolega_code.cli.main eagerly loaded: aiohttp"
"""

#: httpx costs ~50 ms of client/SSL/h11 setup. The LLM error mapper classifies an
#: httpx exception from ``sys.modules`` (it can only exist once a client imported
#: httpx), and the OAuth refresh, catalog refresh and MCP transport paths import it
#: when they make a request. None of them needs it on a CLI start.
_HTTPX_CHECK = """
import sys
import kolega_code.cli.main

assert "httpx" not in sys.modules, "import kolega_code.cli.main eagerly loaded: httpx"
"""

#: firecrawl-py builds its entire pydantic type tree at import time (~205 ms).
#: Only a keyed Firecrawl search uses the SDK, so it may not load on a CLI start.
_FIRECRAWL_CHECK = """
import sys
import kolega_code.cli.main
import kolega_code.agent.tool_backend.search_backends.firecrawl

assert "firecrawl" not in sys.modules, "import kolega_code.cli.main eagerly loaded: firecrawl"
"""


def test_importing_the_package_does_not_load_the_native_tinker_stack() -> None:
    result = subprocess.run(
        [sys.executable, "-c", _CHECK],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_cli_start_does_not_load_the_html_extraction_stack() -> None:
    """trafilatura/readability/markdownify load on first extraction, not on start.

    They cost ~215 ms of import work; ``web_fetch`` is the only caller and most
    sessions never fetch a page.
    """
    result = subprocess.run(
        [sys.executable, "-c", _WEB_FETCH_CHECK],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_cli_start_does_not_load_bs4() -> None:
    """bs4/lxml load on the first fetched body or extracted page, not on start.

    Importing them costs ~55 ms of parser and encoding-detector setup; every
    CLI invocation paid it even when the session never fetched a page.
    """
    result = subprocess.run(
        [sys.executable, "-c", _BS4_CHECK],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_cli_start_does_not_load_aiohttp() -> None:
    """aiohttp loads on the gateway's first Telegram call, not on start.

    Importing it costs ~110 ms of aiohttp client/connector construction; the LLM
    error mapper is the only startup-path importer and classifies aiohttp errors
    from ``sys.modules`` instead.
    """
    result = subprocess.run(
        [sys.executable, "-c", _AIOHTTP_CHECK],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_cli_start_does_not_load_httpx() -> None:
    """httpx loads on the first request that needs a client, not on start.

    Importing it costs ~50 ms of client/SSL/h11 setup, and it used to arrive
    through ``llm.exceptions``, ``auth.tokens``, ``llm.timeouts``, the model
    catalogs or the MCP transport. The error mapper resolves httpx exception
    classes from ``sys.modules``; the rest import it inside the call that makes
    a request. Deferring them cut CLI-start import time by ~37 ms.
    """
    result = subprocess.run(
        [sys.executable, "-c", _HTTPX_CHECK],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


def test_cli_start_does_not_load_the_firecrawl_sdk() -> None:
    """The Firecrawl SDK loads on the first keyed search, not on start.

    Importing it costs ~205 ms of pydantic type construction; the keyless path
    talks to the REST endpoint over httpx and most sessions never select the
    backend at all.
    """
    result = subprocess.run(
        [sys.executable, "-c", _FIRECRAWL_CHECK],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
