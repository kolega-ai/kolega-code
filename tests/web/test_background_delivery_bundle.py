from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import pytest

from kolega_code.events import AgentEvent, ArtifactPurpose
from kolega_code.session.inmemory import InMemoryArtifactStore
from kolega_code.session.recording import _preview
from kolega_code.web.bundle import export_bundle
from tests.web.test_background_delivery_fold import DELIVERY, DELIVERY_EVENT


@pytest.mark.asyncio
async def test_inline_delivery_report_embeds_only_selected_redacted_shareable_text(tmp_path: Path) -> None:
    store = InMemoryArtifactStore()
    secret = "report-specific-credential"
    full_text = f"Complete report 測試\n{secret}\n" + "long report line\n" * 2000 + "REPORT END"
    report = await store.put(
        full_text.encode(),
        media_type="text/plain; charset=utf-8",
        purpose=ArtifactPurpose.TOOL_RESULT,
        encoding="utf-8",
        chars=len(full_text),
    )
    unrelated = await store.put(
        b"unrelated tool text",
        media_type="text/plain",
        purpose=ArtifactPurpose.TOOL_RESULT,
        encoding="utf-8",
    )
    opaque = await store.put(
        b"private provider state",
        media_type="text/plain",
        purpose=ArtifactPurpose.PROVIDER_SIGNATURE,
        encoding="utf-8",
    )
    event = AgentEvent(
        sender="root",
        event_type=DELIVERY_EVENT,
        content={**DELIVERY, "text": _preview(full_text, 256, report)},
        artifacts=[report, unrelated, opaque],
    )
    original = event.model_dump()
    result = await export_bundle(
        [event],
        tmp_path / "report.html",
        session_id="session",
        artifact_store=store,
        extra_secrets=[secret],
        single_file=True,
    )
    document = result.path.read_text()
    match = re.search(r"globalThis\.__KC_REPLAY__ = (.*?);</script>", document)
    assert match is not None
    inline = json.loads(match.group(1))
    assert set(inline["artifacts"]) == {report.sha256}
    source = inline["artifacts"][report.sha256]
    assert source.startswith("data:text/plain; charset=utf-8;base64,")
    decoded = base64.b64decode(source.split(",", 1)[1]).decode()
    assert secret not in decoded
    assert decoded.startswith("Complete report 測試\n")
    assert decoded.endswith("REPORT END")
    assert decoded.count("long report line\n") == 2000
    assert result.artifact_count == 1
    assert event.model_dump() == original
