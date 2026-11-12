"""Run the old writer in its own isolated interpreter.

This deliberately imports no checkout code. The orchestrator executes it with
``python -I`` in a temporary working directory.
"""

from __future__ import annotations

import asyncio
from importlib.metadata import distribution
import json
from pathlib import Path
import sys

import kolega_code
from kolega_code.cli.session_event_store import FileSessionEventStore
from kolega_code.cli.session_store import SessionStore
from kolega_code.cli.settings import SettingsStore
from kolega_code.llm.models import Message


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_origin(origin: Path, installed_module: Path, prefix: Path, checkout: Path) -> None:
    require(origin == installed_module, f"Imported {origin}, not distribution module {installed_module}")
    require(origin.is_relative_to(prefix), f"Package is outside isolated environment: {origin}")
    require(not origin.is_relative_to(checkout / "kolega_code"), f"Package was imported from checkout: {origin}")


def verify_original_history(history: list, expected: list) -> None:
    require(history == expected, "Previous release did not replay the original conversation")


def verify_original_metadata(metadata: dict, expected: dict) -> None:
    require(
        {key: metadata[key] for key in expected if key in metadata} == expected,
        "Previous release did not read the original session metadata",
    )


def verify_task_list_payloads(events: list, expected: list) -> None:
    tasks = [event for event in events if event.event_type == "task_list_updated"]
    require(
        [event.content for event in tasks] == expected, "Previous release did not preserve canonical task-list payloads"
    )
    require(
        all(event.sub_agent_info is None and event.schema_version == 2 for event in tasks),
        "Previous release changed task-list scope or schema version",
    )


def verify_usage_payloads(events: list, expected: list) -> None:
    usage = [event for event in events if event.event_type == "usage_updated"]
    require(
        [event.content for event in usage] == expected, "Previous release did not preserve canonical usage payloads"
    )
    require(
        all(event.sub_agent_info is None and event.schema_version == 2 for event in usage),
        "Previous release changed usage scope or schema version",
    )


def verify_queue_payloads(events: list, expected: list) -> None:
    queue = [event for event in events if event.event_type == "queue_updated"]
    require(
        [event.content for event in queue] == expected, "Previous release did not preserve canonical queue payloads"
    )
    require(
        all(event.sub_agent_info is None and event.schema_version == 2 for event in queue),
        "Previous release changed queue scope or schema version",
    )


def main() -> None:
    root = Path(sys.argv[1])
    manifest = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    expected_version = sys.argv[3]
    checkout = Path(sys.argv[4]).resolve()
    dist = distribution("kolega-code")
    origin = Path(kolega_code.__file__).resolve()
    installed_module = Path(str(dist.locate_file("kolega_code/__init__.py"))).resolve()
    verify_origin(origin, installed_module, Path(sys.prefix).resolve(), checkout)
    require(dist.version == expected_version, f"Installed {dist.version}, expected {expected_version}")
    print(f"Previous release: {dist.version} ({origin})", flush=True)

    settings_store = SettingsStore(root)
    settings = settings_store.load()
    require(settings.active_provider == "openai", "Previous release did not read the current provider")
    require(
        settings.agent_models["building"]["model"] == "compat-current-model",
        "Previous release did not read the current agent model",
    )

    store = SessionStore(root)
    session_id = manifest["session_id"]
    journal = store.journal(session_id)
    before = journal.read_events(repair_tail=False)
    expected_types = manifest["event_types"]
    require([event.event_type for event in before] == expected_types, "Previous release did not read all events")
    loaded = store.load(session_id)
    require(loaded is not None, "Previous release could not load session history")
    verify_original_history(loaded.history, manifest["original_history"])
    verify_original_metadata(loaded.to_metadata_dict(), manifest["original_metadata"])
    presentation = asyncio.run(FileSessionEventStore(journal).read(session_id))
    expected_ui = [name[3:] for name in expected_types if name.startswith("ui.")]
    require([event.event_type for event in presentation] == expected_ui, "Previous release did not read UI events")
    verify_task_list_payloads(presentation, manifest["task_list_payloads"])
    verify_usage_payloads(presentation, manifest["usage_payloads"])
    verify_queue_payloads(presentation, manifest["queue_payloads"])

    # Exercise real mutations, not a load-and-save no-op or a synthetic append.
    settings.active_theme = "compat-previous-theme"
    settings.agent_models["building"]["model"] = "compat-previous-model"
    settings_store.save(settings)
    recorder = store.recorder(session_id, recover=False)
    recorder.start_turn(Message(role="user", content="previous-release user"))
    recorder.record_assistant(Message(role="assistant", content="previous-release assistant"))
    recorder.finish_turn("completed")
    after = journal.read_events(repair_tail=False)
    require(len(after) > len(before), "Previous release did not append to the journal")
    original = (root / manifest["snapshot"]).read_bytes()
    require(
        store.events_path_for(session_id).read_bytes().startswith(original), "Previous release rewrote original records"
    )
    verify_task_list_payloads(
        asyncio.run(FileSessionEventStore(journal).read(session_id)), manifest["task_list_payloads"]
    )
    verify_usage_payloads(asyncio.run(FileSessionEventStore(journal).read(session_id)), manifest["usage_payloads"])
    verify_queue_payloads(asyncio.run(FileSessionEventStore(journal).read(session_id)), manifest["queue_payloads"])
    print(f"Previous release read {len(before)} records and appended {len(after) - len(before)} records", flush=True)


if __name__ == "__main__":
    main()
