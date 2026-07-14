from pathlib import Path
from unittest.mock import AsyncMock, Mock
import uuid

import pytest

from kolega_code.agent.tool_backend.edit_tool import EditTool
from kolega_code.agent.tool_backend.hashline_v2 import (
    HashlineMismatchError,
    apply_hashline_edits,
    compute_line_hash,
    format_hash_lines,
    format_line_tag,
    parse_edits,
    parse_tag,
)
from kolega_code.agent.prompt_provider import AgentMode
from kolega_code.config import AgentConfig, ModelConfig, ModelProvider
from kolega_code.services.snapshots import SnapshotService


@pytest.fixture
def edit_tool(tmp_path: Path) -> EditTool:
    model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="test-model")
    config = AgentConfig(
        anthropic_api_key="test",
        long_context_config=model,
        fast_config=model,
        thinking_config=model,
    )
    caller = Mock()
    caller.agent_name = "hashline-test"
    caller.current_tool_execution_id = "call-1"
    return EditTool(tmp_path, "workspace", str(uuid.uuid4()), AsyncMock(), config, caller)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", "ZR"),
        ("hello", "HK"),
        ("world", "XQ"),
        ("foo", "BK"),
        ("bar", "MJ"),
        ("  return 42;", "KZ"),
        ("αβγ", "ZT"),
        ("a b\t c", "HH"),
    ],
)
def test_hash_vectors_match_bun_xxhash32(content: str, expected: str) -> None:
    assert compute_line_hash(content) == expected


def test_hash_ignores_whitespace_and_line_number() -> None:
    assert compute_line_hash("a b\t c") == compute_line_hash("abc")
    assert format_line_tag(1, "abc").split("#", 1)[1] == format_line_tag(999, "abc").split("#", 1)[1]


def test_format_and_parse_original_v2_tags() -> None:
    assert format_hash_lines("foo\n\nbar", start_line=7) == "7#BK:foo\n8#ZR:\n9#MJ:bar"
    assert format_hash_lines("foo\rbar") == "1#BK:foo\n2#MJ:bar"
    assert parse_tag(">>> 7#BK:foo").line == 7
    with pytest.raises(ValueError, match="LINE#ID"):
        parse_tag("7:foo")


def test_applies_multiple_operations_bottom_up_against_original_snapshot() -> None:
    original = "alpha\nbeta\ngamma"
    edits = parse_edits(
        [
            {"op": "set", "tag": format_line_tag(2, "beta"), "content": ["BETA"]},
            {
                "op": "insert",
                "after": format_line_tag(1, "alpha"),
                "before": format_line_tag(2, "beta"),
                "content": ["between"],
            },
            {"op": "append", "after": format_line_tag(3, "gamma"), "content": "tail"},
        ]
    )

    assert apply_hashline_edits(original, edits) == "alpha\nbetween\nBETA\ngamma\ntail"


def test_set_and_range_null_delete_but_array_empty_string_keeps_blank_line() -> None:
    original = "one\ntwo\nthree\nfour"
    edits = parse_edits(
        [
            {"op": "set", "tag": format_line_tag(2, "two"), "content": [""]},
            {
                "op": "replace",
                "first": format_line_tag(3, "three"),
                "last": format_line_tag(4, "four"),
                "content": None,
            },
        ]
    )

    assert apply_hashline_edits(original, edits) == "one\n"


def test_collects_all_stale_anchors_with_fresh_context() -> None:
    edits = parse_edits(
        [
            {"op": "set", "tag": "2#ZZ", "content": "BETA"},
            {"op": "set", "tag": "4#ZZ", "content": "DELTA"},
        ]
    )

    with pytest.raises(HashlineMismatchError) as raised:
        apply_hashline_edits("alpha\nbeta\ngamma\ndelta\nepsilon", edits)

    assert len(raised.value.mismatches) == 2
    assert ">>> 2#" in str(raised.value)
    assert ">>> 4#" in str(raised.value)


def test_rejects_nonadjacent_dual_anchor_and_noop() -> None:
    original = "one\ntwo\nthree"
    with pytest.raises(ValueError, match="adjacent"):
        apply_hashline_edits(
            original,
            parse_edits(
                [
                    {
                        "op": "insert",
                        "after": format_line_tag(1, "one"),
                        "before": format_line_tag(3, "three"),
                        "content": ["new"],
                    }
                ]
            ),
        )
    with pytest.raises(ValueError, match="No changes"):
        apply_hashline_edits(
            original,
            parse_edits([{"op": "set", "tag": format_line_tag(2, "two"), "content": ["two"]}]),
        )


@pytest.mark.asyncio
async def test_backend_preserves_bom_and_crlf(edit_tool: EditTool, tmp_path: Path) -> None:
    target = tmp_path / "source.py"
    target.write_bytes("\ufeffone\r\ntwo\r\n".encode())

    result = await edit_tool.hashline_edit(
        "source.py",
        [{"op": "set", "tag": format_line_tag(2, "two"), "content": ["TWO"]}],
    )

    assert result == "Updated source.py"
    assert target.read_bytes() == "\ufeffone\r\nTWO\r\n".encode()


@pytest.mark.asyncio
async def test_backend_creates_moves_and_deletes(edit_tool: EditTool, tmp_path: Path) -> None:
    created = await edit_tool.hashline_edit(
        "new/file.txt",
        [{"op": "append", "content": ["one", "two"]}],
    )
    assert created == "Created new/file.txt"
    assert (tmp_path / "new/file.txt").read_text() == "one\ntwo"

    moved = await edit_tool.hashline_edit(
        "new/file.txt",
        [{"op": "set", "tag": format_line_tag(2, "two"), "content": ["TWO"]}],
        rename="moved/file.txt",
    )
    assert moved == "Updated and moved new/file.txt to moved/file.txt"
    assert not (tmp_path / "new/file.txt").exists()
    assert (tmp_path / "moved/file.txt").read_text() == "one\nTWO"

    deleted = await edit_tool.hashline_edit("moved/file.txt", [], delete=True)
    assert deleted == "Deleted moved/file.txt"
    assert not (tmp_path / "moved/file.txt").exists()


@pytest.mark.asyncio
async def test_backend_stale_failure_and_bad_path_write_nothing(edit_tool: EditTool, tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("one\ntwo\n")

    with pytest.raises(HashlineMismatchError):
        await edit_tool.hashline_edit("file.txt", [{"op": "set", "tag": "2#ZZ", "content": ["TWO"]}])
    assert target.read_text() == "one\ntwo\n"

    with pytest.raises(ValueError, match="outside the project"):
        await edit_tool.hashline_edit("../escape.txt", [{"op": "append", "content": ["bad"]}])
    assert not (tmp_path.parent / "escape.txt").exists()

    with pytest.raises(ValueError, match="outside the project"):
        await edit_tool.hashline_write("../escape.txt", "bad")


@pytest.mark.asyncio
async def test_backend_checks_every_vibe_protected_path(edit_tool: EditTool, tmp_path: Path) -> None:
    source = tmp_path / "safe.txt"
    source.write_text("safe\n")
    edit_tool.caller.agent_mode = AgentMode.VIBE.value
    edit_tool.caller.protected_files = {"package.json"}

    result = await edit_tool.hashline_edit(
        "safe.txt",
        [{"op": "set", "tag": format_line_tag(1, "safe"), "content": ["changed"]}],
        rename="package.json",
    )

    assert "not allowed" in result
    assert source.read_text() == "safe\n"
    assert not (tmp_path / "package.json").exists()


@pytest.mark.asyncio
async def test_backend_snapshot_restores_rename_source_and_destination(edit_tool: EditTool, tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    destination = tmp_path / "destination.txt"
    source.write_text("before\n")
    destination.write_text("occupied\n")
    snapshots = SnapshotService(
        tmp_path,
        "workspace",
        "thread",
        "session",
        edit_tool.filesystem,
        root=tmp_path / "state",
    )
    edit_tool._snapshot_service = snapshots

    await edit_tool.hashline_edit(
        "source.txt",
        [{"op": "set", "tag": format_line_tag(1, "before"), "content": ["after"]}],
        rename="destination.txt",
    )
    record = snapshots.latest_snapshot()
    assert record is not None

    snapshots.restore_snapshot(record.snapshot_id)

    assert source.read_text() == "before\n"
    assert destination.read_text() == "occupied\n"
