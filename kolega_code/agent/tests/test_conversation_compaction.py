"""Unit tests for the recency-preserving compaction model on Conversation."""

import pytest

from kolega_code.agent.conversation import Conversation
from kolega_code.llm.models import Message, MessageHistory, TextBlock

from .compaction_helpers import long_history, skill_msg, text_msg, tool_pair


def _conv(messages):
    return Conversation(list(messages))


def test_effective_history_no_compaction_returns_full():
    conv = _conv(long_history(3))  # 6 messages, uncompacted
    effective = conv.effective_history()
    assert list(effective) == list(conv.history)
    assert conv.last_compression_index is None


def test_effective_history_after_compaction_order():
    skill = skill_msg()
    u0, a0 = text_msg("user", "u0"), text_msg("assistant", "a0")
    u1, a1 = text_msg("user", "u1"), text_msg("assistant", "a1")
    u2, a2 = text_msg("user", "u2"), text_msg("assistant", "a2")
    conv = _conv([skill, u0, a0, u1, a1, u2, a2])  # 7 messages

    conv.apply_compaction("SUMMARY TEXT", split_point=5)  # fold [skill,u0,a0,u1,a1]; keep [u2,a2]
    effective = list(conv.effective_history())

    assert effective[0] is skill  # protected skill content survives from the folded prefix
    assert effective[1].get_text_content() == "SUMMARY TEXT"  # the summary
    assert effective[2:] == [u2, a2]  # recent turns kept verbatim, in order


def test_effective_history_preserves_only_protected_prefix():
    skill = skill_msg()
    u0, a0, u1, a1 = (
        text_msg("user", "u0"),
        text_msg("assistant", "a0"),
        text_msg("user", "u1"),
        text_msg("assistant", "a1"),
    )
    conv = _conv([u0, skill, a0, u1, a1])
    conv.apply_compaction("S", split_point=4)  # fold first 4; keep [a1]
    effective = list(conv.effective_history())
    # Only the protected skill message survives verbatim from the folded prefix.
    assert effective[0] is skill
    assert effective[1].get_text_content() == "S"
    assert effective[2:] == [a1]
    assert u0 not in effective and a0 not in effective and u1 not in effective


def test_recent_tail_is_verbatim_identity():
    history = long_history(5)  # 10 messages
    conv = _conv(history)
    conv.apply_compaction("S", split_point=4)
    effective = list(conv.effective_history())
    assert effective[0].get_text_content() == "S"  # no protected prefix here -> summary first
    tail = effective[1:]
    # Every tail message is the *same object* as in history (not re-summarized).
    for original, shown in zip(list(history)[4:], tail):
        assert original is shown


@pytest.mark.parametrize("keep_recent", [1, 2, 3, 4, 5])
def test_compaction_never_splits_tool_pair(keep_recent):
    a_tc, u_tr = tool_pair("tool_x")
    history = MessageHistory(
        [
            text_msg("user", "u0"),
            text_msg("assistant", "a0"),
            text_msg("user", "u1"),
            text_msg("assistant", "a1"),
            a_tc,
            u_tr,
            text_msg("user", "u2"),
            text_msg("assistant", "a2"),
        ]
    )
    conv = _conv(history)
    split = conv.compaction_split_point(keep_recent=keep_recent, min_prefix=1)
    if split is None:
        pytest.skip("prefix too small for this keep_recent")
    conv.apply_compaction("S", split)
    # No tool_use is orphaned across the boundary.
    assert conv.is_valid_for_anthropic(list(conv.effective_history())) is True


def test_stale_compacted_through_is_clamped():
    conv = _conv(long_history(3))  # 6 messages
    conv.summary = Message(role="user", content=[TextBlock(text="S")])
    conv.compacted_through = 999  # beyond len(history)
    # min(compacted_through, len) keeps it safe; no IndexError.
    effective = list(conv.effective_history())
    assert effective[-1].get_text_content() == "S"  # everything folded, tail empty


def test_clear_resets_compaction_state():
    conv = _conv(long_history(4))
    conv.apply_compaction("S", 3)
    conv.clear()
    assert list(conv.history) == []
    assert conv.summary is None
    assert conv.compacted_through == 0
    assert conv.last_compression_index is None
    assert list(conv.effective_history()) == []


def test_history_setter_resets_compaction_state():
    conv = _conv(long_history(4))
    conv.apply_compaction("S", 3)
    conv.history = MessageHistory([text_msg("user", "fresh")])
    assert conv.summary is None
    assert conv.compacted_through == 0
    assert conv.last_compression_index is None


def test_compaction_survives_extend():
    conv = _conv(long_history(4))  # 8 messages
    conv.apply_compaction("S", 4)
    conv.extend([text_msg("user", "a live follow-up turn")])
    # Appending live turns must NOT wipe the active summary.
    assert conv.summary is not None
    assert conv.compacted_through == 4
    effective = list(conv.effective_history())
    assert effective[0].get_text_content() == "S"
    assert effective[-1].get_text_content() == "a live follow-up turn"


def test_record_compression_shim_folds_everything():
    history = long_history(3)  # 6 messages
    conv = _conv(history)
    conv.record_compression(Message(role="user", content=[TextBlock(text="WHOLE")]))
    assert conv.compacted_through == len(history)
    effective = list(conv.effective_history())
    assert len(effective) == 1  # only the summary (no protected prefix, empty tail)
    assert effective[0].get_text_content() == "WHOLE"


def test_last_compression_index_back_compat():
    conv = _conv(long_history(3))
    conv.apply_compaction("S", split_point=3)
    assert conv.last_compression_index == 2  # compacted_through - 1
    conv.last_compression_index = None  # legacy clear
    assert conv.summary is None
    assert conv.compacted_through == 0


def test_dump_restore_resets_compaction():
    conv = _conv(long_history(4))
    conv.apply_compaction("S", 4)
    dumped = conv.dump()
    restored = Conversation()
    restored.restore(dumped)
    # restore() treats the dumped log as authentic + uncompacted; it re-compacts
    # on the next over-budget turn.
    assert restored.summary is None
    assert restored.compacted_through == 0
    assert len(restored.history) == 8


def test_dump_restore_compaction_round_trip():
    conv = _conv(long_history(6))  # 12 messages
    conv.apply_compaction("THE SUMMARY", 6)
    data = conv.dump_compaction()
    assert data == {"summary": "THE SUMMARY", "compacted_through": 6}

    # A fresh conversation gets the full history back, then the boundary on top.
    restored = Conversation(list(conv.history))
    restored.restore_compaction(data)
    assert restored.summary is not None
    assert restored.compacted_through == 6
    effective = list(restored.effective_history())
    assert effective[0].get_text_content() == "THE SUMMARY"
    assert effective[1:] == list(conv.history)[6:]  # verbatim tail preserved


def test_restore_compaction_empty_is_noop():
    conv = _conv(long_history(3))
    for data in ({}, None, {"summary": "", "compacted_through": 5}, {"summary": "S", "compacted_through": 0}):
        conv.restore_compaction(data)
        assert conv.summary is None
        assert conv.compacted_through == 0


def test_restore_compaction_clamps_boundary():
    conv = _conv(long_history(2))  # 4 messages
    conv.restore_compaction({"summary": "S", "compacted_through": 999})
    assert conv.compacted_through == 4  # clamped to len(history)
    assert conv.summary is not None
