from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from PIL import Image

from kolega_code.agent.context_breakdown import (
    CATEGORY_AGENT_RESPONSES,
    CATEGORY_AUTOCOMPACT_BUFFER,
    CATEGORY_BUILTIN_TOOLS,
    CATEGORY_COMPACTION_SUMMARY,
    CATEGORY_FREE,
    CATEGORY_GUIDANCE_MEMORY,
    CATEGORY_IMAGES,
    CATEGORY_MCP_TOOLS,
    CATEGORY_REASONING,
    CATEGORY_SERVER_OVERHEAD,
    CATEGORY_SKILLS,
    CATEGORY_SYSTEM_PROMPT,
    CATEGORY_SYSTEM_REMINDERS,
    CATEGORY_TOOL_USAGE,
    CATEGORY_USER_MESSAGES,
    ContextBreakdown,
    ContextCategory,
    allocate_grid_cells,
    analyze_agent_context,
    compute_context_breakdown,
    format_context_breakdown_text,
    format_token_count,
    snapshot_agent_context,
)
from kolega_code.agent.conversation import Conversation
from kolega_code.agent.prompt_provider import PromptExtension
from kolega_code.llm.models import (
    ContentBlock,
    ImageBlock,
    Message,
    RedactedThinkingBlock,
    ResponsesReasoningBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    ToolResult,
)


@dataclass
class _FakeToolCollection:
    tools: list[ToolDefinition] = field(default_factory=list)
    mcp_tools: list[str] = field(default_factory=list)
    extension_tool_sources: dict[str, str] = field(default_factory=dict)
    extension_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_tool_list(self) -> list[ToolDefinition]:
        return list(self.tools)


@dataclass
class _FakeCustomAgents:
    agent_names: tuple[str, ...]
    catalog_text: str

    def has_agents(self) -> bool:
        return bool(self.agent_names)

    def names(self) -> tuple[str, ...]:
        return self.agent_names

    def model_catalog(self) -> str:
        return self.catalog_text


def _make_agent(
    *,
    provider: str = "anthropic",
    model: str = "claude-sonnet",
    effort: str | None = "medium",
    agent_name: str = "coder_agent",
    context_length: int = 200_000,
    completion_tokens: int = 64_000,
    max_input_tokens: int = 136_000,
    strict_context_budget: bool = False,
    compression_threshold: float = 0.95,
    system_prompt_text: str = "You are Kolega Code.",
    prompt_extensions: list[PromptExtension] | None = None,
    tools: list[ToolDefinition] | None = None,
    mcp_tool_names: list[str] | None = None,
    mcp_tool_metadata: dict[str, dict[str, Any]] | None = None,
    custom_agents: _FakeCustomAgents | None = None,
    conversation: Conversation | None = None,
    supports_vision: bool = True,
    residual_tokens: int = 0,
    last_raw_count: int | None = None,
) -> Any:
    ext_sources = {name: "mcp" for name in (mcp_tool_names or [])}
    tool_collection = _FakeToolCollection(
        tools=tools or [],
        mcp_tools=list(mcp_tool_names or []),
        extension_tool_sources=ext_sources,
        extension_metadata=mcp_tool_metadata or {},
    )
    conv = conversation if conversation is not None else Conversation([])
    agent = SimpleNamespace(
        primary_model_config=SimpleNamespace(provider=provider, model=model, thinking_effort=effort),
        agent_name=agent_name,
        model_context_length=context_length,
        model_completion_tokens=completion_tokens,
        model_max_input_tokens=max_input_tokens,
        strict_context_budget=strict_context_budget,
        history_compression_threshold=compression_threshold,
        system_prompt=Message(role="system", content=[TextBlock(text=system_prompt_text)]),
        prompt_extensions=prompt_extensions or [],
        tool_collection=tool_collection,
        custom_agent_catalog=custom_agents,
        conversation=conv,
        history=conv.history,
        supports_vision=supports_vision,
        edit_protocol=None,
        _context_residual_tokens=residual_tokens,
        _last_raw_context_count=last_raw_count,
        count_current_context=AsyncMock(),
    )
    return agent


def test_format_token_count_boundaries():
    assert format_token_count(0) == "0"
    assert format_token_count(704) == "704"
    assert format_token_count(1_000) == "1k"
    assert format_token_count(4_620) == "4.6k"
    assert format_token_count(999_949) == "999.9k"
    assert format_token_count(999_950) == "1M"
    assert format_token_count(1_048_576) == "1M"
    assert format_token_count(1_250_000) == "1.2M"


@pytest.mark.asyncio
async def test_fresh_session_baseline_counts_static_prefix_without_provider_call():
    skills_md = "- `deep-research`: Deep multi-source reports.\n- `pdf`: Work with PDFs."
    system_text = f"Core system instructions.\n\n### Agent Skills\n\n{skills_md}"
    agent = _make_agent(
        system_prompt_text=system_text,
        prompt_extensions=[PromptExtension(id="cli-agent-skills", title="Agent Skills", markdown=skills_md)],
        tools=[
            ToolDefinition(
                name="read", description="Read files from disk.", parameters=[], input_schema={"type": "object"}
            ),
            ToolDefinition(
                name="exec_command", description="Run shell commands.", parameters=[], input_schema={"type": "object"}
            ),
        ],
        conversation=Conversation(
            [
                Message(
                    role="user",
                    content=[
                        TextBlock(
                            text='<system-reminder source="session">\nWorking directory: /repo\n</system-reminder>'
                        )
                    ],
                )
            ]
        ),
    )

    breakdown = await analyze_agent_context(agent, refresh_provider_count=True)

    agent.count_current_context.assert_not_awaited()
    assert breakdown.total_source == "estimated"
    assert breakdown.total_tokens == breakdown.local_total_tokens > 0
    by_key = {cat.key: cat for cat in breakdown.categories}
    assert CATEGORY_SYSTEM_PROMPT in by_key
    assert CATEGORY_BUILTIN_TOOLS in by_key
    assert CATEGORY_SKILLS in by_key
    assert CATEGORY_SYSTEM_REMINDERS in by_key
    assert by_key[CATEGORY_BUILTIN_TOOLS].detail == "2 tools"
    assert by_key[CATEGORY_SKILLS].detail == "2 catalog"


def _make_png_base64(width: int = 128, height: int = 128) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=(64, 128, 192)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_full_multicategory_breakdown_and_mcp_server_aliasing():
    skills_md = "- `deep-research`: Multi-source research.\n- `docx`: Word files."
    scratch_md = "Scratchpad lives at /tmp/scratch."
    system_text = (
        f"Base prompt instructions.\n\n### Agent Skills\n\n{skills_md}\n\n### Session Scratchpad\n\n{scratch_md}"
    )

    # Include a hashed/aliased MCP wire name whose true server_id carries "__".
    aliased_mcp = "mcp__sabcdef123456__list_issues"
    tools = [
        ToolDefinition(
            name="dispatch_agent",
            description="Dispatch autonomous sub-agents.\n- `reviewer`: Review PRs.",
            parameters=[],
            input_schema={"type": "object"},
        ),
        ToolDefinition(name="read", description="Read a file.", parameters=[], input_schema={"type": "object"}),
        ToolDefinition(
            name="mcp__github__search_code",
            description="Search GitHub code.",
            parameters=[],
            input_schema={"type": "object"},
        ),
        ToolDefinition(
            name=aliased_mcp,
            description="List issues from aliased server.",
            parameters=[],
            input_schema={"type": "object"},
        ),
    ]

    volatile_combined = "\n".join(
        [
            '<system-reminder source="guidance" path="AGENTS.md">\nFollow repo conventions.\n</system-reminder>',
            '<system-reminder source="memory">\n# Memory index\n- build quirk\n</system-reminder>',
            '<system-reminder source="date">\nToday is 2026-11-26.\n</system-reminder>',
        ]
    )
    png_b64_large = _make_png_base64(256, 256)
    png_b64_small = _make_png_base64(64, 64)
    conv = Conversation(
        [
            Message(role="user", content=[TextBlock(text=volatile_combined)]),
            Message(
                role="user",
                content=[
                    TextBlock(text="Please inspect this screenshot and file."),
                    TextBlock(text='<attached-file path="src/app.py">\nprint("hello")\n</attached-file>'),
                    ImageBlock(image_type="base64", media_type="image/png", data=png_b64_large),
                ],
            ),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="Let me read the file and activate the skill.", signature="sig123"),
                    TextBlock(text="Reading the file now."),
                    ToolCall(id="call_1", name="read", input={"file_path": "src/app.py"}),
                    ToolCall(id="call_2", name="skill", input={"name": "deep-research"}),
                    ToolCall(id="call_3", name="read_image", input={"path": "diagram.png"}),
                ],
                usage_metadata={"provider": "anthropic"},
            ),
            Message(
                role="user",
                content=list[ContentBlock](
                    [
                        ToolResult(
                            tool_use_id="call_1",
                            name="read",
                            content="def main():\n    return 42\n" * 40,
                            is_error=False,
                        ),
                        ToolResult(
                            tool_use_id="call_2",
                            name="skill",
                            content='<skill_content name="deep-research">\nConduct rigorous research.\n</skill_content>',
                            is_error=False,
                        ),
                        ToolResult(
                            tool_use_id="call_3",
                            name="read_image",
                            content=[
                                TextBlock(text="Loaded diagram.png"),
                                ImageBlock(image_type="base64", media_type="image/png", data=png_b64_small),
                            ],
                            is_error=False,
                        ),
                    ]
                ),
            ),
            Message(
                role="user",
                content=[
                    TextBlock(text='<skill_content name="docx">\nAuthor DOCX safely.\n</skill_content>\nCreate doc.')
                ],
            ),
            Message(
                role="assistant",
                content=[
                    RedactedThinkingBlock(data="redacted_blob"),
                    ResponsesReasoningBlock(encrypted_content="enc", summary=["short summary"], content=[]),
                    TextBlock(text="All done with analysis."),
                ],
                usage_metadata={"provider": "anthropic"},
            ),
        ]
    )

    agent = _make_agent(
        system_prompt_text=system_text,
        prompt_extensions=[
            PromptExtension(id="cli-agent-skills", title="Agent Skills", markdown=skills_md),
            PromptExtension(id="session-scratchpad", title="Session Scratchpad", markdown=scratch_md),
        ],
        tools=tools,
        mcp_tool_names=["mcp__github__search_code", aliased_mcp],
        mcp_tool_metadata={
            "mcp__github__search_code": {"mcp_server_id": "github", "mcp_tool_id": "search_code"},
            aliased_mcp: {"mcp_server_id": "corp__tracker", "mcp_tool_id": "list_issues"},
        },
        custom_agents=_FakeCustomAgents(("reviewer",), "- `reviewer`: Review PRs."),
        conversation=conv,
        supports_vision=True,
        residual_tokens=450,
    )

    breakdown = compute_context_breakdown(snapshot_agent_context(agent))
    by_key = {cat.key: cat for cat in breakdown.categories}

    for expected in (
        CATEGORY_SYSTEM_PROMPT,
        CATEGORY_BUILTIN_TOOLS,
        CATEGORY_MCP_TOOLS,
        CATEGORY_SKILLS,
        CATEGORY_GUIDANCE_MEMORY,
        CATEGORY_SYSTEM_REMINDERS,
        CATEGORY_USER_MESSAGES,
        CATEGORY_AGENT_RESPONSES,
        CATEGORY_REASONING,
        CATEGORY_TOOL_USAGE,
        CATEGORY_IMAGES,
        CATEGORY_SERVER_OVERHEAD,
    ):
        assert expected in by_key and by_key[expected].tokens > 0, f"Missing {expected}"

    # Image heuristic (~794 + ~509), never raw base64 BPE (~35k+).
    assert breakdown.image_count == 2
    assert by_key[CATEGORY_IMAGES].tokens < 3_000
    assert by_key[CATEGORY_SERVER_OVERHEAD].tokens == 450
    assert breakdown.mcp_server_count == 2
    assert breakdown.active_skill_count == 2
    assert breakdown.custom_agent_count == 1

    groups = {group.key: group for group in breakdown.detail_groups}
    assert {"AGENTS.md", "MEMORY.md (startup index)"} == {item.label for item in groups["guidance_memory"].items}
    assert {"Skills catalog", "/deep-research", "/docx"} == {item.label for item in groups["skills"].items}
    # Aliased server resolved back to real human server_id.
    assert {"github", "corp__tracker"} == {item.label for item in groups["mcp_servers"].items}
    assert groups["custom_agents"].items[0].label == "reviewer"
    assert groups["heaviest_tools"].items[0].label == "read"
    assert groups["attachments"].items[0].label == "src/app.py"
    assert groups["prompt_extensions"].items[0].label == "Session Scratchpad"


def test_compacted_history_uses_effective_history_and_keeps_protected_skills():
    conv = Conversation(
        [
            Message(role="user", content=[TextBlock(text="old huge message " * 300)]),
            Message(
                role="user",
                content=[TextBlock(text='<skill_content name="pdf">\nPDF guidance\n</skill_content>')],
            ),
            Message(role="assistant", content=[TextBlock(text="old huge reply " * 300)]),
            Message(role="user", content=[TextBlock(text="recent question")]),
            Message(role="assistant", content=[TextBlock(text="recent answer")]),
        ]
    )
    conv.apply_compaction("Compact summary of earlier work.", split_point=3)
    agent = _make_agent(conversation=conv)

    breakdown = compute_context_breakdown(snapshot_agent_context(agent))
    by_key = {cat.key: cat for cat in breakdown.categories}

    assert breakdown.compacted_messages == 3
    assert CATEGORY_COMPACTION_SUMMARY in by_key
    assert breakdown.compaction_summary_tokens == by_key[CATEGORY_COMPACTION_SUMMARY].tokens > 0
    # Old huge messages were folded, so user/assistant tokens stay small.
    assert by_key[CATEGORY_USER_MESSAGES].tokens < 50
    assert by_key[CATEGORY_AGENT_RESPONSES].tokens < 50
    # Protected skill survived compaction and is still reported.
    assert CATEGORY_SKILLS in by_key
    assert breakdown.active_skill_count == 1


def test_provider_adaptation_drops_foreign_reasoning_and_non_vision_images():
    conv = Conversation(
        [
            Message(
                role="user",
                content=[
                    TextBlock(text="Look at this"),
                    ImageBlock(image_type="base64", media_type="image/png", data="X" * 50_000),
                ],
            ),
            Message(
                role="assistant",
                content=[
                    ThinkingBlock(thinking="Foreign chain of thought " * 100, signature="s"),
                    TextBlock(text="Answer."),
                ],
                usage_metadata={"provider": "anthropic"},
            ),
        ]
    )
    agent = _make_agent(
        provider="openai",
        model="gpt-4.1",
        conversation=conv,
        supports_vision=False,
    )

    breakdown = compute_context_breakdown(snapshot_agent_context(agent))
    by_key = {cat.key: cat for cat in breakdown.categories}

    assert breakdown.image_count == 0
    assert CATEGORY_IMAGES not in by_key
    assert CATEGORY_REASONING not in by_key
    assert CATEGORY_USER_MESSAGES in by_key
    assert CATEGORY_AGENT_RESPONSES in by_key


def test_special_token_literals_do_not_crash():
    conv = Conversation(
        [
            Message(
                role="user", content=[TextBlock(text="Tokenizer literals: <|endoftext|> <|channel|> <|im_start|>")]
            ),
            Message(role="assistant", content=[TextBlock(text="Handled <|endoftext|> safely.")]),
        ]
    )
    agent = _make_agent(conversation=conv)

    breakdown = compute_context_breakdown(snapshot_agent_context(agent))
    assert breakdown.total_tokens > 0


@pytest.mark.asyncio
async def test_authoritative_total_reconciliation_and_offline_fallback():
    conv = Conversation(
        [
            Message(role="user", content=[TextBlock(text="Hello")]),
            Message(role="assistant", content=[TextBlock(text="Hi there")]),
        ]
    )
    agent = _make_agent(conversation=conv, last_raw_count=4_200, residual_tokens=300)
    agent.count_current_context = AsyncMock(return_value=SimpleNamespace(input_tokens=5_000))

    provider_bd = await analyze_agent_context(agent, refresh_provider_count=True)
    assert provider_bd.total_tokens == 5_000
    assert provider_bd.total_source == "provider"

    live_bd = await analyze_agent_context(agent, is_live_turn=True, refresh_provider_count=True)
    assert live_bd.total_tokens == 4_500
    assert live_bd.total_source == "cached"
    assert live_bd.is_live_turn is True

    agent.count_current_context = AsyncMock(side_effect=RuntimeError("offline"))
    fallback_bd = await analyze_agent_context(agent, refresh_provider_count=True)
    assert fallback_bd.total_tokens == 4_500
    assert fallback_bd.total_source == "cached"


@pytest.mark.parametrize(
    ("total_tokens", "budget", "threshold", "cat_tokens", "total_cells"),
    [
        (0, 100_000, 0.95, (), 200),
        (500, 1_000_000, 0.95, (200, 200, 100), 200),
        (30_000, 100_000, 0.95, (5_000, 10_000, 15_000), 200),
        (92_000, 100_000, 0.95, (12_000, 30_000, 50_000), 200),
        (97_000, 100_000, 0.95, (17_000, 30_000, 50_000), 200),
        (120_000, 100_000, 0.95, (20_000, 40_000, 60_000), 200),
        (45_000, 100_000, 0.80, (5_000, 40_000), 160),
        (45_000, 100_000, 1.00, (5_000, 40_000), 240),
    ],
)
def test_allocate_grid_cells_invariants(
    total_tokens: int,
    budget: int,
    threshold: float,
    cat_tokens: tuple[int, ...],
    total_cells: int,
):
    keys = (CATEGORY_SYSTEM_PROMPT, CATEGORY_BUILTIN_TOOLS, CATEGORY_TOOL_USAGE)
    cats = tuple(ContextCategory(keys[i], keys[i], toks) for i, toks in enumerate(cat_tokens))
    autocompact_at = int(budget * threshold)
    breakdown = ContextBreakdown(
        provider="openai",
        model="gpt-test",
        thinking_effort=None,
        agent_role="Build",
        total_tokens=total_tokens,
        local_total_tokens=sum(cat_tokens),
        total_source="estimated",
        model_context_length=budget,
        model_completion_tokens=0,
        model_max_input_tokens=budget,
        strict_context_budget=False,
        compression_threshold=threshold,
        autocompact_at_tokens=autocompact_at,
        autocompact_buffer_tokens=max(0, budget - autocompact_at),
        free_tokens=max(0, budget - total_tokens),
        free_before_compact_tokens=max(0, autocompact_at - total_tokens),
        usage_percentage=(total_tokens / budget * 100.0),
        compacted_messages=0,
        compaction_summary_tokens=0,
        is_live_turn=False,
        categories=cats,
        detail_groups=(),
    )

    alloc = allocate_grid_cells(breakdown, total_cells=total_cells)
    assert sum(count for _, count in alloc) == total_cells
    assert all(count > 0 for _, count in alloc)

    by_key = dict(alloc)
    for cat in cats:
        assert by_key[cat.key] >= 1

    if total_tokens >= budget:
        assert CATEGORY_FREE not in by_key
        assert CATEGORY_AUTOCOMPACT_BUFFER not in by_key
    elif total_tokens > autocompact_at:
        assert CATEGORY_FREE not in by_key
        assert by_key.get(CATEGORY_AUTOCOMPACT_BUFFER, 0) > 0


def test_format_context_breakdown_text_includes_budget_reserve_and_capped():
    agent = _make_agent(
        context_length=200_000,
        completion_tokens=64_000,
        max_input_tokens=136_000,
        strict_context_budget=True,
        compression_threshold=0.90,
        conversation=Conversation([Message(role="user", content=[TextBlock(text="hello world")])]),
    )
    breakdown = compute_context_breakdown(snapshot_agent_context(agent), authoritative_tokens=12_000)
    text = format_context_breakdown_text(breakdown)

    assert "anthropic/claude-sonnet (medium)" in text
    assert "12k / 136k tokens" in text
    assert "[capped]" in text
    assert "Auto-compact at 90%" in text
    assert "200k window - 64k output reserve" in text
    assert "Token usage by category:" in text
    assert "Free space:" in text
    assert "Auto-compact buffer:" in text
