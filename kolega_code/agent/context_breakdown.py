from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from kolega_code.llm.models import (
    ImageBlock,
    Message,
    RedactedThinkingBlock,
    ResponsesReasoningBlock,
    TextBlock,
    ThinkingBlock,
    ToolCall,
    ToolDefinition,
    ToolResult,
    WebSearchCallBlock,
)

from .conversation import Conversation, adapt_history_for_provider

CATEGORY_SYSTEM_PROMPT = "system_prompt"
CATEGORY_BUILTIN_TOOLS = "builtin_tools"
CATEGORY_MCP_TOOLS = "mcp_tools"
CATEGORY_SKILLS = "skills"
CATEGORY_GUIDANCE_MEMORY = "guidance_memory"
CATEGORY_SYSTEM_REMINDERS = "system_reminders"
CATEGORY_COMPACTION_SUMMARY = "compaction_summary"
CATEGORY_USER_MESSAGES = "user_messages"
CATEGORY_AGENT_RESPONSES = "agent_responses"
CATEGORY_REASONING = "reasoning"
CATEGORY_TOOL_USAGE = "tool_usage"
CATEGORY_IMAGES = "images"
CATEGORY_SERVER_OVERHEAD = "server_overhead"
CATEGORY_FREE = "free"
CATEGORY_AUTOCOMPACT_BUFFER = "autocompact_buffer"

_SKILL_PROMPT_EXTENSION_ID = "cli-agent-skills"
_SYSTEM_REMINDER_RE = re.compile(r"<system-reminder\s+([^>]*)>(.*?)</system-reminder>", re.DOTALL)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
_SKILL_CONTENT_RE = re.compile(r'<skill_content\s+name="([^"]+)">(.*?)</skill_content>', re.DOTALL)
_ATTACHED_FILE_RE = re.compile(r'<attached-file\s+path="([^"]+)">(.*?)</attached-file>', re.DOTALL)
_SKILL_BULLET_RE = re.compile(r"^- `([^`]+)`:", re.MULTILINE)
_MCP_PREFIX = "mcp__"
_MCP_SEPARATOR = "__"
_HEAVIEST_TOOLS_LIMIT = 8


@dataclass(frozen=True, slots=True)
class ContextCategory:
    key: str
    label: str
    tokens: int
    detail: str = ""
    kind: str = "message"  # "static" | "message" | "overhead"


@dataclass(frozen=True, slots=True)
class ContextDetailItem:
    label: str
    tokens: int
    note: str = ""


@dataclass(frozen=True, slots=True)
class ContextDetailGroup:
    key: str
    title: str
    related_command: str = ""
    items: tuple[ContextDetailItem, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextSnapshot:
    provider: str
    model: str
    thinking_effort: Optional[str]
    agent_role: str
    model_context_length: int
    model_completion_tokens: int
    model_max_input_tokens: int
    strict_context_budget: bool
    compression_threshold: float
    system_prompt_text: str
    prompt_extensions: tuple[tuple[str, str, str], ...]
    tools: tuple[ToolDefinition, ...]
    mcp_tool_names: frozenset[str]
    mcp_tool_servers: dict[str, str]
    custom_agent_names: tuple[str, ...]
    custom_agent_catalog_text: str
    effective_history: list[Message]
    summary_message: Optional[Message]
    compacted_through: int
    supports_vision: bool
    edit_protocol: Optional[Any]
    residual_tokens: int
    cached_authoritative_tokens: Optional[int]
    has_conversation_turns: bool
    is_live_turn: bool = False


@dataclass(frozen=True, slots=True)
class ContextBreakdown:
    provider: str
    model: str
    thinking_effort: Optional[str]
    agent_role: str
    total_tokens: int
    local_total_tokens: int
    total_source: str  # "provider" | "cached" | "estimated"
    model_context_length: int
    model_completion_tokens: int
    model_max_input_tokens: int
    strict_context_budget: bool
    compression_threshold: float
    autocompact_at_tokens: int
    autocompact_buffer_tokens: int
    free_tokens: int
    free_before_compact_tokens: int
    usage_percentage: float
    compacted_messages: int
    compaction_summary_tokens: int
    is_live_turn: bool
    categories: tuple[ContextCategory, ...]
    detail_groups: tuple[ContextDetailGroup, ...]
    custom_agent_count: int = 0
    mcp_server_count: int = 0
    active_skill_count: int = 0
    image_count: int = 0


def format_token_count(tokens: int) -> str:
    if tokens < 1_000:
        return str(tokens)
    if tokens < 999_950:
        return f"{tokens / 1_000:.1f}k".replace(".0k", "k")
    return f"{tokens / 1_000_000:.1f}M".replace(".0M", "M")


def _provider_str(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "")


def _latest_billed_input_tokens(messages: Sequence[Message]) -> Optional[int]:
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        usage = getattr(message, "usage", None)
        if usage is not None and getattr(usage, "reported", False):
            inp = getattr(usage, "input_tokens", None)
            if isinstance(inp, int) and inp > 0:
                return inp
        metadata = getattr(message, "usage_metadata", None) or {}
        for key in ("prompt_tokens", "input_tokens", "prompt_token_count"):
            val = metadata.get(key)
            if isinstance(val, int) and val > 0:
                return val
    return None


def _is_session_bootstrap_only(messages: Sequence[Message]) -> bool:
    if not messages:
        return True
    for message in messages:
        if message.role != "user":
            return False
        if isinstance(message.content, str):
            text = message.content.strip()
            if not text.startswith('<system-reminder source="session">'):
                return False
        elif isinstance(message.content, list):
            for block in message.content:
                if not isinstance(block, TextBlock):
                    return False
                if not block.text.strip().startswith('<system-reminder source="session">'):
                    return False
        else:
            return False
    return True


def _agent_role_label(agent: Any) -> str:
    agent_name = str(getattr(agent, "agent_name", "") or "").lower()
    if "plan" in agent_name:
        return "Plan"
    if "coder" in agent_name or "build" in agent_name:
        return "Build"
    if agent_name:
        return agent_name.replace("_", " ").title()
    return ""


def snapshot_agent_context(agent: Any, *, is_live_turn: bool = False) -> ContextSnapshot:
    """Capture immutable context inputs on the event loop before off-thread analysis."""
    model_config = getattr(agent, "primary_model_config", None)
    provider = _provider_str(getattr(model_config, "provider", ""))
    model = str(getattr(model_config, "model", "") or "")
    thinking_effort = getattr(model_config, "thinking_effort", None)
    if thinking_effort is not None:
        thinking_effort = str(thinking_effort)

    model_max_input = int(getattr(agent, "model_max_input_tokens", 0) or 0)
    model_context_length = int(getattr(agent, "model_context_length", model_max_input) or model_max_input)
    model_completion_tokens = int(getattr(agent, "model_completion_tokens", 0) or 0)
    strict_budget = bool(getattr(agent, "strict_context_budget", False))
    threshold_raw = getattr(agent, "history_compression_threshold", 0.95)
    threshold = float(threshold_raw) if isinstance(threshold_raw, (int, float)) else 0.95
    threshold = max(0.0, min(1.0, threshold))

    system_prompt = getattr(agent, "system_prompt", None)
    system_prompt_text = ""
    if system_prompt is not None:
        get_text = getattr(system_prompt, "get_text_content", None)
        if callable(get_text):
            system_prompt_text = str(get_text() or "")
        elif isinstance(system_prompt, str):
            system_prompt_text = system_prompt

    prompt_extensions: list[tuple[str, str, str]] = []
    for ext in getattr(agent, "prompt_extensions", None) or ():
        ext_id = str(getattr(ext, "id", "") or "")
        ext_title = str(getattr(ext, "title", "") or ext_id)
        ext_markdown = str(getattr(ext, "markdown", "") or "")
        if ext_id or ext_markdown:
            prompt_extensions.append((ext_id, ext_title, ext_markdown))

    tool_collection = getattr(agent, "tool_collection", None)
    tools: tuple[ToolDefinition, ...] = ()
    mcp_tool_names: set[str] = set()
    mcp_tool_servers: dict[str, str] = {}
    if tool_collection is not None:
        get_tool_list = getattr(tool_collection, "get_tool_list", None)
        if callable(get_tool_list):
            raw_tools: Any = get_tool_list()
            if isinstance(raw_tools, (list, tuple)):
                tools = tuple(raw_tools)
        for name in getattr(tool_collection, "mcp_tools", None) or ():
            mcp_tool_names.add(str(name))
        for name, source in (getattr(tool_collection, "extension_tool_sources", None) or {}).items():
            if source == "mcp":
                mcp_tool_names.add(str(name))
        for name, meta in (getattr(tool_collection, "extension_metadata", None) or {}).items():
            if isinstance(meta, dict):
                server_id = meta.get("mcp_server_id")
                if isinstance(server_id, str) and server_id:
                    mcp_tool_names.add(str(name))
                    mcp_tool_servers[str(name)] = server_id

    custom_agent_names: tuple[str, ...] = ()
    custom_agent_catalog_text = ""
    catalog = getattr(agent, "custom_agent_catalog", None)
    if catalog is not None:
        has_agents = getattr(catalog, "has_agents", None)
        if callable(has_agents) and has_agents():
            names_fn = getattr(catalog, "names", None)
            if callable(names_fn):
                raw_names: Any = names_fn()
                if isinstance(raw_names, (list, tuple, set, frozenset)):
                    custom_agent_names = tuple(str(n) for n in raw_names)
            model_cat_fn = getattr(catalog, "model_catalog", None)
            if callable(model_cat_fn):
                custom_agent_catalog_text = str(model_cat_fn() or "")

    conversation = getattr(agent, "conversation", None)
    summary_message: Optional[Message] = None
    compacted_through = 0
    effective_history: list[Message] = []
    if isinstance(conversation, Conversation):
        effective_history = list(conversation.effective_history())
        summary_message = conversation.summary
        compacted_through = int(conversation.compacted_through or 0)
    else:
        raw_history: Any = getattr(agent, "history", None) or ()
        effective_history = list(raw_history)

    residual_raw = getattr(agent, "_context_residual_tokens", 0)
    residual_tokens = max(0, int(residual_raw)) if isinstance(residual_raw, int) else 0
    last_raw = getattr(agent, "_last_raw_context_count", None)
    cached_authoritative: Optional[int] = None
    if isinstance(last_raw, int) and last_raw > 0:
        cached_authoritative = last_raw + residual_tokens
    else:
        billed = _latest_billed_input_tokens(effective_history)
        if billed is not None:
            cached_authoritative = billed

    has_turns = not _is_session_bootstrap_only(effective_history)

    return ContextSnapshot(
        provider=provider,
        model=model,
        thinking_effort=thinking_effort,
        agent_role=_agent_role_label(agent),
        model_context_length=model_context_length,
        model_completion_tokens=model_completion_tokens,
        model_max_input_tokens=model_max_input,
        strict_context_budget=strict_budget,
        compression_threshold=threshold,
        system_prompt_text=system_prompt_text,
        prompt_extensions=tuple(prompt_extensions),
        tools=tools,
        mcp_tool_names=frozenset(mcp_tool_names),
        mcp_tool_servers=mcp_tool_servers,
        custom_agent_names=custom_agent_names,
        custom_agent_catalog_text=custom_agent_catalog_text,
        effective_history=effective_history,
        summary_message=summary_message,
        compacted_through=compacted_through,
        supports_vision=bool(getattr(agent, "supports_vision", False)),
        edit_protocol=getattr(agent, "edit_protocol", None),
        residual_tokens=residual_tokens,
        cached_authoritative_tokens=cached_authoritative,
        has_conversation_turns=has_turns,
        is_live_turn=is_live_turn,
    )


def _tool_wire_payload(tool: ToolDefinition) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": tool.name,
        "description": tool.description,
    }
    if tool.input_schema is not None:
        payload["input_schema"] = tool.input_schema
    elif tool.freeform_format is not None:
        payload["format"] = tool.freeform_format
    return payload


def _mcp_server_id(tool_name: str, server_map: dict[str, str]) -> str:
    if tool_name in server_map:
        return server_map[tool_name]
    if tool_name.startswith(_MCP_PREFIX):
        rest = tool_name[len(_MCP_PREFIX) :]
        server, sep, _ = rest.partition(_MCP_SEPARATOR)
        if sep and server:
            return server
    return "mcp"


def _parse_attrs(attr_str: str) -> dict[str, str]:
    return {match.group(1): match.group(2) for match in _ATTR_RE.finditer(attr_str)}


def _is_reminder_block(text: str) -> bool:
    return text.lstrip().startswith('<system-reminder source="')


def compute_context_breakdown(
    snapshot: ContextSnapshot,
    *,
    authoritative_tokens: Optional[int] = None,
    authoritative_source: str = "provider",
) -> ContextBreakdown:
    """Compute categorized token usage locally using cl100k_base (pure CPU, thread-safe)."""
    from kolega_code.llm.providers._token_encoding import estimate_image_tokens, get_counting_encoding

    encoding = get_counting_encoding("cl100k_base")

    def count_text(text: str) -> int:
        if not text:
            return 0
        return len(encoding.encode(text))

    # 1. System prompt, skills catalog, and prompt extensions
    full_system_tokens = count_text(snapshot.system_prompt_text)
    skills_catalog_tokens = 0
    available_skills_count = 0
    extension_items: list[ContextDetailItem] = []

    for ext_id, ext_title, ext_markdown in snapshot.prompt_extensions:
        if not ext_markdown:
            continue
        rendered_block = f"### {ext_title}\n\n{ext_markdown}"
        present = ext_markdown in snapshot.system_prompt_text
        if not present:
            continue
        ext_tokens = count_text(rendered_block)
        if ext_id == _SKILL_PROMPT_EXTENSION_ID:
            skills_catalog_tokens += min(ext_tokens, full_system_tokens)
            available_skills_count += len(_SKILL_BULLET_RE.findall(ext_markdown))
        else:
            extension_items.append(ContextDetailItem(label=ext_title or ext_id, tokens=ext_tokens))

    skills_catalog_tokens = min(skills_catalog_tokens, full_system_tokens)
    system_prompt_tokens = max(0, full_system_tokens - skills_catalog_tokens)

    # 2. Tool definitions: built-in vs MCP (per server)
    builtin_tools_tokens = 0
    builtin_tools_count = 0
    mcp_tools_tokens = 0
    mcp_server_stats: dict[str, list[int]] = {}  # server_id -> [tool_count, tokens]
    dispatch_agent_present = False

    for tool in snapshot.tools:
        if tool.name == "dispatch_agent":
            dispatch_agent_present = True
        wire_tokens = count_text(json.dumps(_tool_wire_payload(tool)))
        is_mcp = (
            tool.name in snapshot.mcp_tool_names
            or tool.name in snapshot.mcp_tool_servers
            or tool.name.startswith(_MCP_PREFIX)
        )
        if is_mcp:
            mcp_tools_tokens += wire_tokens
            server_id = _mcp_server_id(tool.name, snapshot.mcp_tool_servers)
            entry = mcp_server_stats.setdefault(server_id, [0, 0])
            entry[0] += 1
            entry[1] += wire_tokens
        else:
            builtin_tools_tokens += wire_tokens
            builtin_tools_count += 1

    # 3. Effective history (provider-adapted copy so non-vision & foreign reasoning match the wire)
    history = list(snapshot.effective_history)
    if history and snapshot.provider and snapshot.model:
        history = Conversation(history).repaired(history)
        history = adapt_history_for_provider(
            history,
            target_provider=snapshot.provider,
            target_model=snapshot.model,
            supports_vision=snapshot.supports_vision,
            target_edit_protocol=snapshot.edit_protocol,
        )

    guidance_memory_tokens = 0
    guidance_items: dict[str, int] = {}
    system_reminders_tokens = 0
    reminder_sources: dict[str, int] = {}
    compaction_summary_tokens = 0
    user_messages_tokens = 0
    agent_responses_tokens = 0
    reasoning_tokens = 0
    tool_call_tokens = 0
    tool_result_tokens = 0
    image_tokens = 0
    image_count = 0
    activated_skill_tokens: dict[str, int] = {}
    attached_files: dict[str, int] = {}
    # tool_name -> [call_count, call_tokens, result_tokens]
    tool_usage_stats: dict[str, list[int]] = {}

    summary_text = snapshot.summary_message.get_text_content() if snapshot.summary_message is not None else ""
    summary_matched = False

    def record_skill_blocks(text: str) -> str:
        """Strip and count <skill_content> blocks, returning remaining non-skill text."""
        if "<skill_content " not in text:
            return text
        parts: list[str] = []
        last = 0
        for match in _SKILL_CONTENT_RE.finditer(text):
            parts.append(text[last : match.start()])
            skill_name = match.group(1)
            tokens = count_text(match.group(0))
            activated_skill_tokens[skill_name] = activated_skill_tokens.get(skill_name, 0) + tokens
            last = match.end()
        parts.append(text[last:])
        return "".join(parts)

    def count_user_text(text: str) -> None:
        nonlocal guidance_memory_tokens, system_reminders_tokens, user_messages_tokens
        if not text:
            return
        if _is_reminder_block(text):
            last = 0
            for match in _SYSTEM_REMINDER_RE.finditer(text):
                leftover = text[last : match.start()].strip()
                if leftover:
                    user_messages_tokens += count_text(leftover)
                attrs = _parse_attrs(match.group(1))
                source = attrs.get("source", "system")
                path = attrs.get("path", "")
                block_tokens = count_text(match.group(0))
                if source == "guidance":
                    guidance_memory_tokens += block_tokens
                    label = path or "Repository guidance"
                    guidance_items[label] = guidance_items.get(label, 0) + block_tokens
                elif source == "memory":
                    guidance_memory_tokens += block_tokens
                    label = "MEMORY.md (startup index)"
                    guidance_items[label] = guidance_items.get(label, 0) + block_tokens
                else:
                    system_reminders_tokens += block_tokens
                    reminder_sources[source] = reminder_sources.get(source, 0) + block_tokens
                last = match.end()
            trailing = text[last:].strip()
            if trailing:
                user_messages_tokens += count_text(trailing)
            return

        remainder = record_skill_blocks(text)
        if "<attached-file " in remainder:
            for match in _ATTACHED_FILE_RE.finditer(remainder):
                file_path = match.group(1)
                attached_files[file_path] = attached_files.get(file_path, 0) + count_text(match.group(0))
        if remainder.strip():
            user_messages_tokens += count_text(remainder)

    for message in history:
        is_summary = False
        if snapshot.compacted_through > 0 and not summary_matched and message.role == "user":
            if message is snapshot.summary_message:
                is_summary = True
            elif summary_text and message.get_text_content() == summary_text:
                is_summary = True
        if is_summary:
            summary_matched = True
            compaction_summary_tokens += count_text(message.get_text_content())
            continue

        content = message.content
        if isinstance(content, str):
            if message.role == "user":
                count_user_text(content)
            elif message.role == "assistant":
                agent_responses_tokens += count_text(content)
            continue

        if not isinstance(content, list):
            continue

        for block in content:
            if isinstance(block, TextBlock):
                if message.role == "user":
                    count_user_text(block.text)
                elif message.role == "assistant":
                    agent_responses_tokens += count_text(block.text)
            elif isinstance(block, dict) and "text" in block:
                text_val = str(block["text"] or "")
                if message.role == "user":
                    count_user_text(text_val)
                elif message.role == "assistant":
                    agent_responses_tokens += count_text(text_val)
            elif isinstance(block, ImageBlock):
                image_count += 1
                image_tokens += estimate_image_tokens(len(block.data))
            elif isinstance(block, ThinkingBlock):
                reasoning_tokens += count_text(block.thinking or "")
                if block.signature:
                    reasoning_tokens += count_text(str(block.signature))
            elif isinstance(block, RedactedThinkingBlock):
                reasoning_tokens += count_text(str(block.data or ""))
            elif isinstance(block, ResponsesReasoningBlock):
                reasoning_tokens += count_text(str(block.encrypted_content or ""))
                for part in block.summary:
                    reasoning_tokens += count_text(str(part))
                for part in block.content:
                    reasoning_tokens += count_text(str(part))
            elif isinstance(block, WebSearchCallBlock):
                call_toks = (
                    count_text(str(block.item_id or ""))
                    + count_text(str(block.status or ""))
                    + count_text(json.dumps(block.action, sort_keys=True))
                )
                tool_call_tokens += call_toks
                stats = tool_usage_stats.setdefault("web_search", [0, 0, 0])
                stats[0] += 1
                stats[1] += call_toks
            elif isinstance(block, ToolCall):
                if isinstance(block.input, str):
                    args_text = block.input
                else:
                    try:
                        args_text = json.dumps(block.input, ensure_ascii=False)
                    except (TypeError, ValueError):
                        args_text = str(block.input)
                call_toks = count_text(block.name) + count_text(args_text) + count_text(str(block.id or ""))
                tool_call_tokens += call_toks
                stats = tool_usage_stats.setdefault(block.name, [0, 0, 0])
                stats[0] += 1
                stats[1] += call_toks
            elif isinstance(block, ToolResult):
                result_name = block.name or "tool"
                res_toks = 0
                if isinstance(block.content, str):
                    remainder = record_skill_blocks(block.content)
                    if remainder.strip():
                        res_toks += count_text(remainder)
                elif isinstance(block.content, list):
                    for inner in block.content:
                        if isinstance(inner, ImageBlock):
                            image_count += 1
                            image_tokens += estimate_image_tokens(len(inner.data))
                        elif hasattr(inner, "data") and hasattr(inner, "media_type"):
                            image_count += 1
                            image_tokens += estimate_image_tokens(len(str(getattr(inner, "data", ""))))
                        elif hasattr(inner, "text") and getattr(inner, "text", None):
                            remainder = record_skill_blocks(str(getattr(inner, "text", "")))
                            if remainder.strip():
                                res_toks += count_text(remainder)
                if res_toks > 0:
                    tool_result_tokens += res_toks
                    stats = tool_usage_stats.setdefault(result_name, [0, 0, 0])
                    stats[2] += res_toks

    skills_total_tokens = skills_catalog_tokens + sum(activated_skill_tokens.values())
    tool_usage_total = tool_call_tokens + tool_result_tokens
    server_overhead_tokens = snapshot.residual_tokens

    # 4. Assemble ordered categories (omit zero-token optional categories)
    categories: list[ContextCategory] = []
    if system_prompt_tokens > 0:
        categories.append(ContextCategory(CATEGORY_SYSTEM_PROMPT, "System prompt", system_prompt_tokens, kind="static"))
    if builtin_tools_tokens > 0:
        categories.append(
            ContextCategory(
                CATEGORY_BUILTIN_TOOLS,
                "Built-in tools",
                builtin_tools_tokens,
                detail=f"{builtin_tools_count} tools",
                kind="static",
            )
        )
    if mcp_tools_tokens > 0:
        srv_n = len(mcp_server_stats)
        categories.append(
            ContextCategory(
                CATEGORY_MCP_TOOLS,
                "MCP tools",
                mcp_tools_tokens,
                detail=f"{srv_n} {'server' if srv_n == 1 else 'servers'}",
                kind="static",
            )
        )
    if skills_total_tokens > 0:
        skill_detail = ""
        if activated_skill_tokens:
            n_act = len(activated_skill_tokens)
            skill_detail = f"{n_act} active"
        elif available_skills_count:
            skill_detail = f"{available_skills_count} catalog"
        categories.append(
            ContextCategory(CATEGORY_SKILLS, "Skills", skills_total_tokens, detail=skill_detail, kind="static")
        )
    if guidance_memory_tokens > 0:
        categories.append(
            ContextCategory(CATEGORY_GUIDANCE_MEMORY, "Guidance & memory", guidance_memory_tokens, kind="static")
        )
    if system_reminders_tokens > 0:
        categories.append(
            ContextCategory(CATEGORY_SYSTEM_REMINDERS, "System reminders", system_reminders_tokens, kind="message")
        )
    if compaction_summary_tokens > 0:
        categories.append(
            ContextCategory(
                CATEGORY_COMPACTION_SUMMARY,
                "Compaction summary",
                compaction_summary_tokens,
                detail=f"{snapshot.compacted_through} msgs folded",
                kind="message",
            )
        )
    if user_messages_tokens > 0:
        categories.append(ContextCategory(CATEGORY_USER_MESSAGES, "User messages", user_messages_tokens))
    if agent_responses_tokens > 0:
        categories.append(ContextCategory(CATEGORY_AGENT_RESPONSES, "Agent responses", agent_responses_tokens))
    if reasoning_tokens > 0:
        categories.append(ContextCategory(CATEGORY_REASONING, "Reasoning", reasoning_tokens))
    if tool_usage_total > 0:
        categories.append(
            ContextCategory(
                CATEGORY_TOOL_USAGE,
                "Tool calls & results",
                tool_usage_total,
                detail=f"calls {format_token_count(tool_call_tokens)} · results {format_token_count(tool_result_tokens)}",
            )
        )
    if image_tokens > 0:
        categories.append(
            ContextCategory(
                CATEGORY_IMAGES,
                "Images",
                image_tokens,
                detail=f"{image_count} {'image' if image_count == 1 else 'images'}",
            )
        )
    if server_overhead_tokens > 0:
        categories.append(
            ContextCategory(
                CATEGORY_SERVER_OVERHEAD,
                "Server / residual overhead",
                server_overhead_tokens,
                kind="overhead",
            )
        )

    local_total_tokens = sum(category.tokens for category in categories)

    # 5. Reconcile headline total with authoritative count when conversation has real turns
    total_tokens = local_total_tokens
    total_source = "estimated"
    if authoritative_tokens is not None and authoritative_tokens > 0 and snapshot.has_conversation_turns:
        total_tokens = authoritative_tokens
        total_source = authoritative_source
    elif (
        authoritative_tokens is None
        and snapshot.cached_authoritative_tokens is not None
        and snapshot.cached_authoritative_tokens > 0
        and snapshot.has_conversation_turns
    ):
        total_tokens = snapshot.cached_authoritative_tokens
        total_source = "cached"

    # 6. Budget geometry
    budget = snapshot.model_max_input_tokens
    autocompact_at = int(budget * snapshot.compression_threshold) if budget > 0 else 0
    autocompact_buffer = max(0, budget - autocompact_at) if budget > 0 else 0
    free_tokens = max(0, budget - total_tokens) if budget > 0 else 0
    free_before_compact = max(0, autocompact_at - total_tokens) if budget > 0 else 0
    usage_percentage = (total_tokens / budget) * 100.0 if budget > 0 else 0.0

    # 7. Actionable detail groups
    detail_groups: list[ContextDetailGroup] = []

    if guidance_items:
        items = tuple(
            ContextDetailItem(label=label, tokens=toks)
            for label, toks in sorted(guidance_items.items(), key=lambda pair: (-pair[1], pair[0]))
        )
        detail_groups.append(
            ContextDetailGroup(
                key="guidance_memory",
                title="Guidance & Memory",
                related_command="/memory",
                items=items,
            )
        )

    if skills_total_tokens > 0:
        skill_items: list[ContextDetailItem] = []
        if skills_catalog_tokens > 0:
            cat_note = f"{available_skills_count} available" if available_skills_count else ""
            skill_items.append(ContextDetailItem(label="Skills catalog", tokens=skills_catalog_tokens, note=cat_note))
        for skill_name, toks in sorted(activated_skill_tokens.items(), key=lambda pair: (-pair[1], pair[0])):
            skill_items.append(ContextDetailItem(label=f"/{skill_name}", tokens=toks, note="active · protected"))
        detail_groups.append(
            ContextDetailGroup(
                key="skills",
                title="Skills",
                related_command="/skills",
                items=tuple(skill_items),
            )
        )

    if mcp_server_stats:
        mcp_items = tuple(
            ContextDetailItem(
                label=server_id,
                tokens=toks,
                note=f"{count} {'tool' if count == 1 else 'tools'}",
            )
            for server_id, (count, toks) in sorted(mcp_server_stats.items(), key=lambda pair: (-pair[1][1], pair[0]))
        )
        detail_groups.append(
            ContextDetailGroup(
                key="mcp_servers",
                title="MCP Servers",
                related_command="/mcp",
                items=mcp_items,
            )
        )

    if snapshot.custom_agent_names and dispatch_agent_present:
        custom_toks = count_text(snapshot.custom_agent_catalog_text)
        label = ", ".join(snapshot.custom_agent_names[:6])
        if len(snapshot.custom_agent_names) > 6:
            label += f" (+{len(snapshot.custom_agent_names) - 6} more)"
        detail_groups.append(
            ContextDetailGroup(
                key="custom_agents",
                title="Custom Agents",
                related_command="/agents",
                items=(
                    ContextDetailItem(
                        label=label,
                        tokens=custom_toks,
                        note=f"{len(snapshot.custom_agent_names)} in dispatch_agent",
                    ),
                ),
            )
        )

    if tool_usage_stats:
        ranked_tools = sorted(
            tool_usage_stats.items(),
            key=lambda pair: (-(pair[1][1] + pair[1][2]), pair[0]),
        )
        heaviest_items = []
        for tool_name, (calls, call_toks, res_toks) in ranked_tools[:_HEAVIEST_TOOLS_LIMIT]:
            combined = call_toks + res_toks
            if combined <= 0:
                continue
            call_word = "call" if calls == 1 else "calls"
            if calls > 0 and res_toks > 0:
                note = (
                    f"{calls} {call_word} · calls {format_token_count(call_toks)} · "
                    f"results {format_token_count(res_toks)}"
                )
            elif calls > 0:
                note = f"{calls} {call_word}"
            else:
                note = "results"
            heaviest_items.append(ContextDetailItem(label=tool_name, tokens=combined, note=note))
        if heaviest_items:
            detail_groups.append(
                ContextDetailGroup(
                    key="heaviest_tools",
                    title="Heaviest Tool Usage in History",
                    items=tuple(heaviest_items),
                )
            )

    if attached_files:
        file_items = tuple(
            ContextDetailItem(label=path, tokens=toks)
            for path, toks in sorted(attached_files.items(), key=lambda pair: (-pair[1], pair[0]))
        )
        detail_groups.append(
            ContextDetailGroup(
                key="attachments",
                title="Attached Files",
                items=file_items,
            )
        )

    if extension_items:
        extension_items.sort(key=lambda item: (-item.tokens, item.label))
        detail_groups.append(
            ContextDetailGroup(
                key="prompt_extensions",
                title="Prompt Extensions",
                items=tuple(extension_items),
            )
        )

    return ContextBreakdown(
        provider=snapshot.provider,
        model=snapshot.model,
        thinking_effort=snapshot.thinking_effort,
        agent_role=snapshot.agent_role,
        total_tokens=total_tokens,
        local_total_tokens=local_total_tokens,
        total_source=total_source,
        model_context_length=snapshot.model_context_length,
        model_completion_tokens=snapshot.model_completion_tokens,
        model_max_input_tokens=budget,
        strict_context_budget=snapshot.strict_context_budget,
        compression_threshold=snapshot.compression_threshold,
        autocompact_at_tokens=autocompact_at,
        autocompact_buffer_tokens=autocompact_buffer,
        free_tokens=free_tokens,
        free_before_compact_tokens=free_before_compact,
        usage_percentage=usage_percentage,
        compacted_messages=snapshot.compacted_through if snapshot.summary_message is not None else 0,
        compaction_summary_tokens=compaction_summary_tokens,
        is_live_turn=snapshot.is_live_turn,
        categories=tuple(categories),
        detail_groups=tuple(detail_groups),
        custom_agent_count=len(snapshot.custom_agent_names) if dispatch_agent_present else 0,
        mcp_server_count=len(mcp_server_stats),
        active_skill_count=len(activated_skill_tokens),
        image_count=image_count,
    )


async def analyze_agent_context(
    agent: Any,
    *,
    is_live_turn: bool = False,
    refresh_provider_count: bool = False,
) -> ContextBreakdown:
    """Snapshot ``agent`` on the event loop and compute its breakdown off-thread."""
    snapshot = snapshot_agent_context(agent, is_live_turn=is_live_turn)

    authoritative_tokens: Optional[int] = None
    authoritative_source = "cached"
    if refresh_provider_count and not is_live_turn and snapshot.has_conversation_turns:
        count_fn: Any = getattr(agent, "count_current_context", None)
        if callable(count_fn):
            try:
                pending: Any = count_fn()
                token_count = await pending
                counted = getattr(token_count, "input_tokens", None)
                if isinstance(counted, int) and counted > 0:
                    authoritative_tokens = counted
                    authoritative_source = "provider"
            except Exception:
                # Fall back to cached or local estimates if provider counting fails or is offline.
                authoritative_tokens = None

    return await asyncio.to_thread(
        compute_context_breakdown,
        snapshot,
        authoritative_tokens=authoritative_tokens,
        authoritative_source=authoritative_source,
    )


def allocate_grid_cells(breakdown: ContextBreakdown, total_cells: int = 200) -> list[tuple[str, int]]:
    """Allocate exactly ``total_cells`` cells across categories, free space, and auto-compact buffer."""
    if total_cells <= 0:
        return []

    budget = breakdown.model_max_input_tokens
    if budget <= 0:
        return [(CATEGORY_FREE, total_cells)]

    buffer_tokens = breakdown.autocompact_buffer_tokens
    buffer_cells = round(total_cells * buffer_tokens / budget)
    if buffer_tokens > 0 and buffer_cells == 0 and total_cells >= 20:
        buffer_cells = 1
    buffer_cells = min(buffer_cells, max(0, total_cells - 1))

    active_categories = [category for category in breakdown.categories if category.tokens > 0]
    if not active_categories or breakdown.total_tokens <= 0:
        free_cells = total_cells - buffer_cells
        result: list[tuple[str, int]] = []
        if free_cells > 0:
            result.append((CATEGORY_FREE, free_cells))
        if buffer_cells > 0:
            result.append((CATEGORY_AUTOCOMPACT_BUFFER, buffer_cells))
        return result

    used_ratio = min(1.0, breakdown.total_tokens / budget)
    used_cells = min(total_cells, round(total_cells * used_ratio))
    # Guarantee every visible non-zero category appears in the grid.
    min_cells = min(len(active_categories), total_cells)
    used_cells = max(used_cells, min_cells)

    allocations: dict[str, int] = {}
    if used_cells <= len(active_categories):
        # More active categories than cells: pick the largest categories first.
        ranked = sorted(active_categories, key=lambda c: (-c.tokens, c.key))
        chosen = {c.key for c in ranked[:used_cells]}
        for category in active_categories:
            if category.key in chosen:
                allocations[category.key] = 1
    else:
        for category in active_categories:
            allocations[category.key] = 1
        extra_pool = used_cells - len(active_categories)
        cat_token_sum = sum(category.tokens for category in active_categories)
        exact_extras = [extra_pool * category.tokens / cat_token_sum for category in active_categories]
        floors = [int(value) for value in exact_extras]
        remainders = [value - floor for value, floor in zip(exact_extras, floors)]
        for category, floor in zip(active_categories, floors):
            allocations[category.key] += floor
        leftover = extra_pool - sum(floors)
        if leftover > 0:
            order = sorted(
                range(len(active_categories)),
                key=lambda idx: (-remainders[idx], -active_categories[idx].tokens, idx),
            )
            for idx in order[:leftover]:
                allocations[active_categories[idx].key] += 1

    remaining = total_cells - sum(allocations.values())
    actual_buffer_cells = min(remaining, buffer_cells)
    free_cells = remaining - actual_buffer_cells

    ordered: list[tuple[str, int]] = [
        (category.key, allocations[category.key])
        for category in active_categories
        if allocations.get(category.key, 0) > 0
    ]
    if free_cells > 0:
        ordered.append((CATEGORY_FREE, free_cells))
    if actual_buffer_cells > 0:
        ordered.append((CATEGORY_AUTOCOMPACT_BUFFER, actual_buffer_cells))
    return ordered


def format_context_breakdown_text(breakdown: ContextBreakdown) -> str:
    """Plain-text report shared by headless /context and TUI clipboard export."""
    model_id = f"{breakdown.provider}/{breakdown.model}" if breakdown.provider and breakdown.model else breakdown.model
    if not model_id:
        model_id = "Context Usage"
    effort_suffix = f" ({breakdown.thinking_effort})" if breakdown.thinking_effort else ""
    budget = breakdown.model_max_input_tokens
    capped = " [capped]" if breakdown.strict_context_budget else ""

    lines: list[str] = []
    if budget > 0:
        lines.append(
            f"{model_id}{effort_suffix} · "
            f"{format_token_count(breakdown.total_tokens)} / {format_token_count(budget)} tokens "
            f"({breakdown.usage_percentage:.1f}%){capped}"
        )
        compact_pct = breakdown.compression_threshold * 100.0
        lines.append(
            f"Auto-compact at {compact_pct:.0f}% ({format_token_count(breakdown.autocompact_at_tokens)} tokens) · "
            f"{format_token_count(breakdown.free_before_compact_tokens)} free before compact"
        )
        if breakdown.model_context_length > budget:
            reserved = breakdown.model_context_length - budget
            lines.append(
                f"Input budget: {format_token_count(budget)} "
                f"({format_token_count(breakdown.model_context_length)} window - "
                f"{format_token_count(reserved)} output reserve)"
            )
    else:
        lines.append(f"{model_id}{effort_suffix} · {format_token_count(breakdown.total_tokens)} tokens{capped}")

    status_parts: list[str] = []
    if breakdown.agent_role:
        status_parts.append(f"{breakdown.agent_role} mode")
    if breakdown.compacted_messages > 0:
        status_parts.append(
            f"Compacted: {breakdown.compacted_messages} older messages "
            f"({format_token_count(breakdown.compaction_summary_tokens)} summary)"
        )
    if breakdown.is_live_turn:
        status_parts.append("Live snapshot (turn in progress)")
    if status_parts:
        lines.append(" · ".join(status_parts))

    lines.append("")
    lines.append("Token usage by category:")
    for category in breakdown.categories:
        pct = (category.tokens / budget * 100.0) if budget > 0 else 0.0
        detail = f" [{category.detail}]" if category.detail else ""
        lines.append(f"  - {category.label}: {format_token_count(category.tokens)} tokens ({pct:.1f}%){detail}")
    if budget > 0:
        free_before = breakdown.free_before_compact_tokens
        free_pct = free_before / budget * 100.0
        lines.append(f"  - Free space: {format_token_count(free_before)} tokens ({free_pct:.1f}%)")
        if breakdown.autocompact_buffer_tokens > 0:
            remaining_buffer = min(breakdown.free_tokens, breakdown.autocompact_buffer_tokens)
            buf_pct = remaining_buffer / budget * 100.0
            breached = " [threshold exceeded]" if breakdown.total_tokens > breakdown.autocompact_at_tokens else ""
            lines.append(
                f"  - Auto-compact buffer: {format_token_count(remaining_buffer)} tokens ({buf_pct:.1f}%){breached}"
            )

    if breakdown.total_source in {"provider", "cached"} and breakdown.local_total_tokens > 0:
        delta = abs(breakdown.total_tokens - breakdown.local_total_tokens)
        if delta > max(200, int(breakdown.total_tokens * 0.05)):
            lines.append(
                f"  Note: category breakdown estimated locally via cl100k "
                f"(~{format_token_count(breakdown.local_total_tokens)} tokens); "
                f"total anchored to {breakdown.total_source} count."
            )

    for group in breakdown.detail_groups:
        lines.append("")
        suffix = f" · {group.related_command}" if group.related_command else ""
        lines.append(f"{group.title}{suffix}:")
        for item in group.items:
            note = f" ({item.note})" if item.note else ""
            lines.append(f"  - {item.label}: {format_token_count(item.tokens)} tokens{note}")

    return "\n".join(lines)
