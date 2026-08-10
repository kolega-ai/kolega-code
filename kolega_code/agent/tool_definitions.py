"""Explicit definitions for every built-in model-facing tool.

Each built-in tool's wire artifact is declared here as data: the description
lives beside the prompt assets in ``prompt_templates/tools/<key>.md`` (it is
tuned prompt surface, like a system prompt), and the input schema is a literal
dict in ``BUILTIN_TOOL_SPECS``. The handler callable is only the
implementation; nothing about what the model sees is derived from signatures
or docstrings.

``dispatch_agent`` is the one template-plus-data definition: its base
description is a static asset, while the agent_type catalog suffix and enum
schema are composed per collection from explicit data
(``ORDINARY_MODEL_OVERRIDE_SCHEMA``, ``dispatch_agent_input_schema``, and
``ToolCollection._DISPATCH_AGENT_TYPE_LINES``).

Spec keys are handler names, not wire names: the edit protocols expose the
same wire tool (``edit``/``write``) with different definitions, so the specs
are keyed by ``claude_edit``, ``hashline_edit``, and friends, and the caller
supplies the wire name.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence

from kolega_code.llm.models import ToolDefinition

from .tool_backend.browser_tool import BROWSER_TOOL_SCHEMAS
from .tool_backend.codex_patch import CODEX_APPLY_PATCH_GRAMMAR
from .tool_backend.workflow_tool import RUN_WORKFLOW_INPUT_SCHEMA

TOOL_DESCRIPTION_DIR = Path(__file__).resolve().parent / "prompt_templates" / "tools"


@lru_cache(maxsize=None)
def tool_description_asset(key: str) -> str:
    """The exact wire description asset for a declared tool.

    Assets store the description plus a single trailing newline (so editors and
    the end-of-file hook agree on the format); the wire text is the file content
    with exactly that newline removed.
    """
    text = (TOOL_DESCRIPTION_DIR / f"{key}.md").read_text(encoding="utf-8")
    return text.removesuffix("\n")


@dataclass(frozen=True)
class BuiltinToolSpec:
    """The declared wire artifact of one built-in tool, minus its description."""

    input_schema: Optional[dict[str, Any]]
    input_kind: str = "json"
    freeform_format: Optional[dict[str, str]] = None


def builtin_tool_definition(spec_key: str, *, name: Optional[str] = None) -> ToolDefinition:
    """Build the declared ToolDefinition for a built-in tool.

    ``name`` is the wire name when it differs from the spec key (the edit
    protocols expose several specs under the same ``edit``/``write`` names).
    """
    spec = BUILTIN_TOOL_SPECS[spec_key]
    return ToolDefinition(
        name=name or spec_key,
        description=tool_description_asset(spec_key),
        parameters=[],
        input_schema=spec.input_schema,
        input_kind=spec.input_kind,  # type: ignore[arg-type]
        freeform_format=spec.freeform_format,
    )


# Atomic ordinary-dispatch routing: all nested fields are required together
# and effort is nullable.
ORDINARY_MODEL_OVERRIDE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "provider": {
            "type": "string",
            "minLength": 1,
            "description": (
                "Non-empty configured provider name returned by list_subagent_models. "
                "Never infer it from the model name. "
                "`openai` and `openai_chatgpt` serve the same models; prefer `openai_chatgpt` when configured."
            ),
        },
        "model": {
            "type": "string",
            "minLength": 1,
            "description": "Non-empty exact model ID returned for that provider by list_subagent_models.",
        },
        "thinking_effort": {
            "anyOf": [{"type": "string", "minLength": 1}, {"type": "null"}],
            "description": (
                "Exact supported effort string, or null only when the selected model has no effort control."
            ),
        },
    },
    "required": ["provider", "model", "thinking_effort"],
}


def dispatch_agent_input_schema(
    agent_types: Sequence[str],
    *,
    browser_targets: tuple[str, ...] = (),
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "agent_type": {
            "type": "string",
            "enum": list(agent_types),
            "description": "Which agent to dispatch. The tool description lists each value's role and tools.",
        },
        "task": {
            "type": "string",
            "description": "Detailed, self-contained task for the sub-agent.",
        },
        "model_override": {
            **ORDINARY_MODEL_OVERRIDE_SCHEMA,
            "description": (
                'Usually omit this property entirely: a normal call is {"agent_type": "...", "task": "..."}. '
                "Only include it after calling list_subagent_models and selecting one exact route. "
                "Never send an empty object, blank strings, placeholder values, or a guessed provider/model. "
                "When present, all three nested fields are required."
            ),
        },
    }
    if "browser" in agent_types and len(browser_targets) > 1:
        properties["browser_target"] = {
            "type": "string",
            "enum": list(browser_targets),
            "description": (
                'Only for agent_type "browser". Omit for Playwright; choose Chrome only when the user directs you '
                "to use their configured Chrome browser."
            ),
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": ["agent_type", "task"],
    }


# Input schema for the generic ``lsp`` tool: one ``operation`` enum routing to
# per-operation argument requirements.
LSP_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": [
                "diagnostics",
                "definition",
                "type_definition",
                "implementation",
                "references",
                "hover",
                "call_hierarchy",
                "code_actions",
                "document_symbols",
                "workspace_symbols",
                "status",
                "capabilities",
                "reload",
            ],
            "description": (
                "The LSP operation to perform. Position operations (definition, "
                "type_definition, implementation, references, hover, call_hierarchy, "
                "code_actions) require path, "
                "line, and symbol. diagnostics and document_symbols require path. "
                "workspace_symbols requires query. status, capabilities, and reload "
                "need no additional args."
            ),
        },
        "path": {
            "type": "string",
            "description": (
                "File path: project-relative, using ../ traversal, or absolute. Required for most operations."
            ),
        },
        "line": {
            "type": "integer",
            "description": "1-based line number for position operations.",
        },
        "symbol": {
            "type": "string",
            "description": "Symbol name to resolve on the line. Supports 'name#N' for the Nth occurrence.",
        },
        "query": {
            "type": "string",
            "description": "Search query for workspace_symbols.",
        },
        "end_line": {
            "type": "integer",
            "description": "Optional 1-based end line for code_actions.",
        },
        "kind": {
            "type": "string",
            "description": "Optional code action kind filter, such as quickfix or refactor.",
        },
        "timeout": {
            "type": "number",
            "description": "Per-call timeout in seconds (default: 30).",
        },
    },
    "required": ["operation"],
}

LSP_EDIT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "operation": {
            "type": "string",
            "enum": ["rename", "rename_file", "format_document", "format_range", "apply_code_action"],
            "description": (
                "Mutating LSP operation. rename requires path, line, symbol, and new_name. "
                "rename_file requires path and new_path. format_document requires path. "
                "format_range requires path and line, with optional end_line. "
                "apply_code_action requires path, line, symbol, and action_id or query."
            ),
        },
        "path": {
            "type": "string",
            "description": "File path: project-relative, using ../ traversal, or absolute.",
        },
        "line": {
            "type": "integer",
            "description": "1-based line number for position/range operations.",
        },
        "symbol": {
            "type": "string",
            "description": "Symbol name to resolve on the line. Supports 'name#N' for the Nth occurrence.",
        },
        "new_name": {
            "type": "string",
            "description": "New symbol name for rename.",
        },
        "new_path": {
            "type": "string",
            "description": ("Destination path for rename_file: project-relative, using ../ traversal, or absolute."),
        },
        "query": {
            "type": "string",
            "description": "Title substring or numeric index for apply_code_action when action_id is not provided.",
        },
        "action_id": {
            "type": "string",
            "description": "Stable action_id listed by lsp code_actions.",
        },
        "end_line": {
            "type": "integer",
            "description": "Optional 1-based end line for format_range and apply_code_action.",
        },
        "kind": {
            "type": "string",
            "description": "Optional code action kind filter, such as quickfix or refactor.",
        },
        "apply": {
            "type": "boolean",
            "description": "Apply the edit when true; preview only when false. Defaults to true.",
        },
        "timeout": {
            "type": "number",
            "description": "Per-call timeout in seconds (default: 30).",
        },
    },
    "required": ["operation"],
}

SNAPSHOT_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["list", "show", "create", "restore"],
            "description": "Snapshot operation. Use restore with snapshot_id='latest' to undo the latest snapshot.",
        },
        "snapshot_id": {
            "type": "string",
            "description": "Snapshot id for show/restore. Use 'latest' to restore the newest snapshot.",
        },
        "paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Project-relative paths for action=create.",
        },
        "force": {
            "type": "boolean",
            "description": "Restore even when tracked files changed after the snapshot.",
        },
        "limit": {
            "type": "integer",
            "description": "Maximum number of snapshots to list.",
        },
    },
}

RESOLVE_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action_id": {
            "type": "string",
            "description": "Pending action id returned by a preview-only tool.",
        },
        "decision": {
            "type": "string",
            "enum": ["apply", "discard"],
            "description": "Apply or discard the pending preview action.",
        },
        "force": {
            "type": "boolean",
            "description": "Apply even if source hashes no longer match.",
        },
    },
    "required": ["action_id", "decision"],
}

HASHLINE_REPLACE_CONTENT_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "string"},
        {"type": "array", "items": {"type": "string"}},
        {"type": "null"},
    ],
    "description": (
        "Replacement file text as one string, an array of complete lines, or null to delete the line(s). "
        "Never include a display-only LINE#ID: prefix in this content."
    ),
}
HASHLINE_INSERT_CONTENT_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {"type": "string", "minLength": 1},
        {"type": "array", "items": {"type": "string"}, "minItems": 1},
    ],
    "description": (
        "Non-empty inserted file text as one string or an array of complete lines. "
        "Never include a display-only LINE#ID: prefix in this content."
    ),
}


def hashline_operation_schema(
    op: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"op": {"type": "string", "enum": [op]}, **properties},
        "required": ["op", *required],
        "additionalProperties": False,
    }


HASHLINE_V2_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to edit: project-relative, using ../ traversal, or absolute.",
        },
        "edits": {
            "type": "array",
            "description": (
                "All operations for this file, validated against one pre-edit snapshot. In displayed "
                "LINE#ID:CONTENT rows, pass LINE#ID to anchor fields and only CONTENT to content fields."
            ),
            "items": {
                "anyOf": [
                    hashline_operation_schema(
                        "set",
                        {
                            "tag": {"type": "string", "description": "Target LINE#ID."},
                            "content": HASHLINE_REPLACE_CONTENT_SCHEMA,
                        },
                        ["tag", "content"],
                    ),
                    hashline_operation_schema(
                        "replace",
                        {
                            "first": {"type": "string", "description": "First LINE#ID, inclusive."},
                            "last": {"type": "string", "description": "Last LINE#ID, inclusive."},
                            "content": HASHLINE_REPLACE_CONTENT_SCHEMA,
                        },
                        ["first", "last", "content"],
                    ),
                    hashline_operation_schema(
                        "append",
                        {
                            "after": {"type": "string", "description": "Optional LINE#ID to insert after."},
                            "content": HASHLINE_INSERT_CONTENT_SCHEMA,
                        },
                        ["content"],
                    ),
                    hashline_operation_schema(
                        "prepend",
                        {
                            "before": {"type": "string", "description": "Optional LINE#ID to insert before."},
                            "content": HASHLINE_INSERT_CONTENT_SCHEMA,
                        },
                        ["content"],
                    ),
                    hashline_operation_schema(
                        "insert",
                        {
                            "after": {"type": "string", "description": "Optional preceding LINE#ID."},
                            "before": {"type": "string", "description": "Optional following LINE#ID."},
                            "content": HASHLINE_INSERT_CONTENT_SCHEMA,
                        },
                        ["content"],
                    ),
                ]
            },
        },
        "delete": {"type": "boolean", "description": "Delete path; requires edits=[] and no rename."},
        "rename": {
            "type": "string",
            "description": ("Move the edited result to this path: project-relative, using ../ traversal, or absolute."),
        },
    },
    "required": ["path", "edits"],
    "additionalProperties": False,
}


BUILTIN_TOOL_SPECS: dict[str, BuiltinToolSpec] = {
    "apply_patch": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {"input": {"type": "string", "description": "The raw Codex apply_patch payload."}},
            "required": ["input"],
        },
        input_kind="freeform",
        freeform_format={"type": "grammar", "syntax": "lark", "definition": CODEX_APPLY_PATCH_GRAMMAR},
    ),
    "browser_click": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_click"]),
    "browser_close": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_close"]),
    "browser_console_messages": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_console_messages"]),
    "browser_drag": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_drag"]),
    "browser_drop": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_drop"]),
    "browser_evaluate": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_evaluate"]),
    "browser_file_upload": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_file_upload"]),
    "browser_fill_form": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_fill_form"]),
    "browser_find": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_find"]),
    "browser_handle_dialog": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_handle_dialog"]),
    "browser_hover": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_hover"]),
    "browser_navigate": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_navigate"]),
    "browser_navigate_back": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_navigate_back"]),
    "browser_network_request": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_network_request"]),
    "browser_network_requests": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_network_requests"]),
    "browser_press_key": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_press_key"]),
    "browser_resize": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_resize"]),
    "browser_scroll": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_scroll"]),
    "browser_select_option": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_select_option"]),
    "browser_snapshot": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_snapshot"]),
    "browser_tabs": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_tabs"]),
    "browser_take_screenshot": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_take_screenshot"]),
    "browser_type": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_type"]),
    "browser_wait_for": BuiltinToolSpec(input_schema=BROWSER_TOOL_SCHEMAS["browser_wait_for"]),
    "build_backend": BuiltinToolSpec(input_schema={"type": "object", "properties": {}, "required": []}),
    "build_frontend": BuiltinToolSpec(input_schema={"type": "object", "properties": {}, "required": []}),
    "claude_edit": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to modify: project-relative, using ../ traversal, or absolute.",
                },
                "old_string": {"type": "string", "description": "Exact text to replace."},
                "new_string": {"type": "string", "description": "Replacement text, which must differ from old_string."},
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace every exact occurrence instead of requiring a unique match.",
                },
            },
            "required": ["file_path", "old_string", "new_string"],
        }
    ),
    "claude_write": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to create or overwrite: project-relative, using ../ traversal, or absolute.",
                },
                "content": {"type": "string", "description": "Complete content to write to the file."},
            },
            "required": ["file_path", "content"],
        }
    ),
    # dispatch_agent input schema and catalog suffix are composed per
    # collection from explicit data (see ToolCollection._builtin_tool_definition).
    "dispatch_agent": BuiltinToolSpec(input_schema=None),
    "edit": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to edit: project-relative, using ../ traversal, or absolute.",
                },
                "block": {
                    "type": "string",
                    "description": "A single search and replace block. The tool replaces one "
                    "uniquely matched occurrence. Matching is attempted in this "
                    "order: 1. Exact match. 2. Per-line stripped match for "
                    "indentation and trailing whitespace differences. 3. "
                    "Normalized line endings. 4. Normalized smart quotes. "
                    "CRITICAL REQUIREMENTS FOR USING THIS TOOL: 1. UNIQUENESS: "
                    "The old_string MUST uniquely identify the specific instance "
                    "you want to change. This means: - Include AT LEAST 3-5 lines "
                    "of context BEFORE the change point - Include AT LEAST 3-5 "
                    "lines of context AFTER the change point - Include all "
                    "whitespace, indentation, and surrounding code exactly as it "
                    "appears in the file 2. SINGLE INSTANCE: This tool can only "
                    "change ONE instance at a time. If you need to change "
                    "multiple instances: - Use multi_edit when all replacements "
                    "are in the same file. - Each block must uniquely identify "
                    "its specific instance using extensive context. 3. "
                    "VERIFICATION: Before using this tool: - Check how many "
                    "instances of the target text exist in the file - If multiple "
                    "instances exist, gather enough context to uniquely identify "
                    "each one",
                },
            },
            "required": ["path", "block"],
        }
    ),
    "eval": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "code to run in this eval call, verbatim. Top-level await is fine.",
                },
                "language": {
                    "type": "string",
                    "description": 'which kernel to run in; defaults to "py". Pass "js" '
                    "(needs bun or node >= 18 on PATH).",
                },
                "title": {
                    "type": "string",
                    "description": 'short label for this step in the transcript (e.g. "load csv", "chart by region").',
                },
                "timeout": {
                    "type": "number",
                    "description": "timeout for this cell in seconds; 0 disables it. Default 120, max 600.",
                },
                "reset": {
                    "type": "boolean",
                    "description": "wipe this language's kernel before running (fresh state); "
                    "the other language is untouched.",
                },
            },
            "required": ["code"],
        }
    ),
    "exec_command": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command line, executed via `bash -c`."},
                "workdir": {
                    "type": "string",
                    "description": "Working directory for the command. Defaults to project root.",
                },
                "yield_time_ms": {
                    "type": "integer",
                    "description": "How long to wait for output/exit before returning, "
                    "in milliseconds (clamped to 250–30000).",
                },
                "max_output_tokens": {
                    "type": "integer",
                    "description": "Requested output budget for this call. Values are hard-clamped "
                    "to the global 10,000-token terminal-result ceiling.",
                },
                "login": {
                    "type": "boolean",
                    "description": "Run the shell as a login shell (sources profile). Default false.",
                },
                "background": {
                    "type": "boolean",
                    "description": "Launch detached and return after a short startup window "
                    "(~2s) with a session_id. The process outlives this call "
                    "and the agent session until kill_command stops it; it "
                    "accepts write_stdin input (no echo; stdin never reaches "
                    "EOF). Commands that exit within the startup window "
                    "report their real exit code.",
                },
            },
            "required": ["command"],
        }
    ),
    "get_host": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {"port": {"type": "integer", "description": ""}},
            "required": ["port"],
        }
    ),
    "hashline_edit": BuiltinToolSpec(input_schema=HASHLINE_V2_INPUT_SCHEMA),
    "hashline_write": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to create or replace: project-relative, using ../ traversal, or absolute.",
                },
                "content": {"type": "string", "description": "Complete file content."},
            },
            "required": ["path", "content"],
        }
    ),
    "kill_command": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "The id of the session to stop."},
                "signal": {"type": "string", "description": '"TERM" (default, graceful) or "INT" (Ctrl-C).'},
            },
            "required": ["session_id"],
        }
    ),
    "list_sessions": BuiltinToolSpec(input_schema={"type": "object", "properties": {}, "required": []}),
    "list_subagent_models": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "provider": {
                    "type": "string",
                    "description": "Optional provider name to filter the catalog. Omit it to "
                    "list every configured provider; a blank value is also "
                    "treated as an unfiltered request.",
                }
            },
            "required": [],
        }
    ),
    "list_workflow_runs": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Maximum number of runs to report (default 20)."}
            },
            "required": [],
        }
    ),
    "lsp": BuiltinToolSpec(input_schema=LSP_INPUT_SCHEMA),
    "lsp_edit": BuiltinToolSpec(input_schema=LSP_EDIT_INPUT_SCHEMA),
    "multi_edit": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to edit: project-relative, using ../ traversal, or absolute.",
                },
                "blocks": {
                    "type": "string",
                    "description": "One or more search and replace blocks formatted as shown above",
                },
            },
            "required": ["path", "blocks"],
        }
    ),
    "read": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file. Relative to the project root is "
                    "preferred; an absolute path is also accepted.",
                },
                "offset": {"type": "integer", "description": "The 1-indexed first line to read (default 1)."},
                "limit": {
                    "type": "integer",
                    "description": "The maximum number of lines to read; omitted reads from the top.",
                },
            },
            "required": ["file_path"],
        }
    ),
    "read_image": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the project root, or an allowed absolute path.",
                }
            },
            "required": ["path"],
        }
    ),
    "resolve": BuiltinToolSpec(input_schema=RESOLVE_INPUT_SCHEMA),
    "run_workflow": BuiltinToolSpec(input_schema=RUN_WORKFLOW_INPUT_SCHEMA),
    "snapshot": BuiltinToolSpec(input_schema=SNAPSHOT_INPUT_SCHEMA),
    "web_fetch": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Full http(s) URL to fetch."},
                "instruction": {"type": "string", "description": "Guidance for how to use the extracted content."},
            },
            "required": ["url", "instruction"],
        }
    ),
    "web_search": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query (natural language or keywords)."},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (clamped to 1-10, default 5).",
                },
            },
            "required": ["query"],
        }
    ),
    "write": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to write: project-relative, using ../ traversal, or absolute.",
                },
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        }
    ),
    "write_stdin": BuiltinToolSpec(
        input_schema={
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": 'The id returned by exec_command when status == "running".',
                },
                "chars": {"type": "string", "description": "Bytes to write to stdin. An empty string polls only."},
                "yield_time_ms": {
                    "type": "integer",
                    "description": "How long to wait for more output or the process to "
                    "exit, in milliseconds (clamped to 250–30000 when "
                    "writing, 5000–300000 when polling).",
                },
                "max_output_tokens": {
                    "type": "integer",
                    "description": "Requested output budget for this call. Values are hard-clamped "
                    "to the global 10,000-token terminal-result ceiling.",
                },
            },
            "required": ["session_id"],
        }
    ),
}
