"""Hook execution backends: command (subprocess), python (in-process), and LLM.

Each backend takes a ``LifecycleEvent``, a ``HookSpec`` and a ``HookCapabilities``
bundle and returns a ``HookOutcome``. Backends raise ``HookExecutionError`` for
operational failures (timeout, bad exit code, unparseable LLM reply); the
dispatcher logs those and treats them as non-blocking (fail-open).
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from .config import HookSpec
from .events import LifecycleEvent
from .outcome import HookOutcome

# (prompt_text, model_hint) -> model response text
PromptRunner = Callable[[str, Optional[str]], Awaitable[str]]
# (task_text) -> sub-agent final report text
AgentRunner = Callable[[str], Awaitable[str]]
# (message) -> None, for surfacing hook errors/diagnostics to the user/logs
LogFn = Callable[[str], Awaitable[None]]


class HookExecutionError(RuntimeError):
    """A hook failed to run (non-blocking; the action proceeds)."""


@dataclass
class HookCapabilities:
    """Host-provided capabilities a hook backend may need.

    ``command``/``python`` hooks need none of these. ``prompt`` needs
    ``prompt_runner``; ``agent`` needs ``agent_runner``. When a required
    capability is absent the LLM hook fails open with a logged error.
    """

    project_path: Optional[Path] = None
    prompt_runner: Optional[PromptRunner] = None
    agent_runner: Optional[AgentRunner] = None
    log: Optional[LogFn] = None


async def run_hook(event: LifecycleEvent, spec: HookSpec, caps: HookCapabilities) -> HookOutcome:
    """Dispatch to the backend for ``spec.type`` and return its outcome."""
    if spec.type == "command":
        return await _run_command(event, spec, caps)
    if spec.type == "python":
        return await _run_python(event, spec, caps)
    return await _run_llm(event, spec, caps)


# --------------------------------------------------------------------------- #
# command
# --------------------------------------------------------------------------- #


async def _run_command(event: LifecycleEvent, spec: HookSpec, caps: HookCapabilities) -> HookOutcome:
    if not spec.command:
        raise HookExecutionError("command hook has no command")

    payload = json.dumps(event.to_hook_input()).encode("utf-8")
    proc = await asyncio.create_subprocess_exec(
        *shlex.split(spec.command),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(caps.project_path) if caps.project_path else None,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(payload), timeout=spec.timeout)
    except asyncio.TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise HookExecutionError(f"command hook timed out after {spec.timeout}s") from exc

    code = proc.returncode
    if code == 0:
        return _parse_command_stdout(stdout.decode("utf-8", "replace"))
    if code == 2:
        reason = stderr.decode("utf-8", "replace").strip() or "Hook blocked the action."
        return HookOutcome.deny(reason)
    detail = stderr.decode("utf-8", "replace").strip()
    raise HookExecutionError(f"command hook exited {code}: {detail}")


def _parse_command_stdout(text: str) -> HookOutcome:
    text = text.strip()
    if not text:
        return HookOutcome.empty()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return HookOutcome.empty()
    if not isinstance(data, dict):
        return HookOutcome.empty()

    hso = data.get("hookSpecificOutput") or {}
    if not isinstance(hso, dict):
        hso = {}
    decision = str(hso.get("permissionDecision") or "").lower()
    updated_input = hso.get("updatedInput")
    blocked = decision == "deny"
    return HookOutcome(
        blocked=blocked,
        reason=str(hso.get("permissionDecisionReason") or data.get("systemMessage") or "") if blocked else "",
        updated_input=updated_input if isinstance(updated_input, dict) else None,
        updated_output=_as_text(hso.get("updatedToolOutput")),
        additional_context=_as_text(hso.get("additionalContext")),
        end_turn=data.get("continue") is False or blocked,
    )


# --------------------------------------------------------------------------- #
# python
# --------------------------------------------------------------------------- #


async def _run_python(event: LifecycleEvent, spec: HookSpec, caps: HookCapabilities) -> HookOutcome:
    target = spec.callable or ""
    module_name, sep, attr = target.partition(":")
    if not module_name or not sep or not attr:
        raise HookExecutionError(f"invalid python callable {target!r} (expected 'module.path:function')")

    try:
        module = importlib.import_module(module_name)
        func = getattr(module, attr)
    except (ImportError, AttributeError) as exc:
        raise HookExecutionError(f"could not import python hook {target!r}: {exc}") from exc

    result = func(event)
    if inspect.isawaitable(result):
        result = await asyncio.wait_for(result, timeout=spec.timeout)
    return _coerce_outcome(result)


def _coerce_outcome(result: Any) -> HookOutcome:
    if isinstance(result, HookOutcome):
        return result
    if result is None:
        return HookOutcome.empty()
    if isinstance(result, dict):
        return HookOutcome(
            blocked=bool(result.get("blocked")),
            reason=str(result.get("reason") or ""),
            updated_input=result.get("updated_input") if isinstance(result.get("updated_input"), dict) else None,
            updated_output=_as_text(result.get("updated_output")),
            additional_context=_as_text(result.get("additional_context")),
            end_turn=bool(result.get("end_turn")) or bool(result.get("blocked")),
        )
    raise HookExecutionError(f"python hook returned unsupported type {type(result).__name__}")


# --------------------------------------------------------------------------- #
# LLM (prompt + agent), using the {"ok": bool, "reason": str} protocol
# --------------------------------------------------------------------------- #


async def _run_llm(event: LifecycleEvent, spec: HookSpec, caps: HookCapabilities) -> HookOutcome:
    rendered = _render_prompt(spec.prompt or "", event)
    if spec.type == "prompt":
        if caps.prompt_runner is None:
            raise HookExecutionError("prompt hook unavailable: no LLM runner in this host")
        text = await asyncio.wait_for(caps.prompt_runner(rendered, spec.model), timeout=spec.timeout)
    else:  # agent
        if caps.agent_runner is None:
            raise HookExecutionError("agent hook unavailable: no agent runner in this host")
        text = await asyncio.wait_for(caps.agent_runner(rendered), timeout=spec.timeout)
    return _parse_ok_reason(text)


def _render_prompt(template: str, event: LifecycleEvent) -> str:
    payload = json.dumps(event.to_hook_input(), indent=2)
    if "$EVENT" in template or "$ARGUMENTS" in template:
        return template.replace("$EVENT", payload).replace("$ARGUMENTS", payload)
    return f"{template}\n\nEvent data:\n{payload}"


def _parse_ok_reason(text: str) -> HookOutcome:
    data = _extract_json_object(text)
    if not isinstance(data, dict) or "ok" not in data:
        raise HookExecutionError(f"LLM hook did not return an {{ok, reason}} object: {text[:200]!r}")
    if data.get("ok"):
        return HookOutcome.empty()
    return HookOutcome.deny(str(data.get("reason") or "Hook blocked the action."))


def _extract_json_object(text: str) -> Any:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Tolerate prose around the JSON: grab the first balanced {...} block.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def _as_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text or None
