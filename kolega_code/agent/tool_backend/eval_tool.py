"""Model-facing `eval` tool backend: persistent code kernels with tool callback.

The heavy lifting lives in kolega_code.agent.eval (kernels, bridge, env); this
backend resolves the session's shared EvalKernelManager, runs cells, and formats
the outcome (stdout/stderr, REPL result, display outputs incl. images, status
lines, errors) for the model.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from kolega_code.agent.eval.kernel import EvalCellResult, EvalKernelManager
from kolega_code.llm.models import ContentBlock, ImageBlock, TextBlock
from kolega_code.tools import ToolError

from .base_tool import BaseTool

EVAL_LANGUAGES = ("py", "js")

DEFAULT_CELL_TIMEOUT_S = 120.0
MAX_CELL_TIMEOUT_S = 600.0

# Output budgets. stdout/stderr keep the tail (recent lines matter most);
# display chunks are individually capped like oh-my-pi's 8 KB display limit.
_STDOUT_TAIL_CHARS = 20_000
_STDERR_TAIL_CHARS = 8_000
_DISPLAY_CHUNK_CHARS = 8_000
_RESULT_CHARS = 8_000

_IMAGE_MIMES = ("image/png", "image/jpeg")


def clamp_cell_timeout(timeout: object) -> Optional[float]:
    """None -> default; 0 -> no timeout; otherwise clamp to (0, MAX].

    Accepts ``object`` because model-emitted JSON can carry non-numeric values;
    anything unparseable falls back to the default.
    """
    if timeout is None:
        return DEFAULT_CELL_TIMEOUT_S
    if not isinstance(timeout, (int, float, str)):
        return DEFAULT_CELL_TIMEOUT_S
    try:
        value = float(timeout)
    except (TypeError, ValueError):
        return DEFAULT_CELL_TIMEOUT_S
    if value == 0:
        return None
    return max(1.0, min(value, MAX_CELL_TIMEOUT_S))


def _tail(text: str, limit: int) -> Tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[-limit:], True


def _bundle_text(bundle: Dict[str, Any], limit: int) -> str:
    """Human-facing text for a MIME bundle (prefers plain text, then JSON)."""
    plain = bundle.get("text/plain")
    if isinstance(plain, str) and plain:
        text = plain
    else:
        data: Any = bundle.get("application/json", None)
        if data is not None:
            try:
                text = json.dumps(data, indent=2, ensure_ascii=False)
            except (TypeError, ValueError):
                text = str(data)
        elif isinstance(bundle.get("text/html"), str):
            text = f"[html output]\n{bundle['text/html']}"
        elif isinstance(bundle.get("image/svg+xml"), str):
            text = f"[svg image]\n{bundle['image/svg+xml']}"
        else:
            text = ""
    if len(text) > limit:
        text = text[:limit] + f"\n[…{len(text) - limit}ch elided…]"
    return text


def _bundle_images(bundle: Dict[str, Any]) -> List[Tuple[str, str]]:
    """(mime, base64) pairs for image payloads in a bundle."""
    images: List[Tuple[str, str]] = []
    for mime in _IMAGE_MIMES:
        data = bundle.get(mime)
        if isinstance(data, str) and data:
            images.append((mime, data))
    return images


class EvalTool(BaseTool):
    """Runs code cells on the session's persistent kernels."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._manager: Optional[EvalKernelManager] = None

    def _kernel_manager(self) -> EvalKernelManager:
        if self._manager is None:
            self._manager = EvalKernelManager.for_thread(
                workspace_id=self.workspace_id,
                thread_id=self.thread_id,
                project_path=self.project_path,
                config=self.config,
                scratchpad_dir=getattr(self.caller, "scratchpad_dir", None),
            )
        return self._manager

    async def shutdown_if_owner(self) -> None:
        """Shut down shared kernels. Only the top-level agent owns them."""
        if getattr(self.caller, "sub_agent", False):
            return
        manager = self._manager
        if manager is not None:
            await manager.shutdown()
            self._manager = None

    async def eval(
        self,
        language: str,
        code: str,
        title: Optional[str] = None,
        timeout: Optional[float] = None,
        reset: bool = False,
    ) -> Union[str, List[ContentBlock]]:
        """Run one cell. See ToolCollection.eval for the model-facing contract."""
        language = str(language).strip().lower()
        if language not in EVAL_LANGUAGES:
            raise ToolError(f"unsupported eval language {language!r}; use 'py' or 'js'")
        if not isinstance(code, str) or not code.strip():
            raise ToolError("eval requires non-empty code")

        manager = self._kernel_manager()
        result = await manager.execute(
            language=language,
            code=code,
            agent=self.caller,
            timeout=clamp_cell_timeout(timeout),
            reset=bool(reset),
        )
        return self._format(result, title=title)

    # -- result formatting ---------------------------------------------------

    def _format(self, result: EvalCellResult, *, title: Optional[str]) -> Union[str, List[ContentBlock]]:
        parts: List[str] = []
        if title:
            parts.append(f"## {title}")

        for note in result.notes:
            parts.append(f"> {note}")

        stdout, stdout_cut = _tail(result.stdout, _STDOUT_TAIL_CHARS)
        if stdout.strip():
            if stdout_cut:
                parts.append(f"[stdout truncated to the last {_STDOUT_TAIL_CHARS} chars]")
            parts.append(f"### stdout\n{stdout.rstrip()}")

        stderr, stderr_cut = _tail(result.stderr, _STDERR_TAIL_CHARS)
        if stderr.strip():
            if stderr_cut:
                parts.append(f"[stderr truncated to the last {_STDERR_TAIL_CHARS} chars]")
            parts.append(f"### stderr\n{stderr.rstrip()}")

        status_lines = self._status_lines(result)
        if status_lines:
            parts.append("### status\n" + "\n".join(status_lines))

        if result.result_bundle:
            text = _bundle_text(result.result_bundle, _RESULT_CHARS)
            if text:
                parts.append(f"### result\n{text}")

        display_parts: List[str] = []
        for index, bundle in enumerate(result.displays, start=1):
            text = _bundle_text(bundle, _DISPLAY_CHUNK_CHARS)
            if text:
                display_parts.append(f"display[{index}]:\n{text}")
            elif _bundle_images(bundle):
                display_parts.append(f"display[{index}]: [image]")
        if display_parts:
            parts.append("\n\n".join(display_parts))

        if result.error is not None:
            error = result.error
            traceback_text = "".join(error.traceback).rstrip()
            block = f"### error\n{error.ename}: {error.evalue}"
            if traceback_text:
                block += f"\n{traceback_text}"
            parts.append(block)
        elif result.status != "ok" and not result.interrupted:
            parts.append(f"### error\ncell finished with status: {result.status}")

        text_out = "\n\n".join(parts) if parts else "(cell completed with no output)"

        images: List[Tuple[str, str]] = []
        for bundle in result.displays:
            images.extend(_bundle_images(bundle))
        images.extend(_bundle_images(result.result_bundle))
        if not images:
            return text_out

        if not bool(getattr(self.caller, "supports_vision", False)):
            return text_out + f"\n\n[{len(images)} image output(s) not shown: model has no vision support]"

        blocks: List[ContentBlock] = [TextBlock(text=text_out)]
        blocks.extend(ImageBlock(image_type="base64", media_type=mime, data=data) for mime, data in images)
        return blocks

    @staticmethod
    def _status_lines(result: EvalCellResult) -> List[str]:
        lines: List[str] = []
        for event in result.statuses:
            op = event.get("op")
            if op == "log":
                lines.append(f"- log: {event.get('message', '')}")
            elif op == "phase":
                lines.append(f"- phase: {event.get('title', '')}")
            elif op in ("read", "write", "pip_install", "npm_install"):
                detail = event.get("path") or ", ".join(str(p) for p in event.get("packages", []) or [])
                lines.append(f"- {op}: {detail}")
        return lines
