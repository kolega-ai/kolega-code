from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from kolega_code.services.lsp.edits import WorkspaceEditApplier
from kolega_code.services.snapshots import PendingAction, SnapshotError, SnapshotRecord, SnapshotService
from kolega_code.tools import ToolError

from .base_tool import BaseTool
from .edit_preview import build_diff_preview

if TYPE_CHECKING:
    from kolega_code.services.lsp.edits import WorkspaceEditResult


class SnapshotTool(BaseTool):
    def __init__(self, *args, snapshot_service: SnapshotService, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.snapshot_service = snapshot_service

    async def snapshot(
        self,
        action: str = "list",
        snapshot_id: str = "",
        paths: Optional[list[str]] = None,
        force: bool = False,
        limit: int = 20,
    ) -> str:
        """Manage file snapshots for undo, inspection, and manual checkpoints.

        Actions:
        - list: show recent snapshots for this session.
        - show: show one snapshot's metadata and touched paths.
        - create: create a manual checkpoint for the provided paths.
        - restore: restore a snapshot's before-state. Use snapshot_id="latest" for undo.

        Args:
            action: One of list, show, create, or restore.
            snapshot_id: Snapshot id to show or restore. Use latest for the newest snapshot.
            paths: Project-relative paths for create.
            force: Restore even when tracked files changed after the snapshot.
            limit: Maximum number of snapshots to list.

        Returns:
            Markdown summary of the requested snapshot operation.
        """
        try:
            normalized_action = action.strip().lower()
            if normalized_action == "list":
                return self._format_snapshot_list(self.snapshot_service.list_snapshots(limit=limit))
            if normalized_action == "show":
                if not snapshot_id:
                    raise SnapshotError("snapshot_id is required for action=show.")
                return self._format_snapshot_detail(self.snapshot_service.load_snapshot(snapshot_id))
            if normalized_action == "create":
                record = self.snapshot_service.create_manual_snapshot(
                    paths=paths or (),
                    reason="manual snapshot",
                    tool_call_id=getattr(self.caller, "current_tool_execution_id", None),
                )
                return f"Created manual snapshot `{record.snapshot_id}` for {', '.join(record.touched_paths)}."
            if normalized_action == "restore":
                target_id = snapshot_id or "latest"
                result = self.snapshot_service.restore_snapshot(target_id, force=force)
                forced = " with force=true" if result.forced else ""
                return (
                    f"Restored snapshot `{result.snapshot_id}`{forced}.\n\n"
                    f"Touched paths: {', '.join(result.restored_paths) or '(none)'}"
                )
            raise SnapshotError("action must be one of: list, show, create, restore.")
        except SnapshotError as exc:
            raise ToolError(str(exc)) from exc

    async def resolve(self, action_id: str, decision: str, force: bool = False) -> str:
        """Apply or discard a pending preview action.

        Pending actions are created by preview-only tools such as lsp_edit(apply=false).
        Applying a pending action checks that the source files still match the preview
        inputs before writing, unless force=true is explicitly provided.

        Args:
            action_id: Pending action id returned by a preview-only tool.
            decision: apply or discard.
            force: Apply even if source hashes no longer match.

        Returns:
            Markdown summary of the resolve decision.
        """
        try:
            action = self.snapshot_service.load_pending_action(action_id)
            normalized_decision = decision.strip().lower()
            if normalized_decision == "discard":
                updated = self.snapshot_service.update_pending_action(action, status="discarded", message="discarded")
                return f"Discarded pending action `{updated.action_id}`."
            if normalized_decision != "apply":
                raise SnapshotError("decision must be apply or discard.")
            if action.status != "pending":
                raise SnapshotError(f"Pending action `{action.action_id}` is already {action.status}.")

            mismatches = self.snapshot_service.diff_expected_current(action.source)
            if mismatches and not force:
                message = self.snapshot_service._format_mismatch_error(action.action_id, mismatches)
                self.snapshot_service.update_pending_action(action, status="stale", message=message)
                raise SnapshotError(message)

            applier = WorkspaceEditApplier(self.project_path, self.filesystem)
            preview = applier.preview(action.workspace_edit)

            def _apply() -> "WorkspaceEditResult":
                return applier.apply(action.workspace_edit)

            mutation = self.snapshot_service.record_mutation(
                tool_name=action.tool_name or "resolve",
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None) or action.tool_call_id,
                reason=f"resolve {action.action_id} ({action.operation})",
                paths=preview.touched_paths,
                mutate=_apply,
            )
            result = mutation.result
            await self._send_previews(result, action)
            updated = self.snapshot_service.update_pending_action(
                action,
                status="applied",
                snapshot_id=mutation.snapshot.snapshot_id if mutation.snapshot else None,
                message="applied",
            )
            snapshot_line = f"\nSnapshot: `{updated.snapshot_id}`" if updated.snapshot_id else ""
            forced = " with force=true" if force else ""
            return (
                f"Applied pending action `{updated.action_id}`{forced}.\n\n"
                + self._format_workspace_edit_result(action.operation, result)
                + snapshot_line
            )
        except SnapshotError as exc:
            raise ToolError(str(exc)) from exc

    async def _send_previews(self, result: "WorkspaceEditResult", action: PendingAction) -> None:
        for change in result.text_changes:
            if change.old_text == change.new_text:
                continue
            await self.send_edit_preview(
                build_diff_preview(change.old_text, change.new_text, change.path),
                tool_call_id=getattr(self.caller, "current_tool_execution_id", None) or action.tool_call_id,
                tool_name="resolve",
            )

    @staticmethod
    def _format_snapshot_list(records: list[SnapshotRecord]) -> str:
        if not records:
            return "No snapshots recorded for this session."
        lines = ["# Snapshots", ""]
        for record in records:
            marker = " manual" if record.manual else ""
            lines.append(
                f"- `{record.snapshot_id}`{marker} - {record.tool_name} - "
                f"{record.created_at} - {', '.join(record.touched_paths) or '(none)'}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_snapshot_detail(record: SnapshotRecord) -> str:
        lines = [
            f"# Snapshot `{record.snapshot_id}`",
            "",
            f"- Tool: `{record.tool_name}`",
            f"- Tool call: `{record.tool_call_id or '-'}`",
            f"- Created: {record.created_at}",
            f"- Reason: {record.reason or '-'}",
            f"- Manual: {str(record.manual).lower()}",
            "",
            "## Paths",
        ]
        for path in record.touched_paths:
            before = record.before.get(path)
            after = record.after.get(path)
            lines.append(f"- `{path}`: {SnapshotTool._state_label(before)} -> {SnapshotTool._state_label(after)}")
        return "\n".join(lines)

    @staticmethod
    def _format_workspace_edit_result(operation: str, result: Any) -> str:
        lines = [f"Applied LSP edit `{operation}`."]
        for summary in result.summaries:
            lines.append(f"- {summary}")
        return "\n".join(lines)

    @staticmethod
    def _state_label(state: Any) -> str:
        if state is None:
            return "missing"
        suffix = f":{state.sha256[:12]}" if getattr(state, "sha256", None) else ""
        return f"{state.kind}{suffix}"
