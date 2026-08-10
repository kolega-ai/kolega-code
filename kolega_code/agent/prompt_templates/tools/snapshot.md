Manage file snapshots for undo, inspection, and manual checkpoints.

Use action="list" to see recent snapshots, action="show" with a snapshot_id
to inspect one, action="create" with paths to make a manual checkpoint, and
action="restore" to restore a snapshot's before-state. Use snapshot_id="latest"
with restore as an undo for the newest snapshot.

Returns:
    Markdown summary of the snapshot operation.
