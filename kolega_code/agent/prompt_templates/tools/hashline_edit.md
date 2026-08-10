Apply precise edits using Hashline v2 ``LINE#ID`` anchors.

Read the target range immediately before editing and copy its anchors
exactly. Every operation is validated against the same pre-edit file
snapshot and applied bottom-up. Re-read the file before a later edit
call because successful edits change its anchors.

Read results render each source line as ``LINE#ID:CONTENT``. The
``LINE#ID:`` prefix is display-only metadata, not part of the file. For
example, given ``1#BM:MAX_RETRIES = 3``, use ``1#BM`` as the anchor and
``MAX_RETRIES = 5`` as replacement content. Never copy ``1#BM:`` or any
other anchor prefix into ``content``.

Use ``set`` for one line, ``replace`` for an inclusive range,
``append``/``prepend`` for insertion after/before an optional anchor,
and ``insert`` for one- or two-sided anchored insertion. ``content`` may
be a string or an array of complete lines; null deletes set/replace
targets. Use ``delete=true`` with an empty edits array to delete a file,
or ``rename`` to move the edited result.

Returns:
    A short summary, or fresh tagged context when an anchor is stale.
