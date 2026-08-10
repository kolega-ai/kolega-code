Return the shared CLI task list, which the build agent owns.

Use this when re-planning work that is already underway, to see what an in-progress build
session has already completed. You cannot modify the list in plan mode; capture the remaining
work in the plan you submit with `write_plan` instead.

Returns:
    The current shared task list, or a note that no task list has been set.
