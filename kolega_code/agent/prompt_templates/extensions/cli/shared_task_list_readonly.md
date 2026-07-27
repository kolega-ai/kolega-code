The CLI keeps a shared Markdown task list that the build agent owns.

Call `get_task_list` to see what an in-progress build session has already completed. This is worth doing when you are
re-planning work that is already underway, so the plan you produce accounts for what is done and what remains.

You cannot change the list here, because plan mode has no tool for writing it. Record the remaining work in the plan you
submit with `write_plan` instead, and the build agent will bring the shared list up to date when it starts implementing.

The list is not available to sub-agents you dispatch, so do not expect them to read it.
