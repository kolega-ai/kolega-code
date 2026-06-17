The CLI provides a shared Markdown task list through `get_task_list` and `update_task_list`.
Use it to coordinate planning and implementation.

In planning mode, create or update the task list before calling `write_plan`.
In build mode, call `get_task_list` when a shared task list exists or when implementing an approved plan.
After each meaningful task is completed, call `update_task_list` to check off that item by rewriting the full Markdown list.
Do not wait until every TODO is complete to update the shared task list.
