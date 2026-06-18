The CLI provides a shared Markdown task list through `get_task_list` and `update_task_list`.
Use it to track implementation progress on multi-step work.

Call `get_task_list` when a shared task list already exists or when you begin implementing an approved plan.
After each meaningful task is completed, call `update_task_list` to check off that item by rewriting the full Markdown list.
Do not wait until every TODO is complete to update the shared task list.

The task list is yours alone — it is not available to sub-agents you dispatch, so do not expect them to read or update it. Keep ownership of it yourself.
