List this session's gigacode workflow runs, newest first, with each run's
runId, name, status, timing, tokens, journaled agent calls, and artifact
paths. Only runs started by the current session are shown.

Use this to recover a run whose id you no longer have — for example after a
`run_workflow` call was interrupted ("Operation was interrupted"). Do not
re-run an interrupted workflow from scratch: resume it with
`run_workflow(resume_from_run_id=...)` so its journaled calls replay instead
of re-running. A run listed as "running" when no workflow is actually in
flight died mid-run and is resumable the same way.
