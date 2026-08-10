Run a gigacode workflow: an authored Python script that orchestrates many
sub-agents with deterministic control flow (parallel fan-out, pipelines,
loop-until-dry, budget loops).

The script's primitives are `agent()`, `parallel()`, `pipeline()`, `phase()`,
and `log()`, plus the `args` and `budget` globals. See the gigacode authoring
guide in your system prompt for the full API and patterns. Artifacts (script,
full result and readable transcript; raw/per-agent debug artifacts are saved
under the run directory but are not advertised by default) are written under the CLI state directory, and a run can be resumed with
`resume_from_run_id`.

Returns:
    A compact artifact manifest: the runId, persisted scriptPath, token count,
    resultPath, and transcriptPath. The workflow result is written to
    resultPath rather than returned inline. Read resultPath for the workflow
    result, or transcriptPath for execution details. For normal workflow
    output, avoid reading individual sub-agent transcripts.
