# Examples

The workflow in this directory is unedited model output. The header of
`parallel-code-review.py` documents its provenance: the complete two-sentence
prompt, the models involved, the execution record, and a sha256 of the model's
bytes so you can verify nothing below the delimiter was touched.

## parallel-code-review.py

The script was authored by `gpt-5.6-sol` from a two-sentence prompt. Every
sub-agent is pinned to `deepseek-v4-flash` by the script's own `model_override`.
This is a two-model workflow: one model wrote the orchestration, and a cheaper
one staffed it. The workflow uses six parallel discovery specialists (API contracts,
auth/security, Zapier actions/dataflow, SDK/runtime, tests/release, plus an
unconstrained bug hunt), one adversarial challenger per candidate finding
whose job is to *disprove* it, and a final synthesis gate that dedupes and
re-checks the survivors. Every worker is a read-only investigation agent, so
the workflow cannot modify the repo it reviews.

The execution record in the header is of these exact bytes: 18 agent calls,
all completed, 267k output tokens, ~11 minutes of workflow wall-clock against
a small Zapier integration repo.

## Rough edges

The script is unedited and has several quirks:

- **Schema-enforced confidence.** The findings schema only allows
  `"high"` and `"very high"` confidence, so every finding claims high
  confidence by construction.
- **One challenger per finding.** Each candidate lives or dies on a single
  adversarial verdict rather than a panel. If a challenger agent fails
  outright, its candidate is silently dropped instead of retried. That errs
  toward fewer, better-supported findings, but relies on one judge.
- **Unused `args`.** The model invoked the workflow with
  `args={repository, head, boundary}` and then never read `args` in the
  script. The boundary is hardcoded inline.
- **Barriers where a pipeline would stream.** Challengers wait for the
  slowest discovery specialist before any of them start; per-item
  pipelining would overlap the stages.
- **Forced precision.** Every finding must cite an integer line number, even
  for repo-wide issues, and deduplication is delegated to the synthesis
  prompt rather than done in code.

## Run the same prompt against your own repo

The script above is tailored to its target repo. Generate a new one for your
own repo instead of running this script:

```
cd your-repo
kolega-code
/gigacode on
write a gigacode workflow for parallel code review of this repo and execute it.
```

To use cheaper workers from a different provider, add a sub-agent model
instruction like the second sentence of the original prompt. The agent explores
your repo, writes a workflow, and runs it. The script, full result, readable
transcript, and resume journal are saved under kolega-code's state directory.
The result path is printed when the run completes.

Cost expectations from the observed run on a ~1,000 LOC repo are ~267k
output tokens total (sub-agent calls typically spend 5k to 30k each; count scales
with repo surface and findings), ~11 minutes wall-clock. Leave `token_budget`
unset unless you need a hard ceiling. A capped run stops cleanly and can be
resumed with its completed calls replaying at no cost.
