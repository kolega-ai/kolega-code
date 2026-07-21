# Publishing a pull-request review

Use this procedure only after a GitHub PR review requested with `--comment` has completed
candidate challenge and final deduplication. The flag authorizes review comments, not an
approval, merge, or request-changes action.

## Resolve publication coordinates

Obtain JSON from `gh`, not human-formatted terminal output. Resolve:

- the base repository's `owner/name`;
- the pull-request number and URL;
- the full head commit ID.

Keep those values fixed for the publication attempt. If the PR head changed after analysis,
stop and report that the review is stale rather than attaching findings to a different
revision.

## Validate locations and duplicates

For each finding:

1. Confirm that its path and line are present on the right side of the reviewed patch.
2. Query existing inline review comments with:

   ```bash
   gh api --paginate "repos/{owner}/{repo}/pulls/{number}/comments"
   ```

3. Drop a comment when an existing one identifies the same root problem at the same path and
   line. Compare meaning, not wording alone.

For a review with no findings, inspect existing top-level conversation comments before
adding another equivalent clean-review summary.

## Compose comments

Make each inline comment understandable on its own:

- Name the defect.
- Explain the concrete consequence.
- Point to the evidence that establishes the failure.
- Give a bounded repair direction.

Use a fenced GitHub suggestion only when a short replacement at that one location fully
repairs the defect. Do not offer a suggestion when another file, migration, generated
artifact, or follow-up step is also required.

When a comment needs a source link, build it from immutable coordinates:

```text
https://github.com/{owner}/{repo}/blob/{full-head-id}/{path}#L{start}-L{end}
```

Link project guidance the same way. Use the complete commit ID.

## Submit once

When findings remain, create one review through the pull-request reviews API:

```bash
gh api --method POST "repos/{owner}/{repo}/pulls/{number}/reviews" --input -
```

Supply one JSON object on standard input:

```json
{
  "commit_id": "<full-head-id>",
  "body": "<brief review summary>",
  "event": "COMMENT",
  "comments": [
    {
      "path": "src/example.py",
      "line": 42,
      "side": "RIGHT",
      "body": "<validated finding>"
    }
  ]
}
```

Generate JSON with a real serializer rather than shell string interpolation. Keep the payload
outside the project tree if a temporary file is unavoidable.

When no findings remain, add one short top-level summary with `gh pr comment`. Do not create
an empty inline review.

## Handle the response

Treat a non-success or indeterminate response as an incomplete publication. Immediately
query the PR again, reconcile the intended comments with the reviews and comments GitHub
accepted, and report their identifiers. Decide whether a retry is safe only after this
reconciliation. Never repeat a write merely because its response was unclear.

After success, report the review URL or returned identifier and the number of inline
comments created.
