# ACP editor display smoke test

Verified locally on **November 1, 2026**, on macOS, against this worktree.
This is an actual editor interaction check, not an initialize-only protocol test.

## Verdict

| Editor | Two real prompt turns | Unknown usage metadata safe | Native context/cost display |
| --- | --- | --- | --- |
| Zed | Pass | Pass | **Pass** |
| VS Code + ACP Client | Pass | Pass | **Not supported by the tested extension UI** |

**Zed is the verified supported-editor display path.** VS Code's successful
handshake, transcript, and metadata delivery must not be described as successful
context/cost display.

## Exact versions

- Kolega package: `0.47.1`, running this worktree's source.
- Python SDK: `agent-client-protocol==0.12.0` (unchanged).
- Zed, as advertised by its actual ACP initialize request:
  `1.26.2+stable.384.c4273271092b0287f7d181f616bd5171e0a2f70c`.
- VS Code: `1.138.0`, commit `7debcd0e2acdea1c52de81bf9ee1620444407dda`.
- Installed VS Code extension: `formulahendry.acp-client` `0.2.0`;
  ACP initialize identifies `vscode-acp-client`, version `0.2.0`.
- Both clients negotiated ACP protocol version `1`.

Neither client advertised the custom `disjoint-v1` end-turn usage opt-in.

## What was exercised

Each editor opened a fresh throwaway project/profile and selected
**Kolega accounting smoke** as its actual external ACP agent. Its stdio relay
forwarded protocol bytes unchanged to:

```text
<worktree>/.venv/bin/python -m kolega_code.cli.main acp
```

This exercised the real CLI, ACP server, factory, provider SDK, ledger,
catalogue pricing, persistence, notifications, prompt responses, and editor UI.
It did not replace the agent factory or inject an agent-side usage object.

The provider was `deepseek`, model `deepseek-flash`, using its native
`https://api.deepseek.com/beta/responses` billing identity. A loopback HTTPS
CONNECT fixture terminated that route locally and supplied two streaming
responses. No request reached DeepSeek, and no paid request was made.

The provider returned token usage but **no monetary charge**. Catalogue pricing
was enabled normally, without pricing overrides. The two controlled response
texts were:

```text
EDITOR_SMOKE_FIRST: catalogue-backed accounting fixture completed.
EDITOR_SMOKE_SECOND: cumulative accounting fixture completed.
```

After selecting the agent, the exact prompts were:

```text
FIRST_ACCOUNTING_PROMPT: reply with fixture completion only.
SECOND_ACCOUNTING_PROMPT: reply with the second fixture completion only.
```

Both responses appeared in each editor's rendered transcript, in the same ACP
session. Both prompt responses returned `stopReason: end_turn`. There were
exactly two provider requests per
editor; all scripted replies were consumed and fixture `errors` remained `[]`.

### Observed values

| Observation | First completed turn | Second completed turn |
| --- | ---: | ---: |
| Provider input/output, this turn | `100 / 10` | `140 / 20` |
| Metadata prompt total tokens | `110` | `160` |
| Metadata session input/output | `100 / 10` | `240 / 30` |
| Metadata session total tokens | `110` | `270` |
| Session inference requests | `1` | `2` |
| Native cumulative cost, USD | `0.000021` | `0.000054` |
| Zed native context used/capacity | `41333 / 1000000` | `41492 / 1000000` |
| VS Code native context used/capacity | `41339 / 1000000` | `41498 / 1000000` |

Context occupancy is independent of consumed inference tokens: neither
editor received `270` as its context-used value. The effective full capacity
was `1000000`; the metadata's smaller usable-input budget was `616000`.

**Preflight counting caveat:** the existing hermetic fixture's
`_accounting_subprocess_bootstrap.py` seeds a byte-level offline tokenizer
and blocks external sockets. Thus these context-used numbers are fixture
estimates, not production tokenizer accuracy measurements. The capacity,
provider token usage, accounting calculation, and UI projection follow the
real implementation. No inference was used to refine context.

## Native Zed evidence

After the first response, hovering the context ring next to the model selector
showed the real native tooltip:

```text
Context
4% · 41k / 1M

Cost
0.0000 USD
```

After the second response, moving away and hovering again showed:

```text
Context
4% · 41k / 1M

Cost
0.0001 USD
```

These costs are Zed's **four-decimal display rounding** of `0.000021` and
`0.000054`, not zero spend and not a different accounting amount. The context
display also rounds, so both turns appear as `41k`.

An already-open tooltip retained its first-turn text across the second turn;
dismissing and reopening it refreshed to `0.0001 USD`. This is a display-refresh
caveat, not an additional provider request. The wire value was already
`0.000054` before the tooltip was reopened.

Screenshots were captured from the isolated Zed window after each completed
turn, including the refreshed second tooltip. The sanitized visible strings
above are transcribed from those screenshots; raw screen captures, accessibility
dumps, and protocol traffic stay in temporary scratch storage rather than
adding user paths or session identifiers to this report.

Scratch evidence manifest (SHA-256; filenames are relative to session scratch):

```text
168bbe61db7b7e18782c78857fbed302cfb419ec22cb738ec1c7fd29bda07f2e  zed-smoke/first-tooltip-crop.png
8be604b8a07de1c89a3dfd24133c33379dd819405317b7ae96065b44086b1fa6  zed-smoke/second-refreshed-crop.png
e6b5ba5a9ac4355e5431e955db40a2fe86c98f1ed901222df13cb29c2c54e78c  editor-smoke/second.png
```

## VS Code limitation and safe ignored metadata

ACP Client `0.2.0` rendered both completed response texts and stayed usable for
the second prompt, but **no context ring, token counter, or monetary cost was
rendered**. Inspecting its actual webview DOM after each response confirmed
the transcript and absence of accounting UI. Read-only inspection of the
installed extension's implementation was consistent with this behavior.
Neither the installed extension nor its settings were edited.

Both editors received `_meta["dev.kolega/usage"]` on session updates and prompt
responses, schema version `1`, with session snapshots, prompt totals, coverage,
and estimated-cost provenance. Neither client requested that extension. Their
continued prompt/response operation demonstrates safe acceptance/ignoring of
unrecognized namespaced metadata in these tested versions. This is a bounded
compatibility observation, not a claim that every ACP client supports it.

Native `PromptResponse.usage` was absent, as required without the opt-in.
Native `UsageUpdate` still contained normal context and cumulative cost fields.
Zed rendered those native fields, **not** a bespoke Kolega cost widget.
VS Code's failure to display them is a client limitation, not missing metadata.

Sanitized second-turn wire projection, common to both editors except context
occupancy:

```json
{
  "sessionUpdate": "usage_update",
  "used": 41492,
  "size": 1000000,
  "cost": {"amount": 0.000054, "currency": "USD"}
}
```

The authoritative second prompt-response metadata reported:

```json
{
  "schemaVersion": 1,
  "session": {
    "inputTokens": 240,
    "outputTokens": 30,
    "totalTokens": 270,
    "requests": {
      "requests": 2,
      "responses": 2,
      "reported": 2,
      "unreported": 0,
      "failed": 0,
      "open": 0,
      "abandoned": 0,
      "partial": 0,
      "missing": 0
    },
    "cost": {
      "monetary": [{
        "currency": "USD",
        "reported": "0",
        "estimated": "0.000054",
        "amount": "0.000054",
        "requests": 2,
        "source": "estimated"
      }],
      "non_monetary": [],
      "missing": 0,
      "partial": 0,
      "complete": true,
      "basis": "published_list_value_before_account_credits_discounts_and_taxes"
    }
  },
  "prompt": {
    "inputTokens": 140,
    "outputTokens": 20,
    "totalTokens": 160
  }
}
```

This is a selected projection, not a complete serialized response. Snapshot
revisions increased from `7` to `13` in Zed and `6` to `11` in VS Code. Revisions
are not token totals and need not agree between editor sessions. Repeated
notifications were replacement snapshots, not extra spend.

**Cost meaning:** the amount is an estimated published catalogue/list value
before credits, discounts, and taxes, not a measured account debit. The native
Zed tooltip does not expose this provenance. The metadata does.

## Reproduce without real credentials or changing a user profile

Prerequisites: the worktree's existing `.venv`, development test dependencies,
`openssl`, and an already-installed editor. Do not install or update extensions.
Use a separate **new scratch directory for each editor**.

The optional helper reuses `NativeProviderWire` and the isolated environment
from `tests/acp/test_accounting_subprocess.py`. The editor verification above
used the equivalent scratch harness; this committed helper packages that route
for repetition. It launches no editor itself:

```sh
# Run from this worktree. Keep this command running in its own terminal.
.venv/bin/python scripts/probe_acp_editor.py \
  --directory "$KOLEGA_SCRATCHPAD/editor-repro-zed"
```

Wait for `{"ready": true, ...}`. The new directory contains generated isolated
Kolega settings, fake credentials, a loopback certificate, editor profile
settings, and the throwaway `project/`.

For macOS Zed, in a separate terminal:

```sh
env ZED_STATELESS=1 ZED_EXPERIMENTAL_A11Y=1 \
  /Applications/Zed.app/Contents/MacOS/zed \
  --user-data-dir "$KOLEGA_SCRATCHPAD/editor-repro-zed/zed-data" \
  "$KOLEGA_SCRATCHPAD/editor-repro-zed/project"
```

`ZED_STATELESS=1` was necessary to keep this instance distinct from the already
running user Zed. Open the agent panel, select **Kolega accounting smoke**
from the new-thread agent menu, and send the two prompts above. Do not select
or start another registry agent. After each completed response, hover the
context ring; dismiss/reopen the tooltip before the second observation.

For VS Code, start the fixture again with a different new directory and use
its generated `vscode-profile/User/settings.json`. Launch a separate instance
with all three isolation flags:

```text
--user-data-dir=<short-path-to-generated-vscode-profile>
--shared-data-dir=<scratch>/shared-data
--extensions-dir=<scratch>/extensions
```

The scratch extensions directory may contain a read-only symlink to the
already-installed `formulahendry.acp-client-0.2.0` directory. Do not install,
patch, or update the user's extension. macOS's Unix socket path limit required
a short temporary profile alias in this run; that alias was removed afterward.
Also pass `--new-window --skip-welcome --skip-release-notes
--disable-workspace-trust` and the scratch project directory. Open the ACP
Client panel, select the generated agent, then send both prompts.

Inspect `<scratch>/provider-summary.json`: exactly two requests to
`api.deepseek.com`, `/beta/responses`, model `deepseek-flash`, zero remaining
replies, and `errors: []`. `traffic.jsonl` contains the real ACP exchange,
including native updates and prompt-response metadata; treat raw logs as
private because they contain local paths and session IDs.

The server accepts exactly two scripted inference responses. An unexpected
third request is a fixture error, not permission to contact a paid service.
Agent subprocesses use fake keys, an allowlisted clean environment, isolated
HOME/state, loopback proxy/CA, and the bootstrap's external-socket block.
Close only the isolated test instance and interrupt only its fixture server.
Do not use app-wide quit, `killall`, or the user's normal editor profile.

### Helper validation

- `ruff check scripts/probe_acp_editor.py`: passed.
- `ruff format --check scripts/probe_acp_editor.py`: passed.
- `pyright scripts/probe_acp_editor.py`: zero errors, warnings, or information.
- `scripts/probe_acp_editor.py --help`: passed.
- A separate fresh helper server plus the existing ACP SDK test client exercised
  the committed relay through initialize, new session, and both real CLI
  prompts. Assertions passed for `110 → 270` session tokens,
  `USD 0.000021 → 0.000054`, `end_turn`, and absent unrequested native end-turn
  usage. The provider summary contained exactly two native DeepSeek responses,
  no remaining replies, and no fixture errors. This helper validation is
  separate from, and does not substitute for, the real-editor observations.

## Safety and boundaries

- No production/source files, dependencies, installed extensions, user
  credentials, or persistent user settings were modified for this smoke test.
- An initial VS Code attempt omitted `--shared-data-dir` and briefly opened
  default shared storage. The completed two-turn test used isolated shared
  storage. No claim is made that the initial launch had zero shared-storage
  side effects; no user configuration was intentionally edited.
- An idle default Copilot selection in the throwaway Zed instance started an
  npm agent process before the custom agent was selected. It was stopped
  before sending any prompt. It was not part of the two-turn test.
- Only identified test editor/fixture processes were stopped. The existing
  user Zed process was left running. The editor worker removed profile alias
  `e` during cleanup; parent verification restored the same symlink target to
  preserve the checkout state requested by the user. It remains untracked.
  The isolated VS Code main process ignored termination and was force-stopped
  only after rechecking its exact scratch-profile arguments; its test CDP port
  was verified closed.
- No source changes, commits, paid requests, or extension installs were made.
- This does not test real provider billing accuracy, production tokenization,
  every editor version, every metadata field, or all lifecycle cases. Those
  require their separate protocol/provider/accounting tests.
