---
title: gateway
description: Run the messaging gateway and talk to your agents from Telegram.
---

# `kolega-code gateway`

The messaging gateway runs a long-lived daemon that connects Kolega Code to a
chat platform, so you can drive your agent from your phone. Today it ships a
Telegram adapter (official Bot API via [@BotFather](https://t.me/BotFather));
the adapter layer is designed for more platforms later.

Each chat gets its own durable Kolega session. Turns stream into the chat as
edit-in-place messages, permission approvals and `ask_user_choice` questions
arrive as inline buttons, and voice notes, images, and documents are handled
(the local voice-transcription provider needs the `stt` extra).

:::caution[Trusted operators only]
Every configured or locally paired user is a **trusted operator** of the local
agent, not a restricted guest. In `ask` mode, tool confirmations go to the
requesting chat: admitted users can approve their own tools and switch their
session to `/permissions auto`. This is not separate machine-owner
authentication or a security sandbox. In groups, admitted participants share
the chat's session and control surface.
:::

## Setup

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy its token.
2. Save it and explicitly authorize your numeric Telegram user ID — either from
   the CLI (replace the example ID with your own):

   ```bash
   kolega-code gateway telegram setup --verify --allow '123456789'
   ```

   or from the TUI: **Settings → Gateway** (bot token, allowed users, pairing,
   permission mode, adapter, and project; voice transcription lives under
   **Settings → Tools**). Everything is
   stored in `settings.json` like every other key — see
   [Gateway settings](../configuration/environment-variables.md#gateway).
3. Run it:

   ```bash
   kolega-code gateway run --project ~/kolega-code-workspace
   ```

Without `--project`/`gateway.project`, sessions work in
`~/kolega-code-workspace` — never the daemon's launch directory. Message the
bot from your phone; unknown senders are dropped (or get a pairing code, see
[Pairing the first or another sender](#pairing-the-first-or-another-sender)).

`--allow '123456789,987654321'` authorizes multiple users. Use numeric IDs, not
`@handles`, display names, or `*`. The bot token authenticates the bot to
Telegram; it does not authorize a person to use your agent.

Token-only setup, including piped input, is a successful **configuration save**,
not proof that anyone can use the bot. When no operators are authorized, setup
warns that access is locked (or pairing-only if already enabled) and explains
how to supply `--allow` or enable pairing in **Settings → Gateway**. Setup never
enables pairing implicitly.

- Omitting `--allow` preserves the existing configured IDs.
- Explicit `--allow ''` clears only `gateway.allowed_users`. It does **not**
  revoke users previously approved through pairing.
- Settings-based access changes require a gateway restart; saving settings
  does not automatically restart the service.

## Commands

| Command | Description |
| --- | --- |
| `run` | Run the gateway in the foreground (graceful stop on Ctrl-C/SIGTERM). |
| `status` | Show daemon/service health, heartbeat freshness, and the running daemon's effective access policy. |
| `telegram setup` | Save the @BotFather token (with optional `--verify`, `--allow <ids>`, `--clear`). |
| `pairing list` / `pairing approve <code>` | List pending sender pairing requests, or admit a sender. |
| `install` / `uninstall` / `restart` | Install, remove, or restart the gateway as a user-level background service (systemd user unit / launchd agent — no root needed). |

`run` options: `--adapter echo|telegram`, `--project`, `--state-dir`,
`--provider`, `--model`.

Running `kolega-code update` automatically restarts the gateway service if it is installed, so the running daemon picks up new versions immediately.
For other upgrade methods or a foreground daemon, explicitly restart the
process. Updating files on disk does not harden an old process still running.

The background service launches through your login shell (`$SHELL`), so the
daemon sees the same environment as an interactive terminal — service
managers alone provide a minimal `PATH` that would hide user-installed tools
from gateway-driven sessions. Restart the service after changing your shell
profile to pick up the new environment.

### Running access policy

Startup and `kolega-code gateway status` report the enforced policy, configured
user count, valid paired-user count, and whether pairing is enabled. Status
uses the running daemon's heartbeat, not newly edited settings, to describe
what that process enforces:

| Policy | Meaning |
| --- | --- |
| `locked` | No configured or valid paired users, and pairing is disabled. Remote messages cannot use the agent. |
| `pairing-only` | No authorized users yet; unknown senders can request a code, but cannot use the agent until locally approved. |
| `restricted` | Only configured or valid paired users are authorized; group restrictions still apply. |

A locked daemon can remain running and warns at startup when onboarding is
disabled. Older heartbeats without access fields show **access policy unknown;
restart/update to verify** — they are not evidence of safe enforcement. Local
echo is identified separately from remote access.

## In-chat commands

`/new`, `/status`, `/model [model]`, `/permissions [ask|auto]`, `/stop`, `/help`.
The same commands are registered with Telegram so they appear in the slash
menu while typing.

## Access rules

Remote access is deny-by-default. Authorized sender IDs are the **union** of
`gateway.allowed_users` in `settings.json` and valid approvals in
`gateway_allowlist.json`, both in the gateway's state directory. An empty
configured list means paired users only; if neither source authorizes anyone,
the gateway is locked with pairing off or onboarding-only with pairing on.
There is no implicit public mode, wildcard, or first-message-wins admission.

Unknown senders cannot create or resume sessions, run slash commands, approve
tools, change permissions, or trigger attachment downloads, file writes, or
transcription. Enabling pairing permits only the onboarding reply, not agent
access. Rejected permission-button taps do not consume the legitimate pending
prompt and do not generate pairing replies.

Group admission also requires an authorized sender. `gateway.group_ids` and
mention gating restrict where the bot responds; neither grants permission to
otherwise unauthorized group members.

## Pairing the first or another sender

Pairing can onboard the **first user**, with no configured IDs. It is disabled
by default and always requires approval from the local machine:

1. Save the bot token, then enable pairing in **Settings → Gateway** and apply
   the change. You may leave allowed users empty.
2. Start the gateway, or restart it if it was already running, so it loads
   `gateway.pairing_enabled: true`.
3. From the intended Telegram account, **DM the bot** to receive a pairing code.
   This does not create an agent session or grant access.
4. On the local machine, using the same state directory as the daemon, run:

   ```bash
   kolega-code gateway pairing list
   ```

   Review the code, **numeric sender ID**, display name, channel, and chat.
   Confirm the code with the intended person through a trusted channel.
   Display names alone are not proof of identity.
5. Approve only that confirmed request locally:

   ```bash
   kolega-code gateway pairing approve <code>
   ```

6. **DM the bot again.** Persisted approvals are reread dynamically, so the
   approved user's next message is admitted without restarting the daemon.

Codes expire after one hour by default (`gateway.pairing_code_ttl_seconds`).
Other unknown senders remain unauthorized. Disabling pairing stops onboarding,
but does not revoke existing approvals.

## Upgrade and exclusive-owner lockdown

**Intentional security change:** old deployments that relied on an empty list
to admit everyone become locked or pairing-only after restarting with the
updated code, unless valid persisted approvals already authorize users.
Restart any old daemon; an upgrade on disk alone is insufficient. No existing
sessions or approvals are automatically deleted.

To restrict a deployment to one owner:

1. Stop the gateway before editing access data. For a foreground run, use
   Ctrl-C. For an installed service, `kolega-code gateway uninstall` stops and
   removes the service registration without deleting settings or approvals;
   check `gateway status` to confirm it is no longer running.
2. Inspect **both** `settings.json` → `gateway.allowed_users` and
   `gateway_allowlist.json` in that daemon's state directory. Remove unwanted
   approval records and set the configured list to only the owner's quoted
   numeric ID, such as `["123456789"]`. Set `gateway.pairing_enabled` to `false`
   if no further onboarding is wanted. Preserve unrelated settings, tokens,
   credentials, and any intended approval records.
3. Restart the foreground run, or use `kolega-code gateway install` to restore
   and start the service removed in step 1. Check `gateway status` for the
   running policy and counts.

Clearing only one authorization source does not clear the other. In particular,
`telegram setup --allow ''` does not revoke persisted approvals, and removing
approvals does not revoke configured users. Removing the final authorization
locks access rather than opening it.

### Repairing malformed access settings

Malformed security settings fail with an actionable error naming the field and
expected form; they are not silently discarded or replaced with permissive
defaults. Repair the identified field in `settings.json` — **do not delete the
settings file or credentials**:

- `gateway` must be a JSON object when present.
- `gateway.allowed_users` and `gateway.group_ids` must be lists of nonempty
  strings. For Telegram, use positive ASCII-decimal user IDs and signed nonzero
  ASCII-decimal group IDs, for example `["123456789"]` and
  `["-1001234567890"]`. JSON numbers, booleans, scalar strings, blank entries,
  handles, and wildcards are invalid.
- `gateway.pairing_enabled` must be the JSON boolean `true` or `false`, not a
  string such as `"false"`.

Missing keys and empty lists are valid safe settings; surrounding ID whitespace
is trimmed and duplicates are removed. Invalid CLI/TUI input is rejected before
saving, including before replacing a token. After repairing settings, restart
the gateway to load the correction.

## Voice transcription

Voice notes transcribe remotely through Groq's hosted `whisper-large-v3-turbo`
(the same provider/model Hermes uses), configured in the settings TUI under
**Tools → Voice transcription**. `stt_enabled`, `stt_provider`, and `stt_model`
live at the top level of `settings.json`; the provider reuses the Groq API key
stored on the Providers page (or `GROQ_API_KEY` in the gateway's environment).
Audio is uploaded to Groq's transcription endpoint for the request and never
stored. Transcription is remote-only — there is no local whisper backend.

## Notes

- Groups are mention-gated (`@yourbot`) and optionally restricted to
  `gateway.group_ids`.
- The echo adapter (`--adapter echo`) drives the full transport pipeline
  over stdin/stdout without any LLM or Telegram account — handy for testing.
  Only the actual local echo adapter's standard `owner` sender is admitted
  without an explicit configured allowlist; explicit lists retain their
  restrictions and custom echo sender IDs must be authorized. This local
  console allowance never applies to Telegram or arbitrary adapters.
- The Telegram connection uses the official Bot API with a BotFather token,
  never a personal user account.
