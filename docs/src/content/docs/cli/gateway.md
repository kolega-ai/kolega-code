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
(voice transcription needs the `stt` extra).

## Setup

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy its token.
2. Save it (optionally checking it against Telegram and saving your
   allowlist — it also switches the adapter on):

   ```bash
   kolega-code gateway telegram setup --verify
   ```

   The command prompts for the token (never echoed) and your Telegram user id
   (ask [@userinfobot](https://t.me/userinfobot) for it). Everything is
   stored in `settings.json` like every other key — see
   [Gateway settings](../configuration/environment-variables.md#gateway).
3. Run it:

   ```bash
   kolega-code gateway run --project ~/kolega-code-workspace
   ```

Without `--project`/`gateway.project`, sessions work in
`~/kolega-code-workspace` — never the daemon's launch directory. Message the
bot from your phone; unknown senders are dropped (or get a pairing code, see
below).

## Commands

| Command | Description |
| --- | --- |
| `run` | Run the gateway in the foreground (graceful stop on Ctrl-C/SIGTERM). |
| `status` | Show whether the daemon is running, with a heartbeat freshness check. |
| `telegram setup` | Save the @BotFather token (with optional `--verify`, `--allow <ids>`, `--clear`). |
| `pairing list` / `pairing approve <code>` | List pending sender pairing requests, or admit a sender. |
| `install` / `uninstall` | Install or remove the gateway as a user-level background service (systemd user unit / launchd agent — no root needed). |

`run` options: `--adapter echo|telegram`, `--project`, `--state-dir`,
`--provider`, `--model`.

## In-chat commands

`/new`, `/status`, `/model [model]`, `/permissions [ask|auto]`, `/stop`, `/help`.
The same commands are registered with Telegram so they appear in the slash
menu while typing.

## Pairing new senders

With an allowlist configured and `gateway.pairing_enabled` set in
`settings.json`, an unknown sender gets a short code and instructions. The
operator runs:

```bash
kolega-code gateway pairing approve <code>
```

and the sender's next message goes through. Codes expire after an hour.

## Voice transcription

Set `gateway.stt_enabled` in `settings.json`, install the extra, and voice
notes transcribe locally with faster-whisper (model size via
`gateway.stt_model`, default `base`):

```bash
uv pip install "kolega-code[stt]"
```

## Notes

- Groups are mention-gated (`@yourbot`) and optionally restricted to
  `gateway.group_ids`.
- The echo adapter (`--adapter echo`) drives the full transport pipeline
  over stdin/stdout without any LLM or Telegram account — handy for testing.
- The Telegram connection uses the official Bot API with a BotFather token,
  never a personal user account.
