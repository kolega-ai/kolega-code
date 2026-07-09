---
title: Terminal & tmux shortcuts
description: Why Shift chords fail in tmux/screen and the portable fallbacks Kolega Code supports.
---

Kolega Code’s TUI uses some Shift-modified shortcuts (`Shift+Enter`, `Shift+Tab`,
`Ctrl+Shift+V`). In a modern terminal with the [Kitty keyboard protocol](https://sw.kovidgoyal.net/kitty/keyboard-protocol/),
those chords reach the app. Inside **tmux** or **GNU screen**, the multiplexer
often sits between the terminal and the app and either drops modifiers or
never forwards the extended key sequences — so Shift chords look like plain
keys, or never arrive at all.

Kolega Code cannot invent modifier information the terminal never sends. Instead
it provides **portable fallbacks** that work without Shift, plus optional tmux
config if you want the original chords restored.

## Portable shortcuts

| Action | Preferred (capable terminals) | Portable fallback |
| --- | --- | --- |
| Send prompt | `Enter` | — |
| Insert newline | `Shift+Enter` | `Ctrl+J` or `Ctrl+Enter` |
| Toggle Plan ⇄ Build | `Shift+Tab` | `/plan` or `/build` |
| Paste image from system clipboard | `Ctrl+Shift+V` | `Alt+V` (Option+V on macOS when the terminal treats Option as Meta) or `/attach` with no path |
| Attach image from a file | — | `/attach <path>` or `@image.png` |

On session start inside tmux/screen, the startup block prints a short reminder
of these fallbacks.

### Image paste notes

- `/attach` with **no path** reads an image from the **system** clipboard (same
  path as `Ctrl+Shift+V` / `Alt+V`). This is the most reliable option under
  plain tmux.
- Clipboard image reading uses host tools independent of tmux:
  - **macOS:** `osascript` (built-in), or `pngpaste` if installed
  - **Linux:** `xclip` or `wl-paste`
  - **Windows / WSL:** PowerShell clipboard APIs
- If Option+V types a special character (e.g. `√`) instead of pasting an image,
  your terminal is not sending Alt as a modifier. Prefer `/attach`, or enable
  “Option as Meta” / “Left Option key acts as +Esc” in the terminal settings.

## Optional: restore Shift+Enter in tmux

With **tmux 3.2+** and an outer terminal that supports extended keys, add to
`~/.tmux.conf`:

```tmux
set -g extended-keys always
set -g extended-keys-format csi-u
set -as terminal-features 'xterm*:extkeys'
```

Then reload (`tmux source-file ~/.tmux.conf`) or restart tmux.

Also make sure `default-terminal` / `terminal-overrides` match the outer
terminal’s `$TERM` family (`xterm-256color`, `xterm-ghostty`, etc.) so the
`extkeys` feature actually applies.

After that, `Shift+Enter` and similar chords often work again inside Kolega Code.
If they still do not, keep using the portable fallbacks above.

## Related

- [Chat Composer](../../tui/composer/) — full composer key reference
- [Interface Tour](../../tui/interface/) — key bindings at a glance
- [Slash Commands](../../tui/slash-commands/) — `/attach`, `/plan`, `/build`
