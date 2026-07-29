---
title: browser
description: Install, verify, troubleshoot, and remove the Kolega Chrome integration.
---

The Kolega Chrome integration lets the browser agent work through the **Kolega
Browser Companion** extension in your regular Chrome installation. Playwright
remains the default browser backend. This integration currently supports macOS
and Google Chrome Stable only.

## Install

Install the production native host and open the official Chrome Web Store listing:

```bash
kolega-code browser install
```

- **Chrome Web Store:** [Kolega Browser Companion](https://chromewebstore.google.com/detail/edihigldhbmimflgjkohkgnjefmhngdn)
- **Production extension ID:** `edihigldhbmimflgjkohkgnjefmhngdn`

Running `browser install` registers the per-user native-messaging host and opens
the Web Store listing; it does not install the extension silently. Installing the
extension through Chrome's **Add to Chrome** flow is the browser-access consent
boundary. Review the listing and verify its extension ID before accepting Chrome's
permission disclosure.

If Chrome was open while the native host was registered, fully quit and restart
Chrome before checking the connection.

## Commands

| Command | What it does |
| --- | --- |
| `kolega-code browser install` | Register or refresh the Kolega native host and open the production Web Store listing |
| `kolega-code browser status` | Check the native-host manifest only — a fast, offline configuration check |
| `kolega-code browser doctor` | Everything `status` checks, plus a real connection attempt to the extension |
| `kolega-code browser uninstall` | Remove Kolega-owned native-host registration and configuration files |

`status` deliberately does not talk to Chrome, so a valid manifest there does not
mean the extension is reachable. Use `doctor` to test the connection; it reports
one of:

| `doctor` extension state | Meaning |
| --- | --- |
| `paired` | The extension is connected and this session may drive the browser |
| `awaiting_selection` | Connected, but several Kolega sessions are advertised and you must pick one |
| `connected_not_selected` | Connected, but no session is confirmed yet |
| `unreachable` | No connection — check the extension is installed and enabled, and Chrome is running |

`doctor` exits non-zero unless the state is `paired`.

Once `browser doctor` reports `paired`, the Chrome integration is
available to the browser agent. Playwright is still selected by default; direct
the agent in your prompt when you want it to use Chrome, for example:

> Use Chrome to check the checkout flow.

The agent chooses Chrome for that task. There is no `--chrome` switch and no saved
browser-backend setting.

## Fixed Chrome operations

The Chrome integration exposes only these fixed protocol operations:

- Navigation: `browser.navigate`, `browser.navigate_back`
- Page inspection and waits: `browser.snapshot`, `browser.find`,
  `browser.wait_for`
- Interaction: `browser.click`, `browser.type`, `browser.fill_form`,
  `browser.select_option`, `browser.hover`, `browser.drag`,
  `browser.press_key`
- Tabs and inspection: `browser.tabs`, `browser.network_requests`,
  `browser.screenshot`
- Disconnect: `browser.detach`

This list is exhaustive. The integration does not expose arbitrary JavaScript,
raw Chrome DevTools Protocol (CDP), cookies or browser storage, request or
response headers or bodies, file uploads or file/data drop, or console messages.

### Choosing which session controls the browser

The extension connects to the native host on its own; you do not need to open it
in normal use. When exactly one Kolega session is running it is selected
automatically.

When two or more sessions are running, the extension will not guess which one
may drive your browser, because selecting a session grants a local process
access to pages in your ordinary profile. The toolbar badge shows `!` while a
choice is pending — click the extension and pick a session. Run
`kolega-code browser doctor` to see the competing sessions and which runtime id
belongs to the session you are using.

Your choice is remembered for as long as Chrome stays open, including across
service-worker restarts, and is cleared when Chrome restarts.

### Regular expressions

`browser.find` and the `filter_pattern` of `browser.network_requests` accept a
restricted, linear-time subset of regular-expression syntax: literals, `.`,
character classes, anchors, escapes, and the quantifiers `?`, `*`, `+` and
`{n,m}` applied to a single character or class. At most four quantifiers are
allowed and repetition counts may not exceed 1000.

Groups `(` `)`, alternation `|`, and backreferences are rejected, which is what
makes the quantifiers safe: with no grouping or alternation a quantifier can only
ever repeat one atom, so catastrophic backtracking is impossible. Write
`[0-9]{4}` rather than `(\d{4}|\d{2})`.

The Playwright backend accepts full Python regular expressions, so a pattern that
works there may be rejected on Chrome.

### Screenshots

`browser.screenshot` returns the image inline. Neither the Chrome nor the
Playwright backend writes screenshots to disk, so there is no artifact file path
to report. Capture only the region you need: a full-page screenshot of a long
page produces a large inline image.

## Native-host manifest location

On macOS, the installer writes the per-user manifest to:

`~/Library/Application Support/Google/Chrome/NativeMessagingHosts/ai.kolega.browser.json`

## Uninstall and troubleshoot

Run:

```bash
kolega-code browser uninstall
```

This removes only Kolega-owned native-host files and registration. Remove the
Chrome extension separately from `chrome://extensions`, then restart Chrome.

If setup stops working:

1. Run `kolega-code browser status` for the current state.
2. Run `kolega-code browser doctor` for the failing path or registration check.
3. After installing, reinstalling, or upgrading Kolega Code, run
   `kolega-code browser install` again to refresh the executable path.
4. Fully quit and restart Chrome so it reloads the native-host manifest.
5. Confirm the installed extension ID is
   `edihigldhbmimflgjkohkgnjefmhngdn`.
6. If the agent still opens Playwright, explicitly direct it to use Chrome in
   the task.
