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
| `kolega-code browser status` | Show whether the native host and Chrome extension configuration are installed and valid |
| `kolega-code browser doctor` | Run detailed checks for manifest, executable, registration, and connection problems |
| `kolega-code browser uninstall` | Remove Kolega-owned native-host registration and configuration files |

Once `browser status` reports a valid configuration, the Chrome integration is
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
