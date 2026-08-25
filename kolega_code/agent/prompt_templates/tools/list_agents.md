List the other agent sessions you can message: sessions of this host
process, plus other kolega-code processes sharing this machine's state
directory (TUIs and headless goal/loop workers alike).

Returns each peer session's name, current status (idle, busy, or
unreachable), project directory when known, and a short session id.
This session is never included. Use this before send_message whenever
you are not sure a peer exists, what it is called, or whether it is
reachable — addressing a peer by name prefix fails loudly if the prefix
is ambiguous.

Peers are same-machine only; nothing here sees other machines.
