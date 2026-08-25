List the other agent sessions that are live in this host process and can
receive your messages.

Returns each peer session's name, current status (idle or busy), project
directory, and a short session id. This session is never included. Use this
before send_message whenever you are not sure a peer exists, what it is
called, or whether it is mid-turn — addressing a peer by name prefix fails
loudly if the prefix is ambiguous.

Peers are sessions of the same host process only; nothing here sees other
machines.
