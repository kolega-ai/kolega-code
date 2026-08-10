List, create, close, or select browser tabs.

The action decides which other argument applies, and anything supplied for
the rest is ignored: list uses neither, new uses url, and select and close
use index. Tab indices shift after a close, so re-list before acting again.

Args:
    action: One of list, new, close, or select.
    index: Tab index, required for select. For close it defaults to the
        current tab. 0 is a real tab index.
    url: URL for a new tab; omit it for a blank tab.
