Capture the current page's accessibility snapshot.

Prefer this over screenshots when deciding what to interact with. Interactive
nodes include stable refs such as e12 that can be passed to action tools.

On a page too large to fit one snapshot, nodes nearest the viewport are shown
first and a Coverage line states what was left out. That is an instruction to
narrow the scope — pass a target, or browser_scroll and snapshot again — not a
sign the page is unreadable.
