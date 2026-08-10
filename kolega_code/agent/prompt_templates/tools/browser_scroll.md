Move the viewport, then return the updated page snapshot.

Supply exactly one movement. On a page too large to snapshot in one call,
scroll and re-snapshot: the snapshot prioritises what is near the viewport,
so moving the viewport is how you reach the rest.
