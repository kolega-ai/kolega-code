Find text or a regular expression in the accessibility snapshot.

Provide exactly one of text or regex. This is cheaper than requesting a
full snapshot when locating a specific element.

A miss distinguishes three cases: absent from the page, present in the page
but outside the region the snapshot covered, and undetermined because the
search was truncated. Only the first is a reliable absence.
