from __future__ import annotations


def task_list_entries(markdown: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith(("- [x]", "- [X]")):
            text = stripped[5:].strip()
            status = "completed"
        elif stripped.startswith("- [ ]"):
            text = stripped[5:].strip()
            status = "pending"
        elif stripped.startswith("- "):
            text = stripped[2:].strip()
            status = "pending"
        else:
            continue
        if text:
            entries.append({"text": text, "status": status})
    if not entries and markdown.strip():
        entries.append({"text": markdown.strip(), "status": "pending"})
    return entries
