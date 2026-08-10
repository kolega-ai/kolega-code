Activate an Agent Skill and load its full instructions.

Call this before using a skill's specialized workflow. The returned content explains where skill resources
live and lists them; read any resource with the `read` tool, whose `file_path` is relative to the absolute
skill directory given in the output.

Returns:
    The activated skill instructions, or a note if the skill is already active.
