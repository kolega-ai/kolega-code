import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from .base_tool import BaseTool

APPLY_PATCH_TOOL_DESC = """This is a custom utility that makes it more convenient to add, remove, move, or edit code files. `apply_patch` effectively allows you to execute a diff/patch against a file, but the format of the diff specification is unique to this task, so pay careful attention to these instructions. To use the `apply_patch` command, you should pass a message of the following structure as "input":

%%bash
apply_patch <<"EOF"
*** Begin Patch
[YOUR_PATCH]
*** End Patch
EOF

Where [YOUR_PATCH] is the actual content of your patch, specified in the following V4A diff format.

*** [ACTION] File: [path/to/file] -> ACTION can be one of Add, Update, or Delete. The path may be relative to the project root (preferred) or absolute to act outside the project.
For each snippet of code that needs to be changed, repeat the following:
[context_before] -> See below for further instructions on context.
- [old_code] -> Precede the old code with a minus sign.
+ [new_code] -> Precede the new, replacement code with a plus sign.
[context_after] -> See below for further instructions on context.

For instructions on [context_before] and [context_after]:
- By default, show 3 lines of code immediately above and 3 lines immediately below each change. If a change is within 3 lines of a previous change, do NOT duplicate the first change’s [context_after] lines in the second change’s [context_before] lines.
- If 3 lines of context is insufficient to uniquely identify the snippet of code within the file, use the @@ operator to indicate the class or function to which the snippet belongs. For instance, we might have:
@@ class BaseClass
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

- If a code block is repeated so many times in a class or function such that even a single @@ statement and 3 lines of context cannot uniquely identify the snippet of code, you can use multiple `@@` statements to jump to the right context. For instance:

@@ class BaseClass
@@ 	def method():
[3 lines of pre-context]
- [old_code]
+ [new_code]
[3 lines of post-context]

Note, then, that we do not use line numbers in this diff format, as the context is enough to uniquely identify code. An example of a message that you might pass as "input" to this function, in order to apply a patch, is shown below.

%%bash
apply_patch <<"EOF"
*** Begin Patch
*** Update File: pygorithm/searching/binary_search.py
@@ class BaseClass
@@     def search():
-          pass
+          raise NotImplementedError()

@@ class Subclass
@@     def search():
-          pass
+          raise NotImplementedError()

*** End Patch
EOF
"""


# Filesystem helpers will be replaced with filesystem service calls in apply_patch method


# Patch types
class ActionType(str, Enum):
    ADD = "add"
    DELETE = "delete"
    UPDATE = "update"


@dataclass
class FileChange:
    type: ActionType
    old_content: Optional[str] = None
    new_content: Optional[str] = None
    move_path: Optional[str] = None


@dataclass
class Commit:
    changes: Dict[str, FileChange] = field(default_factory=dict)


@dataclass
class Chunk:
    orig_index: int = -1
    del_lines: List[str] = field(default_factory=list)
    ins_lines: List[str] = field(default_factory=list)


@dataclass
class PatchAction:
    type: ActionType
    new_file: Optional[str] = None
    chunks: List[Chunk] = field(default_factory=list)
    move_path: Optional[str] = None


@dataclass
class Patch:
    actions: Dict[str, PatchAction] = field(default_factory=dict)


# Errors


class DiffError(ValueError):
    """Any problem detected while parsing or applying a patch."""


# Patch helpers


def find_context_core(lines: List[str], context: List[str], start: int) -> Tuple[int, int]:
    if not context:
        return start, 0

    for i in range(start, len(lines)):
        if lines[i : i + len(context)] == context:
            return i, 0
    for i in range(start, len(lines)):
        if [s.rstrip() for s in lines[i : i + len(context)]] == [s.rstrip() for s in context]:
            return i, 1
    for i in range(start, len(lines)):
        if [s.strip() for s in lines[i : i + len(context)]] == [s.strip() for s in context]:
            return i, 100
    return -1, 0


def find_context(lines: List[str], context: List[str], start: int, eof: bool) -> Tuple[int, int]:
    if eof:
        new_index, fuzz = find_context_core(lines, context, len(lines) - len(context))
        if new_index != -1:
            return new_index, fuzz
        new_index, fuzz = find_context_core(lines, context, start)
        return new_index, fuzz + 10_000
    return find_context_core(lines, context, start)


def peek_next_section(lines: List[str], index: int) -> Tuple[List[str], List[Chunk], int, bool]:
    old: List[str] = []
    del_lines: List[str] = []
    ins_lines: List[str] = []
    chunks: List[Chunk] = []
    mode = "keep"
    orig_index = index

    while index < len(lines):
        s = lines[index]
        if s.startswith(
            (
                "@@",
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            )
        ):
            break
        if s == "***":
            break
        if s.startswith("***"):
            raise DiffError(f"Invalid Line: {s}")
        index += 1

        last_mode = mode
        if s == "":
            s = " "
        if s[0] == "+":
            mode = "add"
        elif s[0] == "-":
            mode = "delete"
        elif s[0] == " ":
            mode = "keep"
        else:
            raise DiffError(f"Invalid Line: {s}")
        s = s[1:]

        if mode == "keep" and last_mode != mode:
            if ins_lines or del_lines:
                chunks.append(
                    Chunk(
                        orig_index=len(old) - len(del_lines),
                        del_lines=del_lines,
                        ins_lines=ins_lines,
                    )
                )
            del_lines, ins_lines = [], []

        if mode == "delete":
            del_lines.append(s)
            old.append(s)
        elif mode == "add":
            ins_lines.append(s)
        elif mode == "keep":
            old.append(s)

    if ins_lines or del_lines:
        chunks.append(
            Chunk(
                orig_index=len(old) - len(del_lines),
                del_lines=del_lines,
                ins_lines=ins_lines,
            )
        )

    if index < len(lines) and lines[index] == "*** End of File":
        index += 1
        return old, chunks, index, True

    if index == orig_index:
        raise DiffError("Nothing in this section")
    return old, chunks, index, False


@dataclass
class Parser:
    current_files: Dict[str, str]
    lines: List[str]
    index: int = 0
    patch: Patch = field(default_factory=Patch)
    fuzz: int = 0

    # ------------- low-level helpers -------------------------------------- #
    def _cur_line(self) -> str:
        if self.index >= len(self.lines):
            raise DiffError("Unexpected end of input while parsing patch")
        return self.lines[self.index]

    @staticmethod
    def _norm(line: str) -> str:
        """Strip CR so comparisons work for both LF and CRLF input."""
        return line.rstrip("\r")

    # ------------- scanning convenience ----------------------------------- #
    def is_done(self, prefixes: Optional[Tuple[str, ...]] = None) -> bool:
        if self.index >= len(self.lines):
            return True
        if prefixes and len(prefixes) > 0 and self._norm(self._cur_line()).startswith(prefixes):
            return True
        return False

    def startswith(self, prefix: Union[str, Tuple[str, ...]]) -> bool:
        return self._norm(self._cur_line()).startswith(prefix)

    def read_str(self, prefix: str) -> str:
        """
        Consume the current line if it starts with *prefix* and return the text
        **after** the prefix.  Raises if prefix is empty.
        """
        if prefix == "":
            raise ValueError("read_str() requires a non-empty prefix")
        if self._norm(self._cur_line()).startswith(prefix):
            text = self._cur_line()[len(prefix) :]
            self.index += 1
            return text
        return ""

    def read_line(self) -> str:
        """Return the current raw line and advance."""
        line = self._cur_line()
        self.index += 1
        return line

    # ------------- public entry point -------------------------------------- #
    def parse(self) -> None:
        while not self.is_done(("*** End Patch",)):
            # ---------- UPDATE ---------- #
            path = self.read_str("*** Update File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Duplicate update for file: {path}")
                move_to = self.read_str("*** Move to: ")
                if path not in self.current_files:
                    raise DiffError(f"Update File Error - missing file: {path}")
                text = self.current_files[path]
                action = self._parse_update_file(text)
                action.move_path = move_to or None
                self.patch.actions[path] = action
                continue

            # ---------- DELETE ---------- #
            path = self.read_str("*** Delete File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Duplicate delete for file: {path}")
                if path not in self.current_files:
                    raise DiffError(f"Delete File Error - missing file: {path}")
                self.patch.actions[path] = PatchAction(type=ActionType.DELETE)
                continue

            # ---------- ADD ---------- #
            path = self.read_str("*** Add File: ")
            if path:
                if path in self.patch.actions:
                    raise DiffError(f"Duplicate add for file: {path}")
                if path in self.current_files:
                    raise DiffError(f"Add File Error - file already exists: {path}")
                self.patch.actions[path] = self._parse_add_file()
                continue

            raise DiffError(f"Unknown line while parsing: {self._cur_line()}")

        if not self.startswith("*** End Patch"):
            raise DiffError("Missing *** End Patch sentinel")
        self.index += 1  # consume sentinel

    # ------------- section parsers ---------------------------------------- #
    def _parse_update_file(self, text: str) -> PatchAction:
        action = PatchAction(type=ActionType.UPDATE)
        lines = text.split("\n")
        index = 0
        while not self.is_done(
            (
                "*** End Patch",
                "*** Update File:",
                "*** Delete File:",
                "*** Add File:",
                "*** End of File",
            )
        ):
            def_str = self.read_str("@@ ")
            section_str = ""
            if not def_str and self._norm(self._cur_line()) == "@@":
                section_str = self.read_line()

            if not (def_str or section_str or index == 0):
                raise DiffError(f"Invalid line in update section:\n{self._cur_line()}")

            if def_str.strip():
                found = False
                if def_str not in lines[:index]:
                    for i, s in enumerate(lines[index:], index):
                        if s == def_str:
                            index = i + 1
                            found = True
                            break
                if not found and def_str.strip() not in [s.strip() for s in lines[:index]]:
                    for i, s in enumerate(lines[index:], index):
                        if s.strip() == def_str.strip():
                            index = i + 1
                            self.fuzz += 1
                            found = True
                            break

            next_ctx, chunks, end_idx, eof = peek_next_section(self.lines, self.index)
            new_index, fuzz = find_context(lines, next_ctx, index, eof)
            if new_index == -1:
                ctx_txt = "\n".join(next_ctx)
                raise DiffError(f"Invalid {'EOF ' if eof else ''}context at {index}:\n{ctx_txt}")
            self.fuzz += fuzz
            for ch in chunks:
                ch.orig_index += new_index
                action.chunks.append(ch)
            index = new_index + len(next_ctx)
            self.index = end_idx
        return action

    def _parse_add_file(self) -> PatchAction:
        lines: List[str] = []
        while not self.is_done(("*** End Patch", "*** Update File:", "*** Delete File:", "*** Add File:")):
            s = self.read_line()
            if not s.startswith("+"):
                raise DiffError(f"Invalid Add File line (missing '+'): {s}")
            lines.append(s[1:])  # strip leading '+'
        return PatchAction(type=ActionType.ADD, new_file="\n".join(lines))


def _get_updated_file(text: str, action: PatchAction, path: str) -> str:
    if action.type is not ActionType.UPDATE:
        raise DiffError("_get_updated_file called with non-update action")
    orig_lines = text.split("\n")
    dest_lines: List[str] = []
    orig_index = 0

    for chunk in action.chunks:
        if chunk.orig_index > len(orig_lines):
            raise DiffError(f"{path}: chunk.orig_index {chunk.orig_index} exceeds file length")
        if orig_index > chunk.orig_index:
            raise DiffError(f"{path}: overlapping chunks at {orig_index} > {chunk.orig_index}")

        dest_lines.extend(orig_lines[orig_index : chunk.orig_index])
        orig_index = chunk.orig_index

        dest_lines.extend(chunk.ins_lines)
        orig_index += len(chunk.del_lines)

    dest_lines.extend(orig_lines[orig_index:])
    return "\n".join(dest_lines)


class ApplyPatchTool(BaseTool):

    def load_files(self, paths: List[str], open_fn: Callable[[str], str]) -> Dict[str, str]:
        return {path: open_fn(path) for path in paths}

    def identify_files_needed(self, text):
        lines = text.splitlines()
        return [line[len("*** Update File: ") :] for line in lines if line.startswith("*** Update File: ")] + [
            line[len("*** Delete File: ") :] for line in lines if line.startswith("*** Delete File: ")
        ]

    def text_to_patch(self, text: str, orig: Dict[str, str]) -> Tuple[Patch, int]:
        lines = text.splitlines()  # preserves blank lines, no strip()
        if (
            len(lines) < 2
            or not Parser._norm(lines[0]).startswith("*** Begin Patch")
            or Parser._norm(lines[-1]) != "*** End Patch"
        ):
            raise DiffError("Invalid patch text - missing sentinels")

        parser = Parser(current_files=orig, lines=lines, index=1)
        parser.parse()
        return parser.patch, parser.fuzz

    def patch_to_commit(self, patch: Patch, orig: Dict[str, str]) -> Commit:
        commit = Commit()
        for path, action in patch.actions.items():
            if action.type is ActionType.DELETE:
                commit.changes[path] = FileChange(type=ActionType.DELETE, old_content=orig[path])
            elif action.type is ActionType.ADD:
                if action.new_file is None:
                    raise DiffError("ADD action without file content")
                commit.changes[path] = FileChange(type=ActionType.ADD, new_content=action.new_file)
            elif action.type is ActionType.UPDATE:
                new_content = _get_updated_file(orig[path], action, path)
                commit.changes[path] = FileChange(
                    type=ActionType.UPDATE,
                    old_content=orig[path],
                    new_content=new_content,
                    move_path=action.move_path,
                )
        return commit

    def apply_commit(
        self,
        commit: Commit,
        write_fn: Callable[[str, str], None],
        remove_fn: Callable[[str], None],
    ) -> None:
        for path, change in commit.changes.items():
            if change.type is ActionType.DELETE:
                remove_fn(path)
            elif change.type is ActionType.ADD:
                if change.new_content is None:
                    raise DiffError(f"ADD change for {path} has no content")
                write_fn(path, change.new_content)
            elif change.type is ActionType.UPDATE:
                if change.new_content is None:
                    raise DiffError(f"UPDATE change for {path} has no new content")
                target = change.move_path or path
                write_fn(target, change.new_content)
                if change.move_path:
                    remove_fn(path)

    async def apply_patch(self, text: str) -> str:
        """
        Based on OpenAI's reference implementation:
        https://cookbook.openai.com/examples/gpt4-1_prompting_guide#reference-implementation-apply_patchpy
        """
        if not text.startswith("*** Begin Patch"):
            raise DiffError("Patch text must start with *** Begin Patch")

        paths = self.identify_files_needed(text)

        # Preflight: if any path is blocked, short-circuit with the message
        for p in paths:
            blocked_msg = self._enforce_vibe_edit_policy(p)
            if blocked_msg:
                return blocked_msg

        # Create filesystem-aware helper functions
        def open_file_fs(path: str) -> str:
            return self.filesystem.read_text(path)

        def write_file_fs(path: str, content: str) -> None:
            # Enforce vibe-mode edit policy before writing
            msg = self._enforce_vibe_edit_policy(path)
            if msg:
                # Abort by raising to unwind; apply_patch will have preflighted, so this is a safeguard
                raise ValueError(msg)
            parent_dir = self.filesystem.get_parent(path)
            if parent_dir and not self.filesystem.exists(parent_dir):
                self.filesystem.create_directory(parent_dir)
            self.filesystem.write_text(path, content)

        def remove_file_fs(path: str) -> None:
            # Enforce vibe-mode edit policy before removing
            msg = self._enforce_vibe_edit_policy(path)
            if msg:
                raise ValueError(msg)
            if self.filesystem.exists(path):
                self.filesystem.delete(path)

        orig = self.load_files(paths, open_file_fs)

        patch, _fuzz = self.text_to_patch(text, orig)
        commit = self.patch_to_commit(patch, orig)

        self.apply_commit(commit, write_file_fs, remove_file_fs)

        return f"Applied patch:\n```\n{text}```\n"
