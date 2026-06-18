"""Load and run an authored Python orchestration script.

The script runs in-process in a *curated namespace*: a restricted ``__builtins__``
(no ``import``/``open``/``eval``/``exec``, no ``time``/``random`` so resume stays
deterministic) plus the injected primitives (``agent``/``parallel``/``pipeline``/
``phase``/``log``/``workflow``) and the ``args``/``budget`` globals.

This is a *soft* sandbox, not a security boundary — the agent already runs
arbitrary shell via its command tools. The restrictions exist to keep scripts
deterministic (a hard requirement for resume) and to fail loudly on the easy
mistakes rather than to contain a hostile script.
"""

from __future__ import annotations

import ast
import builtins
from typing import Any, Dict

from .errors import WorkflowScriptError

WRAPPER_NAME = "__workflow_main__"

# Builtins the script may use. Deliberately omits __import__, open, eval, exec,
# compile, input, and the time/random surface. __build_class__ is included so a
# script that defines a helper class doesn't crash.
_SAFE_BUILTIN_NAMES = (
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes", "callable",
    "chr", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
    "getattr", "hasattr", "hash", "hex", "int", "isinstance", "issubclass", "iter",
    "len", "list", "map", "max", "min", "next", "oct", "ord", "pow", "range",
    "repr", "reversed", "round", "set", "setattr", "slice", "sorted", "str", "sum",
    "tuple", "type", "zip",
    # exception types scripts may raise/catch
    "Exception", "ValueError", "KeyError", "IndexError", "TypeError", "RuntimeError",
    "StopIteration", "StopAsyncIteration", "ArithmeticError", "ZeroDivisionError",
    "AttributeError", "NotImplementedError", "AssertionError",
    # class definition support
    "__build_class__",
)


def safe_builtins() -> Dict[str, Any]:
    """A restricted ``__builtins__`` mapping for workflow script execution."""
    table: Dict[str, Any] = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(builtins, name)}
    table["True"] = True
    table["False"] = False
    table["None"] = None
    return table


def extract_meta(source: str) -> Dict[str, Any]:
    """Parse and validate the module-level ``meta = {...}`` literal.

    Mirrors the ultracode rule that ``meta`` must be a pure literal (no variables,
    calls, or interpolation) so it can be read without executing the script.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise WorkflowScriptError(f"workflow script has a syntax error: {exc}") from exc

    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "meta":
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError) as exc:
                        raise WorkflowScriptError(
                            "`meta` must be a pure literal dict (no variables, calls, or f-strings)."
                        ) from exc
                    if not isinstance(value, dict):
                        raise WorkflowScriptError("`meta` must be a dict literal.")
                    if not value.get("name") or not value.get("description"):
                        raise WorkflowScriptError("`meta` must include non-empty `name` and `description`.")
                    return value

    raise WorkflowScriptError(
        "workflow script must define a top-level `meta = {\"name\": ..., \"description\": ...}` literal."
    )


def _wrap_as_coroutine(source: str) -> ast.Module:
    """Wrap the whole script body in an ``async def`` so top-level ``await`` and
    ``return`` are legal. Done at the AST level (not by text indentation) so
    multi-line string literals are never corrupted.
    """
    tree = ast.parse(source)
    func = ast.AsyncFunctionDef(
        name=WRAPPER_NAME,
        args=ast.arguments(
            posonlyargs=[], args=[], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]
        ),
        body=tree.body or [ast.Pass()],
        decorator_list=[],
        returns=None,
        type_comment=None,
    )
    module = ast.Module(body=[func], type_ignores=[])
    ast.fix_missing_locations(module)
    return module


async def run_script(source: str, namespace: Dict[str, Any]) -> Any:
    """Compile and run ``source`` in ``namespace``; return the script's value.

    ``namespace`` must already carry ``__builtins__`` and the injected globals.
    """
    try:
        module = _wrap_as_coroutine(source)
        code = compile(module, "<workflow>", "exec")
    except SyntaxError as exc:
        raise WorkflowScriptError(f"workflow script has a syntax error: {exc}") from exc

    exec(code, namespace)  # noqa: S102 - curated namespace; see module docstring
    main = namespace[WRAPPER_NAME]
    return await main()
