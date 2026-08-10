Run one step of code in a persistent kernel: state (imports, variables, functions) persists across calls, tools, and sub-agents in this session — each call is one logical step.

language="py" runs Python in kolega-code's managed environment (see python_info()): numpy, pandas, matplotlib, pillow preinstalled; pip_install("scipy") adds more. Use tool.exec_command for the project's own venv; language="js" runs JavaScript on Bun or Node (>= 18) when available.

Work incrementally: imports → define → test → use, one cell each; re-run setup only after reset or a kernel crash — never re-import or re-declare prior top-level names.

Both kernels can call back into your own tools over a loopback bridge. Bridge calls count as real tool calls — permissions and hooks apply; list_tools() shows available names. tool.* results arrive in the tool's model-facing format (tool.read wraps content in a markdown code fence); use read()/write() for raw file bytes.

Python prelude (sync; pass kwargs): display(value) (dict/list → JSON, Figure → image), print(value), read(path, offset=1, limit=None), write(path, content), env(key=None, value=None), tool.<name>(args_dict, **kwargs) — any session tool, parallel([lambda: ...]) — thunks, results in order, pip_install(*pkgs). Top-level await works; do NOT call asyncio.run() inside a coroutine cell.

JavaScript prelude (async; ONE trailing object literal, never positional args): same helpers as Python, camelCase (tool.<name>({...}), listTools(), parallel, npm_install, setGlobal). Top-level await works; declarations inside await-wrapped cells do NOT persist — use setGlobal(name, value) for cross-cell state. Redeclaring an existing top-level name errors: assign without redeclaring, or pass reset=true.

Returns:
    The cell's stdout/stderr, the last expression's value (REPL echo), display() outputs (images when the model supports vision), log()/phase() status lines, and any error with its traceback.
