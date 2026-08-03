/**
 * Kolega Code eval-kernel runner (JavaScript, Node >= 18 or Bun).
 *
 * Spawned as `<runtime> runner.js` by the host. Speaks NDJSON over
 * stdin/stdout, one JSON frame per line (see protocol.py in the same package).
 *
 * Cells run in a single persistent vm context: top-level let/const/function/
 * class declarations survive across cells via the context's global lexical
 * environment. Top-level await is supported by retrying the cell wrapped in an
 * async IIFE (declarations inside wrapped cells do NOT persist — the prelude
 * exposes setGlobal() for that case). The V8 script completion value of
 * unwrapped cells is emitted as the result frame.
 *
 * Input uses a blocking fs.readSync(0) line reader rather than stream events:
 * Bun (as of 1.2.x) does not deliver piped-stdin `data` events until more data
 * or EOF arrives, which deadlocks the request/response protocol. Blocking while
 * idle is safe because the host serializes frames — it sends the next cell only
 * after reading the current cell's `done` frame — so at most one frame is ever
 * in flight and the event loop is free for the duration of each cell.
 */
"use strict";

const vm = require("node:vm");
const util = require("node:util");
const path = require("node:path");
const fs = require("node:fs");
const { createRequire } = require("node:module");

const JS_HOME = process.env.KOLEGA_EVAL_JS_HOME || process.cwd();
const CURRENT = { id: 0, run: "" };

function emit(frame) {
  process.stdout.write(JSON.stringify(frame) + "\n");
}

function formatArgs(args) {
  return util.format(...args) + "\n";
}

const consoleShim = {
  log: (...args) => emit({ type: "stdout", id: CURRENT.id, data: formatArgs(args) }),
  info: (...args) => emit({ type: "stdout", id: CURRENT.id, data: formatArgs(args) }),
  debug: (...args) => emit({ type: "stdout", id: CURRENT.id, data: formatArgs(args) }),
  warn: (...args) => emit({ type: "stderr", id: CURRENT.id, data: formatArgs(args) }),
  error: (...args) => emit({ type: "stderr", id: CURRENT.id, data: formatArgs(args) }),
};

function safeStringify(value) {
  const seen = new WeakSet();
  return JSON.stringify(value, (key, val) => {
    if (typeof val === "bigint") return val.toString();
    if (typeof val === "function") return `[Function ${val.name || "anonymous"}]`;
    if (typeof val === "object" && val !== null) {
      if (seen.has(val)) return "[Circular]";
      seen.add(val);
    }
    return val;
  });
}

function toBundle(value) {
  const bundle = {};
  if (typeof value === "string") {
    bundle["text/plain"] = value;
    return bundle;
  }
  try {
    bundle["application/json"] = JSON.parse(safeStringify(value));
  } catch {
    /* not JSON-representable; text/plain below carries it */
  }
  let plain;
  try {
    plain = util.inspect(value, { depth: 4, maxStringLength: 4000, breakLength: 120 });
  } catch {
    plain = String(value);
  }
  bundle["text/plain"] = plain;
  return bundle;
}

function emitError(id, err) {
  const stack = typeof err?.stack === "string" ? err.stack.split("\n") : [String(err)];
  emit({
    type: "error",
    id,
    ename: (err && err.constructor && err.constructor.name) || "Error",
    evalue: (err && err.message) || String(err),
    traceback: stack,
  });
}

// Errors raised inside the vm context belong to that realm, so `instanceof
// SyntaxError` fails across realms (this is how Bun's parse errors arrive).
function isSyntaxError(err) {
  return (
    err instanceof SyntaxError ||
    Boolean(err && (err.name === "SyntaxError" || (err.constructor && err.constructor.name === "SyntaxError")))
  );
}

function isRedeclaration(err) {
  return isSyntaxError(err) && /already been declared|duplicate variable/i.test(err.message || "");
}

// Parse-level syntax failures (vs. runtime errors from code the cell itself
// evals). Node reports them at `new vm.Script(...)`; Bun parses lazily, so they
// surface from runInContext with an "at <parse>" stack frame instead.
function isParseLevelSyntaxError(err) {
  if (!isSyntaxError(err)) return false;
  if (/await is only valid|await outside|Unexpected identifier|Unexpected token/i.test(err.message || "")) {
    return true;
  }
  return String(err.stack || "").includes("at <parse>");
}

function augmentSyntaxError(err) {
  if (isRedeclaration(err)) {
    err.message +=
      " (this name already exists in the persistent kernel — assign to it without " +
      "redeclaring, pick a new name, or re-run with reset=true)";
  }
  return err;
}

const sandbox = {
  console: consoleShim,
  fetch: globalThis.fetch,
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  queueMicrotask,
  structuredClone: globalThis.structuredClone,
  TextEncoder,
  TextDecoder,
  URL,
  URLSearchParams,
  Buffer,
  crypto: globalThis.crypto,
  atob: globalThis.atob,
  btoa: globalThis.btoa,
  process: { env: process.env, cwd: () => process.cwd() },
  require: createRequire(path.join(JS_HOME, "__kernel__.js")),
  // Bridged by the prelude; defined here so cells can call them directly too.
  __kolega_display__: (value) => emit({ type: "display", id: CURRENT.id, bundle: toBundle(value) }),
  __kolega_status__: (op, data) =>
    emit({ type: "status", id: CURRENT.id, event: { op, ...(data || {}) } }),
};
sandbox.globalThis = sandbox;
const context = vm.createContext(sandbox);

async function runCell(code, id, silent) {
  const filename = `<cell-${id}>`;
  // Top-level await (or return) needs an async wrapper. Declarations inside a
  // wrapped cell do NOT persist across cells — the prelude's setGlobal() covers
  // that case, and the tool description says so.
  const wrappedSource = `(async () => {\n${code}\n})()`;

  let script = null;
  try {
    script = new vm.Script(code, { filename });
  } catch (err) {
    if (isRedeclaration(err)) throw augmentSyntaxError(err);
    if (isSyntaxError(err)) {
      try {
        script = new vm.Script(wrappedSource, { filename });
      } catch {
        throw augmentSyntaxError(err);
      }
    } else {
      throw err;
    }
  }

  let value;
  try {
    value = script.runInContext(context, { breakOnSigint: true });
  } catch (err) {
    if (isRedeclaration(err)) throw augmentSyntaxError(err);
    if (isParseLevelSyntaxError(err)) {
      // Bun parses lazily at runInContext time, so a top-level await only
      // surfaces here. Retry once with the async wrapper.
      let wrappedScript;
      try {
        wrappedScript = new vm.Script(wrappedSource, { filename });
      } catch {
        throw augmentSyntaxError(err);
      }
      try {
        value = wrappedScript.runInContext(context, { breakOnSigint: true });
      } catch (wrappedErr) {
        if (isSyntaxError(wrappedErr)) throw augmentSyntaxError(err);
        throw wrappedErr;
      }
    } else {
      throw err;
    }
  }
  if (value && typeof value.then === "function") {
    value = await value;
  }
  return { value, emitResult: !silent };
}

// The cell currently executing, so SIGINT can finish it early when it is stuck
// in a pending await (sync code is interrupted via breakOnSigint instead). The
// handler only signals; handleExec owns all frame emission so `done` is sent
// exactly once and the main loop resumes reading frames immediately.
let activeCell = null;

process.on("SIGINT", () => {
  if (activeCell && !activeCell.settled) {
    activeCell.settled = true;
    activeCell.interrupt();
  }
});

async function handleExec(msg) {
  const id = Number(msg.id) || 0;
  CURRENT.id = id;
  CURRENT.run = typeof msg.run === "string" ? msg.run : "";
  sandbox.__kolega_run_id__ = CURRENT.run;
  if (typeof msg.cwd === "string" && msg.cwd) {
    try {
      process.chdir(msg.cwd);
    } catch (err) {
      emit({ type: "stderr", id, data: `[kernel] chdir failed: ${err.message}\n` });
    }
  }
  emit({ type: "started", id });
  const cell = { id, settled: false, interrupt: () => {} };
  activeCell = cell;
  const interrupted = new Promise((resolve) => {
    cell.interrupt = () => resolve({ kind: "interrupted" });
  });
  // Guard both outcomes so a cell finished early by SIGINT cannot produce an
  // unhandled rejection when its promise settles later.
  const work = runCell(String(msg.code ?? ""), id, Boolean(msg.silent)).then(
    (outcome) => ({ kind: "ok", outcome }),
    (err) => ({ kind: "error", err }),
  );
  const verdict = await Promise.race([work, interrupted]);
  cell.settled = true;
  activeCell = null;

  if (verdict.kind === "interrupted") {
    emitError(id, new Error("cell interrupted (SIGINT)"));
    emit({ type: "done", id, status: "error" });
    return;
  }
  if (verdict.kind === "error") {
    emitError(id, verdict.err);
    emit({ type: "done", id, status: "error" });
    return;
  }
  const { value, emitResult } = verdict.outcome;
  if (emitResult && value !== undefined) {
    emit({ type: "result", id, bundle: toBundle(value) });
  }
  emit({ type: "done", id, status: "ok" });
}

/**
 * Read one newline-terminated frame from fd 0, blocking while idle. Returns
 * null on EOF. Bytes after the first newline in a read chunk are dropped,
 * which the host's strict request/response pacing makes unreachable.
 */
function readLineSync() {
  const chunk = Buffer.alloc(65536);
  let acc = "";
  while (true) {
    let n;
    try {
      n = fs.readSync(0, chunk, 0, chunk.length, null);
    } catch (err) {
      if (err && (err.code === "EAGAIN" || err.code === "EINTR")) continue;
      throw err;
    }
    if (n === 0) return acc.length > 0 ? acc : null;
    acc += chunk.toString("utf8", 0, n);
    const newline = acc.indexOf("\n");
    if (newline >= 0) return acc.slice(0, newline);
  }
}

async function main() {
  while (true) {
    const line = readLineSync();
    if (line === null) break;
    const trimmed = line.trim();
    if (!trimmed) continue;
    let msg;
    try {
      msg = JSON.parse(trimmed);
    } catch {
      continue;
    }
    if (!msg || typeof msg !== "object") continue;
    if (msg.type === "exit") break;
    if (msg.type === "exec") {
      // The host serializes cells per kernel; no re-entry guard needed here.
      await handleExec(msg);
    }
  }
}

main()
  .then(() => {
    // Exit explicitly: an interrupted cell can leave timers or sockets in the
    // event loop that would otherwise keep the process alive after the host
    // asked it to exit. Flush stdout first so no result frame is truncated.
    process.stdout.write("", () => process.exit(0));
  })
  .catch((err) => {
    emitError(CURRENT.id, err);
    process.exit(1);
  });
