/**
 * Kolega Code eval-kernel prelude (JavaScript).
 *
 * Executed once in the kernel's persistent vm context at startup (sent by the
 * host as a silent init cell). Defines the model-facing in-kernel API:
 * display(), read()/write()/env(), the tool.<name>(args) bridge proxy,
 * listTools(), parallel(), npm_install(), setGlobal(), log()/phase().
 *
 * Style: async; helpers that hit the bridge return promises — use top-level
 * await. The tool proxy takes ONE trailing object literal, never positional.
 */

const __kcFs = require("node:fs");
const __kcPath = require("node:path");
const __kcCp = require("node:child_process");

async function __kolegaBridgeCall(name, args) {
  const base = process.env.KOLEGA_TOOL_BRIDGE_URL;
  const token = process.env.KOLEGA_TOOL_BRIDGE_TOKEN;
  const session = process.env.KOLEGA_TOOL_BRIDGE_SESSION;
  if (!base || !token || !session) throw new Error("tool bridge is unavailable in this kernel");
  const run = globalThis.__kolega_run_id__ || "";
  const response = await fetch(base.replace(/\/+$/, "") + "/v1/tool", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ session, run, name, args }),
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(`bridge call '${name}' failed: non-JSON response (HTTP ${response.status})`);
  }
  if (!data || data.ok !== true) {
    throw new Error((data && data.error) || `bridge call '${name}' failed`);
  }
  const value = data.value;
  if (
    value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    Object.prototype.hasOwnProperty.call(value, "text") &&
    Object.keys(value).every((key) => key === "text" || key === "images")
  ) {
    return value.images && value.images.length ? value : value.text;
  }
  return value;
}

const tool = new Proxy(
  {},
  {
    get: (_target, name) => {
      if (typeof name !== "string" || name.startsWith("_")) return undefined;
      return (args) => __kolegaBridgeCall(name, args ?? {});
    },
  },
);

async function listTools() {
  return __kolegaBridgeCall("__list_tools__", {});
}

function display(value) {
  __kolega_display__(value);
}

function log(message) {
  __kolega_status__("log", { message: String(message) });
}

function phase(title) {
  __kolega_status__("phase", { title: String(title) });
}

function read(path, offset = 1, limit = undefined) {
  let data = __kcFs.readFileSync(String(path), "utf-8");
  if (offset > 1 || limit !== undefined) {
    const lines = data.split(/(?<=\n)/);
    const start = Math.max(0, offset - 1);
    const end = limit ? start + limit : lines.length;
    data = lines.slice(start, end).join("");
  }
  __kolega_status__("read", { path: String(path), chars: data.length });
  return data;
}

function write(path, content) {
  const target = __kcPath.resolve(String(path));
  __kcFs.mkdirSync(__kcPath.dirname(target), { recursive: true });
  __kcFs.writeFileSync(target, String(content), "utf-8");
  __kolega_status__("write", { path: target, chars: String(content).length });
  return target;
}

function env(key = undefined, value = undefined) {
  if (key === undefined) return { ...process.env };
  if (value !== undefined) {
    process.env[String(key)] = String(value);
    return value;
  }
  return process.env[String(key)];
}

function setGlobal(name, value) {
  globalThis[name] = value;
  return value;
}

function parallel(thunks) {
  return Promise.all(thunks.map((thunk) => thunk()));
}

function npm_install(...packages) {
  if (!packages.length) throw new Error("npm_install(...) needs at least one package name");
  const cmd = JSON.parse(process.env.KOLEGA_EVAL_NPM_INSTALL_CMD || "[]");
  if (!cmd.length) throw new Error("npm_install is unavailable in this kernel");
  const result = __kcCp.spawnSync(cmd[0], [...cmd.slice(1), ...packages.map(String)], {
    encoding: "utf-8",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`npm_install failed (exit ${result.status}):\n${String(result.stderr).slice(-2000)}`);
  }
  __kolega_status__("npm_install", { packages });
  return String(result.stdout).slice(-2000);
}
