/**
 * Kolega Code replay player.
 *
 * Plays a recorded session back from its event log: real time, sped up, or
 * instantly, with pause and seek. Reads the same events any frontend renders and
 * folds them with the shared projection in fold.js.
 *
 * Two design points worth knowing before editing:
 *
 * 1. **Playback time is not wall time.** Events carry ``elapsed_ms`` from the
 *    session's start, but a session contains long idle gaps (a model thinking, a
 *    user away from the keyboard). Replaying those verbatim would be unwatchable,
 *    so gaps are capped, exactly as terminal recorders do. All timeline maths is
 *    in capped "playback time"; the clock shows real session time.
 *
 * 2. **Folding is incremental forward, restarted backward.** Advancing folds only
 *    the newly reached events, which keeps normal playback linear in session
 *    length. Seeking backwards cannot un-fold, so it rebuilds from the start.
 */

import { emptyState, fold } from "./fold.js";

const IDLE_CAP_MS = 2000;
const SPEEDS = [1, 2, 4, 0]; // 0 renders the remainder instantly
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
const GLYPHS = {
  user: "❯",
  assistant: "●",
  thinking: "◦",
  tool: "⏺",
  system: "·",
};
const BAR_FILLED = "█";
const BAR_EMPTY = "░";

const dom = {};
let events = [];
let manifest = {};
/** Cumulative capped playback offset per event index. */
let playbackAt = [];
let totalPlaybackMs = 0;

let state = emptyState();
let appliedCount = 0;
let position = 0;
let playing = false;
let speedIndex = 0;
let lastFrame = 0;
let renderedCount = 0;
/** Per-index snapshot of the mutable fields renderEntry draws, for change detection. */
let renderedEntries = [];
let spinnerTick = 0;

// ---------------------------------------------------------------- bootstrap ---

async function main() {
  cacheDom();
  try {
    manifest = await fetchJson("manifest.json");
    events = await fetchEvents("events.jsonl");
  } catch (error) {
    showError(
      `Could not load this recording: ${error.message}. ` +
        "A bundle must be served alongside its manifest.json and events.jsonl.",
    );
    return;
  }
  if (manifest.format && manifest.format > 1) {
    showError(`This recording was written by a newer version (format ${manifest.format}).`);
    return;
  }

  applyTheme(manifest.theme);
  buildTimeline();
  populateChrome();
  wireControls();
  seek(0);
  requestAnimationFrame(frame);
}

function cacheDom() {
  for (const id of [
    "title",
    "meta",
    "truncated",
    "transcript",
    "turns",
    "subAgents",
    "terminal",
    "gauge",
    "gaugeLabel",
    "status",
    "play",
    "speed",
    "scrub",
    "ticks",
    "clock",
    "theme",
    "error",
  ]) {
    dom[id] = document.getElementById(id);
  }
}

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  return response.json();
}

async function fetchEvents(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} returned ${response.status}`);
  const text = await response.text();
  return text
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line));
}

function showError(message) {
  if (!dom.error) return;
  dom.error.textContent = message;
  dom.error.hidden = false;
}

// ----------------------------------------------------------------- timeline ---

/** Map each event onto a playback offset, capping idle gaps. */
function buildTimeline() {
  playbackAt = [];
  let cursor = 0;
  let previous = events.length ? (events[0].elapsed_ms || 0) : 0;
  for (const event of events) {
    const elapsed = event.elapsed_ms || 0;
    cursor += Math.min(Math.max(elapsed - previous, 0), IDLE_CAP_MS);
    previous = elapsed;
    playbackAt.push(cursor);
  }
  totalPlaybackMs = cursor;
}

/** Number of events that have occurred at playback offset ``ms``. */
function countAt(ms) {
  let low = 0;
  let high = playbackAt.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if (playbackAt[mid] <= ms) low = mid + 1;
    else high = mid;
  }
  return low;
}

function applyCount(target) {
  if (target === appliedCount) return;
  if (target < appliedCount) {
    // Folding cannot run backwards, so rebuild from the beginning.
    state = emptyState();
    appliedCount = 0;
    renderedCount = 0;
    renderedEntries = [];
    dom.transcript.replaceChildren();
  }
  for (let index = appliedCount; index < target; index += 1) {
    fold(state, events[index]);
  }
  appliedCount = target;
  render();
}

function seek(ms) {
  position = Math.min(Math.max(ms, 0), totalPlaybackMs);
  applyCount(countAt(position));
  syncTransport();
}

function frame(now) {
  const delta = lastFrame ? now - lastFrame : 0;
  lastFrame = now;
  if (playing) {
    const speed = SPEEDS[speedIndex];
    if (speed === 0) {
      seek(totalPlaybackMs);
      setPlaying(false);
    } else {
      seek(position + delta * speed);
      if (position >= totalPlaybackMs) setPlaying(false);
    }
  }
  advanceSpinner();
  requestAnimationFrame(frame);
}

function setPlaying(next) {
  playing = next;
  if (playing && position >= totalPlaybackMs) {
    seek(0);
  }
  dom.play.textContent = playing ? "❙❙" : "▶";
  dom.play.setAttribute("aria-label", playing ? "Pause" : "Play");
}

// -------------------------------------------------------------------- chrome ---

function populateChrome() {
  dom.title.textContent = manifest.title || manifest.session_id || "Session replay";
  const parts = [];
  if (manifest.event_count) parts.push(`${manifest.event_count} events`);
  if (manifest.turns && manifest.turns.length) parts.push(`${manifest.turns.length} turns`);
  parts.push(formatDuration(manifest.duration_ms || 0));
  dom.meta.textContent = parts.join("  ·  ");
  // The truncation badge reflects folded state, not the manifest, so it appears
  // at the point in the replay where recording actually stopped.
  dom.truncated.hidden = true;

  for (const slug of manifest.themes || []) {
    const option = document.createElement("option");
    option.value = slug;
    option.textContent = slug.replace(/-/g, " ");
    dom.theme.append(option);
  }
  dom.theme.value = manifest.theme || "kolega-dark";

  dom.ticks.replaceChildren();
  for (const turn of manifest.turns || []) {
    const offset = playbackOffsetForElapsed(turn.elapsed_ms || 0);
    const tick = document.createElement("span");
    tick.className = "kc-tick";
    tick.style.left = `${totalPlaybackMs ? (offset / totalPlaybackMs) * 100 : 0}%`;
    tick.title = turn.user_text || turn.turn_id;
    dom.ticks.append(tick);
  }
  renderTurns();
}

/** Nearest playback offset for a session-time value, used for turn seeking. */
function playbackOffsetForElapsed(elapsedMs) {
  for (let index = 0; index < events.length; index += 1) {
    if ((events[index].elapsed_ms || 0) >= elapsedMs) return playbackAt[index];
  }
  return totalPlaybackMs;
}

function renderTurns() {
  const turns = manifest.turns || [];
  if (!turns.length) {
    dom.turns.innerHTML = '<p class="kc-empty">No turns recorded.</p>';
    return;
  }
  dom.turns.replaceChildren();
  turns.forEach((turn, index) => {
    const button = document.createElement("button");
    button.className = "kc-turn";
    button.type = "button";
    const mark = document.createElement("span");
    mark.className = "kc-turn-mark";
    mark.dataset.status = turn.status || "open";
    mark.textContent = turn.status === "failed" ? "✗" : turn.status === "completed" ? "✓" : "○";
    const text = document.createElement("span");
    text.className = "kc-turn-text";
    text.textContent = turn.user_text || `Turn ${index + 1}`;
    // The rail is narrow and long prompts are elided, so keep the full text
    // reachable on hover rather than only in the accessibility tree.
    button.title = turn.user_text || `Turn ${index + 1}`;
    const time = document.createElement("span");
    time.className = "kc-turn-time";
    time.textContent = formatDuration(turn.elapsed_ms || 0);
    button.append(mark, text, time);
    button.addEventListener("click", () => {
      setPlaying(false);
      seek(playbackOffsetForElapsed(turn.elapsed_ms || 0));
    });
    dom.turns.append(button);
  });
}

function wireControls() {
  dom.play.addEventListener("click", () => setPlaying(!playing));
  dom.speed.addEventListener("click", () => {
    speedIndex = (speedIndex + 1) % SPEEDS.length;
    dom.speed.textContent = SPEEDS[speedIndex] === 0 ? "max" : `${SPEEDS[speedIndex]}x`;
  });
  dom.scrub.addEventListener("input", () => {
    setPlaying(false);
    seek((Number(dom.scrub.value) / 1000) * totalPlaybackMs);
  });
  dom.theme.addEventListener("change", () => applyTheme(dom.theme.value));
  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
    if (event.code === "Space") {
      event.preventDefault();
      setPlaying(!playing);
    } else if (event.code === "ArrowRight") {
      setPlaying(false);
      seek(position + 5000);
    } else if (event.code === "ArrowLeft") {
      setPlaying(false);
      seek(position - 5000);
    } else if (event.code === "Home") {
      seek(0);
    } else if (event.code === "End") {
      seek(totalPlaybackMs);
    }
  });
}

function applyTheme(slug) {
  if (slug) document.documentElement.dataset.theme = slug;
}

function syncTransport() {
  const ratio = totalPlaybackMs ? position / totalPlaybackMs : 0;
  dom.scrub.value = String(Math.round(ratio * 1000));
  dom.scrub.style.setProperty("--kc-progress", `${ratio * 100}%`);
  const sessionMs = appliedCount ? events[appliedCount - 1].elapsed_ms || 0 : 0;
  dom.clock.textContent = `${formatDuration(sessionMs)} / ${formatDuration(manifest.duration_ms || 0)}`;
  const current = (manifest.turns || []).reduce(
    (best, turn, index) => ((turn.elapsed_ms || 0) <= sessionMs ? index : best),
    -1,
  );
  Array.from(dom.turns.children).forEach((child, index) => {
    child.setAttribute("aria-current", String(index === current));
  });
}

// -------------------------------------------------------------------- render ---

function render() {
  renderTranscript();
  renderRail();
  syncTransport();
}

/**
 * Snapshot the fields renderEntry draws that the fold can still mutate.
 *
 * Every fold mutation installs a fresh value or reference (`text` is rebuilt by
 * concatenation, `artifacts` and the edit preview are reassigned), so comparing
 * a snapshot is exact and costs a handful of identity checks per entry.
 */
function entrySnapshot(item) {
  return {
    text: item.text,
    complete: item.complete,
    status: item.status,
    toolName: item.tool_name || item.toolName,
    artifacts: item.artifacts,
    preview: item.editPreview || item.edit_preview,
  };
}

function entryChanged(previous, item) {
  return (
    previous === undefined ||
    previous.text !== item.text ||
    previous.complete !== item.complete ||
    previous.status !== item.status ||
    previous.toolName !== (item.tool_name || item.toolName) ||
    previous.artifacts !== item.artifacts ||
    previous.preview !== (item.editPreview || item.edit_preview)
  );
}

function renderTranscript() {
  // An entry that is no longer the newest can still change. Streaming segments
  // are keyed by uuid, so reasoning keeps accumulating after the assistant prose
  // that interrupted it was appended — every turn interleaves them — and a tool
  // entry resolves long after later entries exist. Re-rendering only the newest
  // entry left those frozen mid-stream, spinner and all.
  const items = state.conversation;
  if (renderedCount > items.length) {
    dom.transcript.replaceChildren();
    renderedCount = 0;
    renderedEntries = [];
  }
  for (let index = 0; index < renderedCount; index += 1) {
    const item = items[index];
    if (!entryChanged(renderedEntries[index], item)) continue;
    dom.transcript.children[index]?.replaceWith(renderEntry(item));
    renderedEntries[index] = entrySnapshot(item);
  }
  for (let index = renderedCount; index < items.length; index += 1) {
    const item = items[index];
    dom.transcript.append(renderEntry(item));
    renderedEntries[index] = entrySnapshot(item);
  }
  renderedCount = items.length;
  const atEnd = dom.transcript.parentElement;
  if (atEnd) atEnd.scrollTop = atEnd.scrollHeight;
}

function renderEntry(item) {
  const row = document.createElement("div");
  row.className = "kc-entry";
  row.dataset.kind = item.kind;
  if (item.tone) row.dataset.tone = item.tone;

  const glyph = document.createElement("span");
  glyph.className = "kc-glyph";
  glyph.textContent = GLYPHS[item.kind] || GLYPHS.system;
  row.append(glyph);

  const body = document.createElement("div");
  body.className = "kc-body";
  if (item.kind === "tool") {
    body.append(renderToolBody(item));
  } else {
    body.textContent = item.text;
    if (!item.complete) body.append(spinnerNode());
  }
  row.append(body);
  return row;
}

function renderToolBody(item) {
  const wrapper = document.createDocumentFragment();
  const head = document.createElement("div");
  head.className = "kc-tool-head";
  const name = document.createElement("span");
  name.className = "kc-tool-name";
  name.textContent = item.tool_name || item.toolName || "tool";
  head.append(name);
  const status = document.createElement("span");
  status.className = "kc-tool-status";
  status.dataset.status = item.status || "running";
  status.textContent = item.status || "running";
  head.append(status);
  if (item.status === "running") head.append(spinnerNode());
  wrapper.append(head);

  if (item.text) {
    const output = document.createElement("div");
    output.className = "kc-tool-output";
    output.textContent = item.text;
    wrapper.append(output);
  }
  const preview = item.editPreview || item.edit_preview;
  if (preview && preview.diff) {
    wrapper.append(renderDiff(String(preview.diff)));
  }
  for (const ref of item.artifacts || []) {
    const badge = document.createElement("span");
    badge.className = "kc-artifact";
    const size = ref.chars ? `${ref.chars.toLocaleString()} chars` : `${ref.bytes} bytes`;
    badge.textContent = `artifact · ${size}`;
    wrapper.append(badge);
  }
  return wrapper;
}

function renderDiff(diff) {
  const block = document.createElement("div");
  block.className = "kc-diff";
  for (const line of diff.split("\n")) {
    const row = document.createElement("div");
    if (line.startsWith("+")) row.className = "kc-diff-line-add";
    else if (line.startsWith("-")) row.className = "kc-diff-line-del";
    row.textContent = line;
    block.append(row);
  }
  return block;
}

function spinnerNode() {
  const span = document.createElement("span");
  span.className = "kc-spinner";
  span.textContent = SPINNER_FRAMES[spinnerTick % SPINNER_FRAMES.length];
  return span;
}

function advanceSpinner() {
  const next = Math.floor(performance.now() / 250);
  if (next === spinnerTick) return;
  spinnerTick = next;
  const frameGlyph = SPINNER_FRAMES[spinnerTick % SPINNER_FRAMES.length];
  for (const node of document.querySelectorAll(".kc-spinner")) {
    node.textContent = frameGlyph;
  }
}

function renderRail() {
  renderGauge();
  renderSubAgents();
  dom.terminal.textContent = state.terminal || "";
  if (!state.terminal) dom.terminal.innerHTML = '<span class="kc-empty">No terminal output yet.</span>';
  dom.status.textContent = state.status || "";
  dom.truncated.hidden = !state.recordingTruncated;
}

function renderGauge() {
  const context = state.context;
  if (!context || !context.max_tokens) {
    dom.gauge.textContent = "";
    dom.gaugeLabel.innerHTML = '<span class="kc-empty">No context reading yet.</span>';
    return;
  }
  // The token is a CSS length (e.g. "18ch"), so it must be parsed as a leading
  // integer rather than coerced: Number("18ch") is NaN, and "█".repeat(NaN)
  // silently yields an empty string instead of a gauge.
  const declared = getComputedStyle(document.documentElement).getPropertyValue("--kc-context-bar-width");
  const parsed = Number.parseInt(declared, 10);
  const width = Number.isFinite(parsed) && parsed > 0 ? parsed : 18;
  const ratio = Math.min(Math.max(context.usage_percentage / 100, 0), 1);
  const filled = Math.round(ratio * width);
  dom.gauge.textContent = BAR_FILLED.repeat(filled) + BAR_EMPTY.repeat(Math.max(0, width - filled));
  dom.gauge.dataset.alert = context.alert_level || "ok";
  dom.gaugeLabel.textContent =
    `${context.input_tokens.toLocaleString()} / ${context.max_tokens.toLocaleString()} tokens` +
    ` · ${context.usage_percentage.toFixed(1)}%`;
}

function renderSubAgents() {
  const entries = Array.from(state.subAgents.values());
  if (!entries.length) {
    dom.subAgents.innerHTML = '<p class="kc-empty">No sub-agents dispatched.</p>';
    return;
  }
  dom.subAgents.replaceChildren();
  for (const activity of entries) {
    const row = document.createElement("div");
    row.className = "kc-sub-agent";
    const name = document.createElement("div");
    name.className = "kc-sub-agent-name";
    name.textContent = `◆ ${activity.name}`;
    const task = document.createElement("div");
    task.className = "kc-sub-agent-task";
    task.textContent = activity.task || activity.lastText || "";
    row.append(name, task);
    dom.subAgents.append(row);
  }
}

function formatDuration(ms) {
  const total = Math.max(0, Math.round(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    return `${hours}h${String(minutes % 60).padStart(2, "0")}m`;
  }
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

main();
