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
 *
 * 3. **A recording may still be growing.** When the manifest names a stream, the
 *    player follows it over a WebSocket and the timeline extends as events
 *    arrive. Following the live edge is a mode, not a position: the viewer can
 *    scrub back to look at something without being yanked forward again, and
 *    returns with the LIVE button. A static bundle has no stream key, which is
 *    what makes an exported replay strictly a replay.
 *
 * 4. **The same player also runs from a single inlined file.** A folder of ten
 *    files cannot be opened from `file://` — browsers block module imports and
 *    `fetch` on that origin, so a recipient who double-clicks gets a blank page.
 *    `share export` therefore embeds the assets, manifest and gzipped events in
 *    one HTML document. When `globalThis.__KC_REPLAY__` is present the player
 *    reads it instead of fetching, so there is one player rather than two.
 */

import { emptyState, fold } from "./fold.js";

const IDLE_CAP_MS = 2000;
//: Backoff bounds for reattaching after the stream drops.
const RECONNECT_MIN_MS = 500;
const RECONNECT_MAX_MS = 10000;
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
/** Running total of capped playback time, so the timeline can be extended. */
let timelineCursor = 0;
/** Open socket to a live session, if this recording has one. */
let socket = null;
/** Highest seq applied, so a reconnect resumes without a gap or a duplicate. */
let liveSeq = 0;
/** Whether new events should pull the view forward to the live edge. */
let following = false;
/** Whether this session has actually shown signs of life, so the badge is honest. */
let liveActive = false;
let reconnectDelay = RECONNECT_MIN_MS;
let reconnectTimer = null;
let speedIndex = 0;
let lastFrame = 0;
let renderedCount = 0;
/** Per-index snapshot of the mutable fields renderEntry draws, for change detection. */
let renderedEntries = [];
/** Cached chronological merge of the main transcript and every sub-agent trajectory. */
let mergedRows = null;
let spinnerTick = 0;

// ---------------------------------------------------------------- bootstrap ---

async function main() {
  cacheDom();
  try {
    // A single-file export injects everything ahead of this script; a served
    // player or a directory bundle fetches it alongside.
    const inlined = globalThis.__KC_REPLAY__;
    manifest = inlined ? inlined.manifest : await fetchJson("manifest.json");
    events = inlined ? await decodeInlinedEvents(inlined.events) : await fetchEvents("events.jsonl");
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
  liveSeq = manifest.last_seq || (events.length ? events[events.length - 1].seq || 0 : 0);
  buildTimeline();
  populateChrome();
  wireControls();
  // A running session opens at its live edge: someone who follows a share link
  // wants what is happening now, not the beginning of a session already in
  // progress. A finished recording still opens at the start.
  liveActive = manifest.status === "open";
  if (manifest.stream && liveActive) {
    seek(totalPlaybackMs);
    following = true;
  } else {
    seek(0);
  }
  if (manifest.stream) attachStream();
  renderLive();
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
    "live",
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
  return parseEventLines(await response.text());
}

function parseEventLines(text) {
  return text
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => JSON.parse(line));
}

/**
 * Decode the events embedded in a single-file export.
 *
 * They are gzipped before base64 because the log is the bulk of the document and
 * compresses about eight-fold, which is the difference between a file you can
 * email and one you cannot.
 */
async function decodeInlinedEvents(payload) {
  if (Array.isArray(payload)) return payload;
  const binary = atob(payload);
  const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
  if (typeof DecompressionStream !== "function") {
    throw new Error("this browser cannot decompress an embedded recording (needs DecompressionStream)");
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return parseEventLines(await new Response(stream).text());
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
  timelineCursor = 0;
  extendTimeline(0);
}

/** Extend the timeline over events appended since ``from``. */
function extendTimeline(from) {
  for (let index = from; index < events.length; index += 1) {
    const elapsed = events[index].elapsed_ms || 0;
    const previous = index === 0 ? elapsed : events[index - 1].elapsed_ms || 0;
    timelineCursor += Math.min(Math.max(elapsed - previous, 0), IDLE_CAP_MS);
    playbackAt.push(timelineCursor);
  }
  totalPlaybackMs = timelineCursor;
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
    mergedRows = null;
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

// --------------------------------------------------------------------- live ---

/**
 * Follow a running session.
 *
 * The socket replays everything after ``last_seq`` and then stays open, so the
 * one-shot event log this page already fetched joins up with live appends
 * without a gap. Seq is the join: the server orders on it, and dropping anything
 * we have already applied makes a reconnect idempotent.
 */
function attachStream() {
  if (!manifest.stream) return;
  const url = new URL(manifest.stream, location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.searchParams.set("from_seq", String(liveSeq + 1));
  // The cookie the server set covers same-origin subresources, but a link that
  // was opened with a token in the query is the only proof this tab has.
  const token = new URLSearchParams(location.search).get("token");
  if (token) url.searchParams.set("token", token);

  socket = new WebSocket(url);
  socket.addEventListener("open", () => {
    reconnectDelay = RECONNECT_MIN_MS;
    // Attaching is not a reason to move the viewer. Whether to sit at the live
    // edge was decided when the page opened, or by the viewer since; a reconnect
    // restores that intent rather than overriding it.
    renderLive();
    if (following) seek(totalPlaybackMs);
  });
  socket.addEventListener("message", (message) => {
    let event = null;
    try {
      event = JSON.parse(message.data);
    } catch {
      return; // A malformed frame must not take down a live view.
    }
    absorbLiveEvent(event);
  });
  socket.addEventListener("close", scheduleReconnect);
  socket.addEventListener("error", () => socket && socket.close());
}

function scheduleReconnect() {
  socket = null;
  renderLive();
  if (reconnectTimer !== null) return;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    attachStream();
  }, reconnectDelay);
  reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX_MS);
}

function absorbLiveEvent(event) {
  const seq = event.seq ?? null;
  if (seq !== null) {
    if (seq <= liveSeq) return; // already applied; a resumed socket overlaps
    liveSeq = seq;
  }
  events.push(event);
  liveActive = true; // something arrived over the socket: it is live after all
  extendTimeline(events.length - 1);
  absorbLiveChrome(event);
  // The chrome describes the whole recording, which just got longer, so it is
  // refreshed whether or not the viewer is watching the edge. Tick positions are
  // a fraction of the total and therefore move on every append.
  renderMeta();
  renderTicks();
  renderTurns();
  if (following) seek(totalPlaybackMs);
  else syncTransport();
}

/** Keep the header, tick marks, and turn rail current as the session grows. */
function absorbLiveChrome(event) {
  manifest.event_count = (manifest.event_count || 0) + 1;
  manifest.duration_ms = Math.max(manifest.duration_ms || 0, event.elapsed_ms || 0);
  if (event.sub_agent_info) return; // delegated turns are not session turns
  const content = event.content || {};
  const turnId = String(content.turn_id || "");
  if (event.event_type === "turn_started") {
    manifest.turns = manifest.turns || [];
    manifest.turns.push({
      turn_id: turnId,
      user_text: String(content.user_text || ""),
      elapsed_ms: event.elapsed_ms || 0,
      seq: event.seq ?? null,
      status: "open",
      ended_ms: null,
    });
    return;
  }
  if (event.event_type !== "turn_ended") return;
  for (let index = (manifest.turns || []).length - 1; index >= 0; index -= 1) {
    if (manifest.turns[index].turn_id === turnId) {
      manifest.turns[index].status = String(content.status || "completed");
      manifest.turns[index].ended_ms = event.elapsed_ms || 0;
      break;
    }
  }
}

function setFollowing(next) {
  following = next;
  if (following) {
    setPlaying(false);
    seek(totalPlaybackMs);
  }
  renderLive();
}

function renderLive() {
  if (!dom.live) return;
  const attached = socket !== null && socket.readyState === WebSocket.OPEN;
  // Offering "jump to live" on a session that ended hours ago is a lie: the
  // socket stays attachable forever, so being attached is not evidence of life.
  dom.live.hidden = !manifest.stream || !liveActive;
  dom.live.dataset.state = attached ? (following ? "following" : "behind") : "detached";
  dom.live.textContent = attached ? (following ? "● LIVE" : "↓ JUMP TO LIVE") : "reconnecting…";
  dom.live.disabled = !attached || following;
}

// -------------------------------------------------------------------- chrome ---

function populateChrome() {
  dom.title.textContent = manifest.title || manifest.session_id || "Session replay";
  renderMeta();
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

  renderTicks();
  renderTurns();
}

function renderMeta() {
  const parts = [];
  if (manifest.event_count) parts.push(`${manifest.event_count} events`);
  if (manifest.turns && manifest.turns.length) parts.push(`${manifest.turns.length} turns`);
  parts.push(formatDuration(manifest.duration_ms || 0));
  dom.meta.textContent = parts.join("  ·  ");
}

function renderTicks() {
  dom.ticks.replaceChildren();
  for (const turn of manifest.turns || []) {
    const offset = playbackOffsetForElapsed(turn.elapsed_ms || 0);
    const tick = document.createElement("span");
    tick.className = "kc-tick";
    tick.style.left = `${totalPlaybackMs ? (offset / totalPlaybackMs) * 100 : 0}%`;
    tick.title = turn.user_text || turn.turn_id;
    dom.ticks.append(tick);
  }
}

/** Nearest playback offset for a session-time value, used for turn seeking. */
function playbackOffsetForElapsed(elapsedMs) {
  // elapsed_ms is non-decreasing along the log, so this is a binary search.
  // It runs per tick per appended event while following a live session, and a
  // linear scan there made redrawing the ticks quadratic in session length.
  let low = 0;
  let high = events.length;
  while (low < high) {
    const mid = (low + high) >> 1;
    if ((events[mid].elapsed_ms || 0) >= elapsedMs) high = mid;
    else low = mid + 1;
  }
  return low < playbackAt.length ? playbackAt[low] : totalPlaybackMs;
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
  // Any deliberate move through the recording means the viewer wants to read
  // something rather than watch the edge, so it releases the live follow.
  const scrubbed = (to) => {
    setFollowing(false);
    setPlaying(false);
    seek(to);
  };
  dom.play.addEventListener("click", () => {
    setFollowing(false);
    setPlaying(!playing);
  });
  dom.speed.addEventListener("click", () => {
    speedIndex = (speedIndex + 1) % SPEEDS.length;
    dom.speed.textContent = SPEEDS[speedIndex] === 0 ? "max" : `${SPEEDS[speedIndex]}x`;
  });
  dom.scrub.addEventListener("input", () => scrubbed((Number(dom.scrub.value) / 1000) * totalPlaybackMs));
  dom.live.addEventListener("click", () => setFollowing(true));
  dom.theme.addEventListener("change", () => applyTheme(dom.theme.value));
  document.addEventListener("keydown", (event) => {
    if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) return;
    if (event.code === "Space") {
      event.preventDefault();
      setFollowing(false);
      setPlaying(!playing);
    } else if (event.code === "ArrowRight") {
      scrubbed(position + 5000);
    } else if (event.code === "ArrowLeft") {
      scrubbed(position - 5000);
    } else if (event.code === "Home") {
      scrubbed(0);
    } else if (event.code === "End") {
      if (manifest.stream) setFollowing(true);
      else seek(totalPlaybackMs);
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

/** How many entries exist across the main thread and every sub-agent. */
function transcriptSize() {
  let total = state.conversation.length;
  for (const activity of state.subAgents.values()) total += activity.steps.length;
  return total;
}

/**
 * The main transcript and every sub-agent trajectory as one chronological list.
 *
 * The fold keeps delegated work out of the main conversation so a client can
 * present it separately, but a replay reads as one thread: a sub-agent's
 * reasoning and tool calls belong where they happened, not in a side panel that
 * only ever showed the task line.
 *
 * Every entry carries the seq of the event that created it and never moves, and
 * events fold in seq order, so a new entry always sorts last. The merge is
 * therefore append-only, which is what lets the render diff stay index-based.
 */
function transcriptRows() {
  if (mergedRows && mergedRows.length === transcriptSize()) return mergedRows;
  const rows = state.conversation.map((item) => ({ item, agent: null }));
  for (const activity of state.subAgents.values()) {
    for (const step of activity.steps) rows.push({ item: step, agent: activity });
  }
  rows.forEach((row, order) => {
    row.order = order;
  });
  rows.sort((left, right) => {
    const a = left.item.seq ?? Number.MAX_SAFE_INTEGER;
    const b = right.item.seq ?? Number.MAX_SAFE_INTEGER;
    return a === b ? left.order - right.order : a - b;
  });
  // Name a sub-agent once per run rather than on every line it produces.
  let previousAgent = null;
  for (const row of rows) {
    row.lead = row.agent !== null && row.agent !== previousAgent;
    previousAgent = row.agent;
  }
  mergedRows = rows;
  return rows;
}

/**
 * Snapshot the fields renderEntry draws that the fold can still mutate.
 *
 * Every fold mutation installs a fresh value or reference (`text` is rebuilt by
 * concatenation, `artifacts` and the edit preview are reassigned), so comparing
 * a snapshot is exact and costs a handful of identity checks per entry.
 */
function entrySnapshot(row) {
  const item = row.item;
  return {
    item,
    agent: row.agent,
    lead: row.lead,
    text: item.text,
    complete: item.complete,
    status: item.status,
    toolName: item.tool_name || item.toolName,
    artifacts: item.artifacts,
    preview: item.editPreview || item.edit_preview,
  };
}

function entryChanged(previous, row) {
  const item = row.item;
  return (
    previous === undefined ||
    previous.item !== item ||
    previous.agent !== row.agent ||
    previous.lead !== row.lead ||
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
  const rows = transcriptRows();
  if (renderedCount > rows.length) {
    dom.transcript.replaceChildren();
    renderedCount = 0;
    renderedEntries = [];
  }
  for (let index = 0; index < renderedCount; index += 1) {
    const row = rows[index];
    if (!entryChanged(renderedEntries[index], row)) continue;
    dom.transcript.children[index]?.replaceWith(renderEntry(row));
    renderedEntries[index] = entrySnapshot(row);
  }
  for (let index = renderedCount; index < rows.length; index += 1) {
    dom.transcript.append(renderEntry(rows[index]));
    renderedEntries[index] = entrySnapshot(rows[index]);
  }
  renderedCount = rows.length;
  const atEnd = dom.transcript.parentElement;
  if (atEnd) atEnd.scrollTop = atEnd.scrollHeight;
}

function renderEntry(entry) {
  const item = entry.item;
  const row = document.createElement("div");
  row.className = "kc-entry";
  row.dataset.kind = item.kind;
  if (item.tone) row.dataset.tone = item.tone;
  if (entry.agent) {
    row.dataset.subAgent = entry.agent.name || entry.agent.key;
    if (entry.lead) {
      const lead = document.createElement("div");
      lead.className = "kc-sub-agent-lead";
      lead.textContent = entry.agent.task
        ? `${entry.agent.name} · ${entry.agent.task}`
        : entry.agent.name;
      row.append(lead);
    }
  }

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
    wrapper.append(renderArtifact(ref));
  }
  return wrapper;
}

/**
 * Where an artifact's bytes can be read from, or null if they are not reachable.
 *
 * A single-file export carries them inline. A directory bundle writes them to
 * `artifacts/<sha256>` beside the player. The served player has no such path, so
 * it keeps the text badge rather than drawing a broken image.
 */
function artifactSource(ref) {
  const inlined = globalThis.__KC_REPLAY__;
  if (inlined && inlined.artifacts && inlined.artifacts[ref.sha256]) {
    return inlined.artifacts[ref.sha256];
  }
  // A single-file export has no sibling files, so an artifact missing from the
  // inline map above is simply unreachable — never a relative path that 404s.
  if (manifest.artifacts_inline) return null;
  if (!manifest.stream && manifest.artifact_count) return `artifacts/${ref.sha256}`;
  return null;
}

function renderArtifact(ref) {
  const source = artifactSource(ref);
  if (source && String(ref.media_type || "").startsWith("image/")) {
    const image = document.createElement("img");
    image.className = "kc-artifact-image";
    image.src = source;
    image.alt = "Image captured during this session";
    image.loading = "lazy";
    return image;
  }
  const badge = document.createElement("span");
  badge.className = "kc-artifact";
  const size = ref.chars ? `${ref.chars.toLocaleString()} chars` : `${ref.bytes} bytes`;
  badge.textContent = `artifact · ${size}`;
  return badge;
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
