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
  workflow: "◆",
  agent: "◈",
  background_delivery: "↳",
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
let renderedTaskList;
let renderedUsage;
/** Tool entries the viewer has opened. Tool output is collapsed by default. */
const expandedTools = new Set();
/** Sub-agent key the transcript is narrowed to, or null for the whole thread. */
let focusedAgent = null;
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
  if (manifest.stream) {
    watchConnectivity();
    attachStream();
  }
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
    "taskList",
    "usage",
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
  dom.terminalOwner = document.createElement("div");
  dom.terminalOwner.id = "terminalOwner";
  dom.terminalOwner.className = "kc-terminal-owner";
  dom.terminalTruncated = document.createElement("div");
  dom.terminalTruncated.id = "terminalTruncated";
  dom.terminalTruncated.className = "kc-terminal-truncated";
  dom.terminalTruncated.textContent = "Older terminal output truncated.";
  dom.terminal.before(dom.terminalOwner, dom.terminalTruncated);
  dom.terminal.setAttribute("aria-labelledby", "terminalOwner");
  renderTerminal();
  renderTaskList();
  renderUsage();
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
    renderedTaskList = undefined;
    renderedUsage = undefined;
    mergedRows = null;
    dom.transcript.replaceChildren();
  }
  for (let index = appliedCount; index < target; index += 1) {
    fold(state, events[index]);
  }
  mergedRows = null;
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

/**
 * Treat losing the network as losing the socket.
 *
 * A dropped connection does not close a WebSocket: the browser keeps it OPEN
 * until TCP gives up, which can take minutes. For that whole time the badge
 * would read "LIVE" over a transcript that stopped growing, which is the one
 * thing it must never do. The browser tells us the moment connectivity goes, so
 * use it and let the existing backoff handle coming back.
 */
function watchConnectivity() {
  addEventListener("offline", () => {
    if (socket) socket.close();
    renderLive();
  });
  addEventListener("online", () => {
    if (reconnectTimer !== null) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    reconnectDelay = RECONNECT_MIN_MS;
    if (!socket) attachStream();
  });
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
  const online = typeof navigator === "undefined" || navigator.onLine !== false;
  const attached = online && socket !== null && socket.readyState === WebSocket.OPEN;
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
  // The recorder guarantees elapsed_ms is non-decreasing along the log, so this
  // is a binary search. It runs per tick per appended event while following a
  // live session, and a linear scan there made redrawing the ticks quadratic in
  // session length.
  //
  // Recordings made before that guarantee existed can step backwards where two
  // streamed segments overlapped, and a turn seek there may land a few events
  // off. It stays a seek, never a crash: extendTimeline clamps a negative gap to
  // zero, so the timeline itself is monotonic whatever the log says.
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
  dom.transcript.addEventListener("click", (event) => {
    if (!(event.target instanceof Element)) return;
    const card = event.target.closest(".kc-agent-card[data-agent-key]");
    if (card) {
      focusAgent(card.dataset.agentKey);
      return;
    }
    const toggle = event.target.closest("[data-tool-key]");
    if (!toggle) return;
    const key = toggle.dataset.toolKey;
    if (expandedTools.has(key)) expandedTools.delete(key);
    else expandedTools.add(key);
    redrawTranscript();
  });
  dom.subAgents.addEventListener("click", (event) => {
    const row = event.target instanceof Element ? event.target.closest("[data-agent-key]") : null;
    if (!row) return;
    focusAgent(focusedAgent === row.dataset.agentKey ? null : row.dataset.agentKey);
  });
  document.addEventListener("keydown", (event) => {
    if (
      event.defaultPrevented ||
      (event.target instanceof Element &&
        event.target.closest("button, a, input, select, textarea, [contenteditable], [role='button']"))
    ) return;
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

/** Rows in the main thread: its own entries plus one card per dispatch. */
function transcriptSize() {
  return state.conversation.length + state.subAgents.size;
}

/** Seq of an activity's first step, which is where its card belongs. */
function activitySeq(activity) {
  for (const step of activity.steps) {
    if (step.seq !== null && step.seq !== undefined) return step.seq;
  }
  return Number.MAX_SAFE_INTEGER;
}

/**
 * The main thread, with each delegated agent standing in as a single card.
 *
 * Splicing a sub-agent's steps into the main transcript reads fine for one
 * delegate and falls apart for several: parallel agents interleave line by
 * line, so three answers arrive in an order unrelated to the three questions
 * above them and nothing says which belongs to which. The terminal has always
 * collapsed a dispatch to one entry you open, and that is the model here — the
 * card reports the agent and whether it is still going, and clicking it shows
 * that agent's thread on its own.
 *
 * An agent first seen through metadata or terminal output may gain its first
 * transcript step later, so the merge is rebuilt after each fold batch.
 */
function transcriptRows() {
  if (mergedRows && mergedRows.length === transcriptSize()) return mergedRows;
  const rows = state.conversation.map((item) => ({ item, agent: null, seq: item.seq }));
  for (const activity of state.subAgents.values()) {
    rows.push({ item: null, agent: activity, card: true, seq: activitySeq(activity) });
  }
  rows.forEach((row, order) => {
    row.order = order;
  });
  rows.sort((left, right) => {
    const a = left.seq ?? Number.MAX_SAFE_INTEGER;
    const b = right.seq ?? Number.MAX_SAFE_INTEGER;
    return a === b ? left.order - right.order : a - b;
  });
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
  if (row.card) {
    // A card is a live view of its agent, so what it draws keeps changing after
    // it is on screen: the status settles, notices change, and the step count climbs.
    return {
      card: true,
      agent: row.agent,
      status: row.agent.status,
      notice: row.agent.notice,
      steps: row.agent.steps.length,
      metadata: agentMetadataSnapshot(row.agent),
    };
  }
  return {
    item,
    agent: row.agent,
    text: item.text,
    complete: item.complete,
    status: item.status,
    toolName: item.tool_name || item.toolName,
    toolSubject: item.tool_subject || item.toolSubject,
    artifacts: item.artifacts,
    imageCount: item.image_count ?? item.imageCount,
    preview: item.editPreview || item.edit_preview,
  };
}

function entryChanged(previous, row) {
  if (previous === undefined) return true;
  if (row.card || previous.card) {
    return (
      previous.card !== row.card ||
      previous.agent !== row.agent ||
      previous.status !== row.agent.status ||
      previous.notice !== row.agent.notice ||
      previous.steps !== row.agent.steps.length ||
      previous.metadata !== agentMetadataSnapshot(row.agent)
    );
  }
  const item = row.item;
  return (
    previous.item !== item ||
    previous.agent !== row.agent ||
    previous.text !== item.text ||
    previous.complete !== item.complete ||
    previous.status !== item.status ||
    previous.toolName !== (item.tool_name || item.toolName) ||
    previous.toolSubject !== (item.tool_subject || item.toolSubject) ||
    previous.artifacts !== item.artifacts ||
    previous.imageCount !== (item.image_count ?? item.imageCount) ||
    previous.preview !== (item.editPreview || item.edit_preview)
  );
}

/**
 * The rows currently on screen: the main thread, or one agent's own steps.
 *
 * Both lists only ever grow at the end, so the index-based render diff holds
 * while a filter is active. Changing the filter itself does not, which is why
 * focusAgent redraws from scratch.
 */
function visibleRows() {
  if (focusedAgent === null) return transcriptRows();
  const activity = state.subAgents.get(focusedAgent);
  if (!activity) return [];
  return activity.steps.map((item) => ({ item, agent: activity, seq: item.seq }));
}

function agentKeyOf(activity) {
  return activity.key ?? activity.name;
}

/** Narrow the transcript to one agent, or clear the filter with null. */
function focusAgent(key) {
  focusedAgent = key;
  redrawTranscript();
  renderSubAgents();
  renderTerminal();
  if (!document.activeElement || document.activeElement === document.body) {
    if (key === null) dom.subAgents.querySelector(".kc-sub-agent")?.focus({ preventScroll: true });
    else document.querySelector(".kc-agent-back")?.focus({ preventScroll: true });
  }
  if (key === null) return;
  // The list scrolls, so the selected agent can sit outside it after a redraw —
  // leaving whichever row happens to be under the cursor looking like the
  // selected one.
  dom.subAgents
    .querySelector(`[data-agent-key="${CSS.escape(key)}"]`)
    ?.scrollIntoView({ block: "nearest" });
}

/** Throw away the rendered transcript and build it again from current state. */
function redrawTranscript() {
  dom.transcript.replaceChildren();
  renderedCount = 0;
  renderedEntries = [];
  renderAgentHeader();
  renderTranscript();
}

/**
 * Say whose thread is on screen while one agent is open.
 *
 * Without it the transcript silently becomes a different, much shorter
 * conversation and the only clue is a highlight in the rail.
 */
function renderAgentHeader() {
  let header = document.getElementById("agentHeader");
  const activity = focusedAgent === null ? null : state.subAgents.get(focusedAgent);
  if (focusedAgent === null) {
    header?.remove();
    return;
  }
  if (!header) {
    header = document.createElement("div");
    header.className = "kc-agent-header";
    header.id = "agentHeader";
    const back = document.createElement("button");
    back.type = "button";
    back.className = "kc-agent-back";
    back.textContent = "← Back to the session";
    back.addEventListener("click", () => focusAgent(null));
    const title = document.createElement("div");
    title.className = "kc-agent-header-title";
    const status = document.createElement("span");
    status.className = "kc-tool-status";
    const notice = document.createElement("span");
    notice.className = "kc-agent-header-notice";
    const details = document.createElement("div");
    details.className = "kc-agent-details";
    header.append(back, title, status, notice, details);
    dom.transcript.parentElement?.prepend(header);
  }
  const title = header.querySelector(".kc-agent-header-title");
  title.textContent = activity ? agentDisplayName(activity) : `${focusedAgent} (not present at this position)`;
  title.title = activity?.task || "";
  const status = header.querySelector(".kc-tool-status");
  status.dataset.status = activity?.status || "unavailable";
  status.textContent = activity?.status || "unavailable";
  const notice = header.querySelector(".kc-agent-header-notice");
  notice.textContent = activity?.notice || "";
  notice.hidden = !activity?.notice;
  const details = header.querySelector(".kc-agent-details");
  const metadata = activity ? agentMetadataSnapshot(activity) : "";
  if (details.dataset.snapshot !== metadata) {
    const hadFocus = details.contains(document.activeElement);
    details.replaceChildren();
    if (activity) fillAgentDetails(details, activity, true);
    details.dataset.snapshot = metadata;
    if (hadFocus) {
      (details.querySelector("button") || header.querySelector(".kc-agent-back")).focus({ preventScroll: true });
    }
  }
}

function renderTranscript() {
  // An entry that is no longer the newest can still change. Streaming segments
  // are keyed by uuid, so reasoning keeps accumulating after the assistant prose
  // that interrupted it was appended — every turn interleaves them — and a tool
  // entry resolves long after later entries exist. Re-rendering only the newest
  // entry left those frozen mid-stream, spinner and all.
  const rows = visibleRows();
  renderAgentHeader();
  const active = dom.transcript.contains(document.activeElement) ? document.activeElement : null;
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
  if (active && !active.isConnected) {
    const selector = active.dataset.toolKey
      ? `[data-tool-key="${CSS.escape(active.dataset.toolKey)}"]`
      : `[data-agent-key="${CSS.escape(active.dataset.agentKey)}"]`;
    dom.transcript.querySelector(selector)?.focus({ preventScroll: true });
  }
  const atEnd = dom.transcript.parentElement;
  if (atEnd) atEnd.scrollTop = atEnd.scrollHeight;
}

/** The label is what tells one agent of a fan-out from the next; they share a name. */
function agentDisplayName(activity) {
  return activity.label || activity.name || activity.key;
}

function agentParent(activity) {
  const id = activity.parentAgentId;
  if (id === null || id === undefined || id === "") return { text: "unknown", target: null };
  if (id === "root") return { text: "root", target: null, root: true };
  const byId = new Map();
  for (const agent of state.subAgents.values()) {
    if (agent.agentId) byId.set(agent.agentId, agent);
  }
  const parent = byId.get(id);
  if (!parent) return { text: `${id} (unavailable)`, target: null };
  const visited = new Set(activity.agentId ? [activity.agentId] : []);
  let ancestor = parent;
  let cyclic = false;
  while (ancestor) {
    if (visited.has(ancestor.agentId)) {
      cyclic = true;
      break;
    }
    visited.add(ancestor.agentId);
    ancestor = byId.get(ancestor.parentAgentId);
  }
  const label = agentDisplayName(parent);
  return {
    text: `${id}${label !== id ? ` · ${label}` : ""}${cyclic ? " (cycle)" : ""}`,
    target: agentKeyOf(parent),
  };
}

function routeText(route) {
  if (route === null || route === undefined) return "unknown";
  return typeof route === "string" ? route : JSON.stringify(route);
}

function agentFacts(activity) {
  return [
    ["Agent", activity.agentId ?? "unknown"],
    ["Parent", agentParent(activity).text],
    ["Parent tool call", activity.parentToolCallId ?? "unknown"],
    ["Depth", activity.depth ?? "unknown"],
    ["Background", activity.background === true ? "yes" : activity.background === false ? "no" : "unknown"],
    ["Requested routing", routeText(activity.requestedRouting)],
    ["Effective routing", routeText(activity.effectiveRouting)],
    ["Total tokens", activity.totalTokens === null || activity.totalTokens === undefined
      ? "unknown" : activity.totalTokens.toLocaleString()],
  ];
}

function agentMetadataSnapshot(activity) {
  return JSON.stringify([
    activity.key, activity.name, activity.label, activity.task, activity.phase, activity.workflowRunId, agentFacts(activity),
  ]);
}

function fillAgentDetails(container, activity, interactive = false) {
  for (const [label, value] of agentFacts(activity)) {
    const fact = document.createElement("div");
    fact.className = "kc-agent-fact";
    const parent = label === "Parent" && interactive ? agentParent(activity) : null;
    fact.textContent = `${label}: `;
    const content = document.createElement(parent && (parent.root || parent.target !== null) ? "button" : "span");
    content.textContent = String(value);
    if (content instanceof HTMLButtonElement) {
      content.type = "button";
      content.className = "kc-agent-parent";
      content.addEventListener("click", () => focusAgent(parent.target));
    }
    fact.append(content);
    container.append(fact);
  }
}

/**
 * A dispatch, as one openable entry in the main thread.
 *
 * This is what a delegate looks like from outside: who it is, what it was
 * asked, whether it is still going, and how much it has done. Its actual
 * thread is one click away rather than spliced into this one.
 */
function renderAgentCard(activity) {
  const row = document.createElement("div");
  row.className = "kc-entry";
  row.dataset.kind = "agent";

  const glyph = document.createElement("span");
  glyph.className = "kc-glyph";
  glyph.textContent = GLYPHS.agent;
  row.append(glyph);

  const card = document.createElement("button");
  card.type = "button";
  card.className = "kc-agent-card";
  card.dataset.agentKey = agentKeyOf(activity);
  card.dataset.status = activity.status || "running";

  const head = document.createElement("div");
  head.className = "kc-agent-card-head";
  const name = document.createElement("span");
  name.className = "kc-agent-card-name";
  name.textContent = agentDisplayName(activity);
  head.append(name);
  if (activity.phase) {
    const phase = document.createElement("span");
    phase.className = "kc-sub-agent-phase";
    phase.textContent = activity.phase;
    head.append(phase);
  }
  const status = document.createElement("span");
  status.className = "kc-tool-status";
  status.dataset.status = activity.status === "running" ? "running" : activity.status;
  status.textContent = activity.status || "running";
  head.append(status);
  if (activity.status === "running") head.append(spinnerNode());
  const steps = document.createElement("span");
  steps.className = "kc-agent-card-steps";
  steps.textContent = `${activity.steps.length} step${activity.steps.length === 1 ? "" : "s"} · open`;
  head.append(steps);
  card.append(head);

  if (activity.task) {
    const task = document.createElement("div");
    task.className = "kc-agent-card-task";
    task.textContent = activity.task;
    card.append(task);
  }
  if (activity.notice) {
    const notice = document.createElement("div");
    notice.className = "kc-agent-card-notice";
    notice.textContent = activity.notice;
    card.append(notice);
  }
  const details = document.createElement("div");
  details.className = "kc-agent-details";
  fillAgentDetails(details, activity);
  card.append(details);
  row.append(card);
  return row;
}

function renderEntry(entry) {
  if (entry.card) return renderAgentCard(entry.agent);
  const item = entry.item;
  const row = document.createElement("div");
  row.className = "kc-entry";
  row.dataset.kind = item.kind;
  if (item.tone) row.dataset.tone = item.tone;
  if (entry.agent) row.dataset.subAgent = agentDisplayName(entry.agent);

  const glyph = document.createElement("span");
  glyph.className = "kc-glyph";
  glyph.textContent = GLYPHS[item.kind] || GLYPHS.system;
  row.append(glyph);

  const body = document.createElement("div");
  body.className = "kc-body";
  if (item.kind === "tool") {
    body.append(renderToolBody(item));
  } else if (item.kind === "background_delivery") {
    body.append(renderBackgroundDelivery(item));
  } else {
    body.textContent = item.text;
    if (item.kind === "user") {
      const images = (item.artifacts || []).filter(
        (ref) => ref.purpose === "image" && String(ref.media_type || "").startsWith("image/"),
      );
      for (const ref of images) body.append(renderArtifact(ref));
      const count = item.image_count ?? item.imageCount;
      if (Number.isInteger(count) && count > images.length) {
        body.append(missingImageBadge(count - images.length));
      }
    }
    if (!item.complete) body.append(spinnerNode());
  }
  row.append(body);
  return row;
}

function renderBackgroundDelivery(item) {
  const delivery = item.background_delivery || item.backgroundDelivery;
  const metadata = delivery.metadata || {};
  const wrapper = document.createElement("section");
  wrapper.className = "kc-background-delivery";
  wrapper.dataset.completionId = delivery.completion_id;
  wrapper.dataset.disposition = delivery.disposition;

  const heading = document.createElement("div");
  heading.className = "kc-background-heading";
  heading.textContent = `Background report · ${delivery.agent_type} · ${delivery.status}`;
  const disposition = document.createElement("div");
  disposition.className = "kc-background-disposition";
  disposition.textContent = delivery.disposition === "archived"
    ? "Archived — NOT added to live context."
    : "Delivered — added to live context.";
  const task = document.createElement("div");
  task.className = "kc-background-task";
  task.textContent = `Task: ${delivery.task}`;
  wrapper.append(heading, disposition, task);

  const details = document.createElement("details");
  details.className = "kc-background-details";
  const summary = document.createElement("summary");
  summary.textContent = "Report attribution";
  details.append(summary);
  const facts = [
    ["Job", delivery.job_id],
    ["Completion", delivery.completion_id],
    ["Epoch", delivery.epoch_id],
    ["Launch turn", metadata.turn_id],
    ["Parent tool call", metadata.parent_tool_call_id],
    ["Effective routing", metadata.effective_routing],
    ["Workspace", metadata.workspace_id],
    ["Thread", metadata.thread_id],
    ["Project", metadata.project_path],
  ];
  for (const [label, value] of facts) {
    const fact = document.createElement("div");
    fact.textContent = `${label}: ${value == null ? "unknown" :
      typeof value === "object" ? JSON.stringify(value) : String(value)}`;
    details.append(fact);
  }
  const durable = document.createElement("pre");
  durable.className = "kc-background-metadata";
  durable.textContent = JSON.stringify(metadata, null, 2);
  details.append(durable);
  wrapper.append(details);

  const report = document.createElement("div");
  report.className = "kc-background-report";
  report.textContent = item.text;
  wrapper.append(report);
  hydrateBackgroundReport(report, item, wrapper);
  return wrapper;
}

const backgroundReportLoads = new Map();

function hydrateBackgroundReport(report, item, wrapper) {
  const ref = (item.artifacts || []).find((candidate) =>
    candidate.purpose === "tool_result" &&
    String(candidate.media_type || "").startsWith("text/plain") &&
    item.text.includes(`characters elided; full payload sha256 ${candidate.sha256} …]`),
  );
  if (!ref) return;
  const notice = document.createElement("div");
  notice.className = "kc-background-artifact-notice";
  notice.textContent = "Loading complete report…";
  wrapper.append(notice);
  const source = artifactSource(ref, true);
  if (!source) {
    notice.textContent = "Complete report unavailable; recorded preview shown.";
    return;
  }
  let loaded = backgroundReportLoads.get(ref.sha256);
  if (!loaded) {
    loaded = fetch(source).then((response) => {
      if (!response.ok) throw new Error("Report unavailable");
      return response.text();
    }).catch(() => null);
    backgroundReportLoads.set(ref.sha256, loaded);
  }
  loaded.then((text) => {
    if (text === null) {
      notice.textContent = "Complete report unavailable; recorded preview shown.";
    } else {
      report.textContent = text;
      notice.remove();
    }
  });
}

/** Identity for remembering whether one tool entry is open across re-renders. */
function toolKey(item) {
  return item.tool_call_id || item.toolCallId || `seq:${item.seq}`;
}

function renderToolBody(item) {
  const wrapper = document.createDocumentFragment();
  const preview = item.editPreview || item.edit_preview;
  const refs = item.artifacts || [];
  // An image is the result someone wants to look at, and it is already compact.
  // Only the wall of text hides behind the toggle.
  const images = refs.filter((ref) => String(ref.media_type || "").startsWith("image/"));
  const others = refs.filter((ref) => !String(ref.media_type || "").startsWith("image/"));
  const hasDetail = Boolean(item.text || (preview && preview.diff) || others.length);
  const key = toolKey(item);
  const open = expandedTools.has(key);

  const head = document.createElement("div");
  head.className = "kc-tool-head";

  // A tool result is usually a wall of output, and a transcript of them is
  // unreadable — you scroll past screens of JSON looking for the sentence
  // between two calls. Collapsed, the head alone says what ran and how it went.
  const toggle = document.createElement(hasDetail ? "button" : "span");
  toggle.className = "kc-tool-name";
  toggle.textContent = item.tool_name || item.toolName || "tool";
  if (hasDetail) {
    toggle.type = "button";
    toggle.dataset.toolKey = key;
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    const caret = document.createElement("span");
    caret.className = "kc-tool-caret";
    caret.setAttribute("aria-hidden", "true");
    caret.textContent = open ? "▾" : "▸";
    toggle.prepend(caret);
  }
  head.append(toggle);

  const toolSubject = item.tool_subject || item.toolSubject;
  if (typeof toolSubject === "string" && toolSubject) {
    head.className += " has-subject";
    const subject = document.createElement("span");
    subject.className = "kc-tool-subject";
    subject.textContent = toolSubject;
    head.append(subject);
  }

  const status = document.createElement("span");
  status.className = "kc-tool-status";
  status.dataset.status = item.status || "running";
  status.textContent = item.status || "running";
  head.append(status);
  if (item.status === "running") head.append(spinnerNode());
  wrapper.append(head);

  for (const ref of images) wrapper.append(renderArtifact(ref));

  if (!hasDetail || !open) return wrapper;

  if (item.text) {
    const output = document.createElement("div");
    output.className = "kc-tool-output";
    output.textContent = item.text;
    wrapper.append(output);
  }
  if (preview && preview.diff) {
    wrapper.append(renderDiff(String(preview.diff)));
  }
  for (const ref of others) wrapper.append(renderArtifact(ref));
  return wrapper;
}

/**
 * Where an artifact's bytes can be read from, or null if they are not reachable.
 *
 * Three shapes, three answers. A single-file export carries them inline. A
 * directory bundle writes them to `artifacts/<sha256>` beside the player. A
 * served session has neither, but it does have the session API, so it reads
 * them from the artifact endpoint — which is gated on the same token and the
 * same shareable-purpose allowlist as everything else.
 */
function artifactSource(ref, text = false) {
  if (!/^[a-f0-9]{64}$/.test(ref.sha256)) return null;
  const inlined = globalThis.__KC_REPLAY__;
  if (inlined && inlined.artifacts && inlined.artifacts[ref.sha256]) {
    const source = inlined.artifacts[ref.sha256];
    const allowed = text ? /^data:text\/plain(?:;\s*charset=utf-8)?;base64,/i : /^data:image\/[^,]+;base64,/i;
    return typeof source === "string" && allowed.test(source) ? source : null;
  }
  // A single-file export has no sibling files, so an artifact missing from the
  // inline map above is simply unreachable — never a relative path that 404s.
  if (manifest.artifacts_inline) return null;
  if (manifest.stream) {
    return manifest.session_id ? `/api/sessions/${manifest.session_id}/artifacts/${ref.sha256}` : null;
  }
  if (manifest.artifact_count) return `artifacts/${ref.sha256}`;
  return null;
}

function renderArtifact(ref) {
  const source = artifactSource(ref);
  if (String(ref.media_type || "").startsWith("image/")) {
    if (!source) return missingImageBadge();
    const image = document.createElement("img");
    image.className = "kc-artifact-image";
    image.src = source;
    image.alt = "Image captured during this session";
    image.loading = "lazy";
    image.addEventListener("error", () => image.replaceWith(missingImageBadge()), { once: true });
    return image;
  }
  const badge = document.createElement("span");
  badge.className = "kc-artifact";
  const size = ref.chars ? `${ref.chars.toLocaleString()} chars` : `${ref.bytes} bytes`;
  badge.textContent = `artifact · ${size}`;
  return badge;
}

function missingImageBadge(count = 1) {
  const badge = document.createElement("span");
  badge.className = "kc-artifact kc-missing-image";
  badge.textContent = count === 1 ? "Image unavailable" : `${count.toLocaleString()} images unavailable`;
  badge.title = "Image bytes are missing, not shared, too large to embed, or could not be loaded.";
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
  renderTaskList();
  renderUsage();
  renderSubAgents();
  renderTerminal();
  dom.status.textContent = state.status || "";
  dom.truncated.hidden = !state.recordingTruncated;
}

function usageText(parent, tag, className, text) {
  const node = document.createElement(tag);
  node.className = className;
  node.textContent = text;
  parent.append(node);
  return node;
}

function usageRow(parent, field, label, value) {
  const row = document.createElement("div");
  row.className = "kc-usage-row";
  row.dataset.field = field;
  usageText(row, "span", "kc-usage-label", label);
  usageText(row, "span", "kc-usage-value", value ?? "Unavailable");
  parent.append(row);
}

function usageGroup(field, title) {
  const group = document.createElement("section");
  group.className = "kc-usage-group";
  group.dataset.group = field;
  usageText(group, "h3", "kc-usage-heading", title);
  dom.usage.append(group);
  return group;
}

function usageCompleteness(parent, complete, subject) {
  const note = usageText(parent, "p", "kc-usage-note",
    complete === true ? `Complete ${subject}` :
      complete === false ? `Incomplete ${subject} · known subtotals only` : `${subject} unavailable`);
  note.dataset.state = complete === true ? "complete" : complete === false ? "incomplete" : "unknown";
}

function renderUsage() {
  const usage = state.usage ?? null;
  const snapshot = JSON.stringify(usage);
  if (renderedUsage === snapshot) return;
  renderedUsage = snapshot;
  dom.usage.replaceChildren();
  dom.usage.dataset.state = usage === null ? "unknown" : "recorded";
  usageText(dom.usage, "p", "kc-usage-summary", "Session-wide · lifetime across all agents and epochs");
  if (usage === null) {
    usageText(dom.usage, "p", "kc-empty", "Usage unavailable. No usage recorded.");
    return;
  }

  const requests = usageGroup("requests", "Requests");
  for (const [field, label] of [
    ["requests", "Total requests"], ["responses", "Responses"], ["reported", "Complete token reports"],
    ["unreported", "Responses without complete tokens"], ["failed", "Failed / cancelled"],
    ["open", "Open"], ["abandoned", "Abandoned"], ["partial", "Partial token reports"],
    ["missing", "Missing token reports"],
  ]) {
    usageRow(requests, field, label, usage[field]);
  }

  const tokens = usageGroup("tokens", "Tokens");
  usageCompleteness(tokens, usage.complete, "primary token evidence");
  const known = usage.known_requests ?? usage.known_fields ?? {};
  for (const [field, label] of [
    ["input_tokens", "Input"], ["output_tokens", "Output"], ["total_tokens", "Total"],
    ["cache_read_input_tokens", "Cache read"], ["cache_write_input_tokens", "Cache write"],
    ["reasoning_output_tokens", "Reasoning output"],
  ]) {
    const count = known[field];
    const available = usage[field] != null && (count > 0 || usage.requests === 0);
    const coverage = count != null && usage.requests != null ? ` · ${count}/${usage.requests} requests` : "";
    const partial = available && count < usage.requests ? " · partial subtotal" : "";
    usageRow(tokens, field, label, `${available ? usage[field] : "Unavailable"}${coverage}${partial}`);
  }

  const coverage = usageGroup("coverage", "Recording coverage");
  const full = usage.coverage?.full;
  const coverageNote = usageText(coverage, "p", "kc-usage-note",
    full === true ? "Full recording coverage" :
      full === false ? "Incomplete recording coverage" : "Recording coverage unavailable");
  coverageNote.dataset.state = full === true ? "complete" : full === false ? "incomplete" : "unknown";
  for (const [field, count] of Object.entries(usage.coverage ?? {})) {
    if (field !== "full" && (count || field === "accounted_runs")) {
      usageRow(coverage, field, field.replaceAll("_", " "), count);
    }
  }

  const costs = usage.costs ?? {};
  const costGroup = usageGroup("costs", "Cost evidence");
  usageCompleteness(costGroup, costs.complete, "cost evidence");
  for (const [field, label, count] of [
    ["reported", "Provider-reported requests", costs.reported],
    ["estimated", "Estimated requests", costs.estimated],
    ["missing", "Missing cost requests", costs.missing_requests ?? costs.missing],
    ["partial", "Partial cost requests", costs.partial_requests ?? costs.partial],
  ]) {
    usageRow(costGroup, field, label, count);
  }
  for (const [field, title] of [["currencies", "Monetary cost"], ["non_monetary", "Non-monetary credits"]]) {
    const group = usageGroup(field, title);
    const rows = Object.entries(costs[field] ?? {});
    if (!rows.length) usageText(group, "p", "kc-empty", "Unavailable · no cost recorded.");
    for (const [denomination, amounts] of rows) {
      const row = document.createElement("section");
      row.className = "kc-usage-group";
      row.dataset.denomination = denomination;
      usageText(row, "h4", "kc-usage-heading", denomination);
      const decimal = (value) => typeof value === "string" ? value : "Unavailable";
      usageRow(row, "total", costs.complete === true ? "Total" : "Known subtotal", decimal(amounts.total));
      for (const [source, label] of [["reported", "Provider-reported"], ["estimated", "Estimated"]]) {
        const count = amounts[`${source}_requests`];
        const amount = count === 0 ? "Unavailable" : decimal(amounts[source]);
        usageRow(row, source, label, `${amount} · ${count ?? "unknown"} requests`);
      }
      usageRow(row, "requests", "Requests in this denomination", amounts.requests);
      if (amounts.token_cost_subtotal != null) {
        usageRow(row, "token_cost_subtotal", "Token cost subtotal", decimal(amounts.token_cost_subtotal));
      }
      group.append(row);
    }
  }

  const persistence = usageGroup("persistence", "Persistence health");
  const health = usage.persistence?.healthy;
  const healthNote = usageText(persistence, "p", "kc-usage-note",
    health === true ? "Healthy" : health === false ? "Persistence issues recorded" : "Persistence health unavailable");
  healthNote.dataset.state = health === true ? "healthy" : health === false ? "unhealthy" : "unknown";
  for (const [field, label, count] of [
    ["persist_failures", "Persistence failures", usage.persistence?.persist_failures ?? usage.persist_failures],
    ["unclosed_runs", "Unclosed runs", usage.persistence?.unclosed_runs ?? usage.coverage?.unclosed_runs],
    ["lost_requests", "Lost requests", usage.persistence?.lost_requests ?? usage.coverage?.lost_requests],
  ]) {
    usageRow(persistence, field, label, count);
  }
}

function renderTaskList() {
  const taskList = state.taskList ?? null;
  const snapshot = JSON.stringify(taskList);
  if (renderedTaskList === snapshot) return;
  renderedTaskList = snapshot;
  dom.taskList.replaceChildren();
  const summary = document.createElement("p");
  summary.className = "kc-task-summary";
  summary.textContent = "Session-wide";
  dom.taskList.append(summary);
  if (taskList === null || (taskList.markdown === "" && taskList.entries.length === 0)) {
    dom.taskList.dataset.state = taskList === null ? "unknown" : "cleared";
    const empty = document.createElement("p");
    empty.className = "kc-empty";
    empty.textContent = taskList === null ? "No task list recorded yet." : "Task list cleared.";
    dom.taskList.append(empty);
    return;
  }
  dom.taskList.dataset.state = "populated";
  const completed = taskList.entries.filter((entry) => entry.status === "completed").length;
  summary.textContent += ` · ${completed}/${taskList.entries.length} completed`;
  if (taskList.entries.length) {
    const list = document.createElement("ul");
    list.className = "kc-task-entries";
    list.setAttribute("aria-label", "Session tasks");
    for (const entry of taskList.entries) {
      const row = document.createElement("li");
      row.className = "kc-task-entry";
      row.dataset.status = entry.status;
      const mark = document.createElement("span");
      mark.className = "kc-task-mark";
      mark.textContent = entry.status === "completed" ? "✓" : "○";
      mark.setAttribute("role", "img");
      mark.setAttribute("aria-label", entry.status === "completed" ? "Completed" : "Pending");
      const text = document.createElement("span");
      text.className = "kc-task-text";
      text.textContent = entry.text;
      row.append(mark, text);
      list.append(row);
    }
    dom.taskList.append(list);
  }
  const label = document.createElement("p");
  label.className = "kc-task-source-label";
  label.textContent = "Full content";
  const source = document.createElement("pre");
  source.className = "kc-task-source";
  source.setAttribute("aria-label", "Full task list content");
  source.textContent = taskList.markdown;
  dom.taskList.append(label, source);
}

function renderTerminal() {
  const activity = focusedAgent === null ? null : state.subAgents.get(focusedAgent);
  const owner = focusedAgent === null ? "root" : activity?.agentId || focusedAgent;
  const label = activity ? agentDisplayName(activity) : owner;
  dom.terminalOwner.textContent = `Owner: ${label === owner ? owner : `${label} (${owner})`}`;
  dom.terminalOwner.dataset.owner = owner;
  const terminal = focusedAgent === null ? state.terminal : activity?.terminal;
  dom.terminalTruncated.hidden = !(focusedAgent === null ? state.terminalTruncated : activity?.terminalTruncated);
  dom.terminal.textContent = terminal || "";
  if (!terminal) {
    const empty = document.createElement("span");
    empty.className = "kc-empty";
    empty.textContent = "No terminal output yet.";
    dom.terminal.append(empty);
  }
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
    if (focusedAgent !== null && document.activeElement === document.body) {
      document.querySelector(".kc-agent-back")?.focus({ preventScroll: true });
    }
    return;
  }
  const active = dom.subAgents.contains(document.activeElement) ? document.activeElement : null;
  let all = dom.subAgents.querySelector(".kc-sub-agent-all");
  if (!all) {
    dom.subAgents.replaceChildren();
    all = document.createElement("button");
    all.type = "button";
    all.className = "kc-sub-agent-all";
    all.addEventListener("click", () => focusAgent(null));
    dom.subAgents.append(all);
  }
  all.textContent = focusedAgent === null ? `All agents (${entries.length})` : "← Back to the whole session";
  all.disabled = focusedAgent === null;
  const retained = new Set([all]);
  for (const activity of entries) {
    const key = agentKeyOf(activity);
    let row = dom.subAgents.querySelector(`[data-agent-key="${CSS.escape(key)}"]`);
    if (!row) {
      row = document.createElement("button");
      row.type = "button";
      row.className = "kc-sub-agent";
      row.dataset.agentKey = key;
      dom.subAgents.append(row);
    }
    retained.add(row);
    row.dataset.status = activity.status || "running";
    row.dataset.focused = String(focusedAgent === key);
    row.setAttribute("aria-pressed", focusedAgent === key ? "true" : "false");
    const snapshot = JSON.stringify([agentMetadataSnapshot(activity), activity.lastText, activity.notice]);
    if (row.dataset.snapshot === snapshot) continue;
    row.dataset.snapshot = snapshot;
    row.replaceChildren();
    const name = document.createElement("div");
    name.className = "kc-sub-agent-name";
    // A workflow labels each agent; without it every agent in a fan-out is
    // called "general-agent" and they are impossible to tell apart.
    name.textContent = activity.label ? `◆ ${activity.label}` : `◆ ${activity.name}`;
    if (activity.phase) {
      const phase = document.createElement("span");
      phase.className = "kc-sub-agent-phase";
      phase.textContent = activity.phase;
      name.append(phase);
    }
    const task = document.createElement("div");
    task.className = "kc-sub-agent-task";
    task.textContent = activity.task || activity.lastText || "";
    row.append(name, task);
    if (activity.notice) {
      const notice = document.createElement("div");
      notice.className = "kc-sub-agent-notice";
      notice.textContent = activity.notice;
      row.append(notice);
    }
  }
  for (const child of Array.from(dom.subAgents.children)) {
    if (!retained.has(child)) child.remove();
  }
  if (active?.isConnected && document.activeElement !== active && !active.disabled) {
    active.focus({ preventScroll: true });
  } else if (active && !active.isConnected) {
    document.querySelector(".kc-agent-back")?.focus({ preventScroll: true });
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
