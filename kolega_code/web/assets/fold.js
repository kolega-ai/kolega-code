/**
 * Presentation projection, JavaScript port.
 *
 * This is a deliberate reimplementation of kolega_code/session/projection.py.
 * Two implementations of one rule is a risk, so it is not left to review: the
 * test suite folds shared golden fixtures through both and requires the emitted
 * state to be deeply equal. If you change one, change the other, and the fixture
 * test will tell you if you did not.
 *
 * Exported state matches PresentationState.to_dict() exactly, including which
 * optional keys are omitted when unset.
 */

export const TERMINAL_BUFFER_CHARS = 2000000;
export const LOG_BUFFER_LINES = 2000;
export const TERMINAL_MAX_PARTS = 1024;
export const TERMINAL_MAX_DROPPED_PREFIX = 512 * 1024;

const TOOL_STATUS = {
  tool_call: "running",
  tool_result: "done",
  tool_error: "failed",
};

//: AgentTool's dispatch lifecycle statuses, which carry no transcript text.
const SUB_AGENT_LIFECYCLE = {
  STOPPED: "completed",
  ERROR: "failed",
};

//: Workflow lifecycle messages, and the tone each carries into the transcript.
const WORKFLOW_TONES = {
  workflow_start: "start",
  workflow_phase: "phase",
  workflow_end: "end",
};

export function emptyState() {
  return withTerminal({
    conversation: [],
    subAgents: new Map(),
    turns: [],
    prompts: [],
    logs: [],
    editPreviews: [],
    context: null,
    compaction: null,
    taskList: null,
    usage: null,
    queue: null,
    goal: null,
    status: "",
    activity: "idle",
    openBrowsers: new Set(),
    lastSeq: 0,
    elapsedMs: 0,
    recordingTruncated: false,
    unknownEventTypes: new Set(),
    streams: new Map(),
    tools: new Map(),
    peerMessages: new Map(),
    pendingPeerTurns: [],
    backgroundCompletions: new Set(),
    retiredQueueIds: new Set(),
    retiredGoalGenerationIds: new Set(),
  });
}

function textOf(event, ...keys) {
  for (const key of keys) {
    const value = event.content ? event.content[key] : undefined;
    if (typeof value === "string" && value) return value;
  }
  return "";
}

function stringContent(event, key) {
  const value = event.content ? event.content[key] : undefined;
  return typeof value === "string" ? value : "";
}

// Match Python str.strip(), which includes U+001C–U+001F and excludes the BOM.
const TASK_LIST_WHITESPACE =
  "[\\u0009-\\u000d\\u001c-\\u0020\\u0085\\u00a0\\u1680\\u2000-\\u200a" +
  "\\u2028\\u2029\\u202f\\u205f\\u3000]";
const TASK_LIST_STRIP = new RegExp(`^${TASK_LIST_WHITESPACE}+|${TASK_LIST_WHITESPACE}+$`, "g");

function stripTaskText(text) {
  return text.replace(TASK_LIST_STRIP, "");
}

export function taskListEntries(markdown) {
  const entries = [];
  for (const line of markdown.split(/\r\n|[\n\v\f\r\u001c-\u001e\u0085\u2028\u2029]/)) {
    const stripped = stripTaskText(line);
    let text;
    let status = "pending";
    if (stripped.startsWith("- [x]") || stripped.startsWith("- [X]")) {
      text = stripTaskText(stripped.slice(5));
      status = "completed";
    } else if (stripped.startsWith("- [ ]")) {
      text = stripTaskText(stripped.slice(5));
    } else if (stripped.startsWith("- ")) {
      text = stripTaskText(stripped.slice(2));
    } else {
      continue;
    }
    if (text) entries.push({ text, status });
  }
  if (!entries.length && stripTaskText(markdown)) {
    entries.push({ text: stripTaskText(markdown), status: "pending" });
  }
  return entries;
}

/**
 * Identity of the delegate an event belongs to.
 *
 * `agent_id` matters as much as `dispatch_id`: a workflow leaves `dispatch_id`
 * unset, so keying on the name alone folded every agent of a fan-out into one
 * activity — a dozen parallel agents appearing as a single "general-agent"
 * whose task line was whichever one happened to arrive first.
 */
function subAgentKey(event) {
  const info = event.sub_agent_info;
  if (!isRecord(info) || Object.keys(info).length === 0) return null;
  for (const field of ["dispatch_id", "agent_id", "agent_name"]) {
    if (typeof info[field] === "string" && info[field]) return info[field];
  }
  return "sub-agent";
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function onQueueUpdated(state, event) {
  const content = event.content;
  if (event.sub_agent_info != null || !isRecord(content)) return;
  if (typeof content.queue_id !== "string" || !content.queue_id || !usageCount(content.revision)) return;
  if (!["tui", "ask"].includes(content.source)) return;
  if (!["snapshot", "enqueued", "dequeued", "cleared", "removed"].includes(content.reason)) return;
  if (!Array.isArray(content.items)) return;
  const items = [];
  const itemIds = new Set();
  for (const item of content.items) {
    if (!isRecord(item) || typeof item.queue_id !== "string" || !item.queue_id || itemIds.has(item.queue_id)) return;
    if (typeof item.text !== "string" || !usageCount(item.image_count)) return;
    let origin = item.origin;
    if (origin !== null) {
      if (!isRecord(origin) || origin.kind !== "peer" ||
        typeof origin.session_id !== "string" || typeof origin.title !== "string") return;
      origin = { kind: origin.kind, session_id: origin.session_id, title: origin.title };
    }
    items.push({ queue_id: item.queue_id, text: item.text, image_count: item.image_count, origin });
    itemIds.add(item.queue_id);
  }
  if (state.retiredQueueIds.has(content.queue_id)) return;
  if (state.queue !== null) {
    if (content.queue_id === state.queue.queue_id) {
      if (content.revision <= state.queue.revision) return;
    } else {
      state.retiredQueueIds.add(state.queue.queue_id);
    }
  }
  state.queue = {
    queue_id: content.queue_id,
    revision: content.revision,
    source: content.source,
    reason: content.reason,
    items,
  };
}

function onGoalUpdated(state, event) {
  const content = event.content;
  if (event.sub_agent_info != null || !isRecord(content)) return;
  if (typeof content.generation_id !== "string" || !content.generation_id || !usageCount(content.revision)) return;
  if (!["tui", "ask"].includes(content.source)) return;
  if (!["snapshot", "started", "replaced", "evaluated", "paused", "resumed", "completed", "cleared"]
    .includes(content.reason)) return;
  if (!("goal" in content) || !("evaluation_met" in content)) return;
  if (content.evaluation_met !== null && typeof content.evaluation_met !== "boolean") return;
  let goal = content.goal;
  if (goal === null) {
    if (!["snapshot", "cleared"].includes(content.reason)) return;
  } else {
    if (content.reason === "cleared" || !isRecord(goal)) return;
    const strings = ["condition", "started_at", "last_reason", "status_note"];
    const counts = ["turns_evaluated", "tokens_spent", "max_turns"];
    const flags = ["run_to_completion", "paused", "met"];
    if (!strings.every((key) => typeof goal[key] === "string")) return;
    if (!counts.every((key) => usageCount(goal[key]))) return;
    if (!flags.every((key) => typeof goal[key] === "boolean")) return;
    if (goal.last_evaluated_at !== null && typeof goal.last_evaluated_at !== "string") return;
    goal = Object.fromEntries([...strings, ...counts, ...flags, "last_evaluated_at"].map((key) => [key, goal[key]]));
  }
  if (state.retiredGoalGenerationIds.has(content.generation_id)) return;
  if (state.goal !== null) {
    if (content.generation_id === state.goal.generation_id) {
      if (content.revision <= state.goal.revision) return;
    } else {
      state.retiredGoalGenerationIds.add(state.goal.generation_id);
    }
  }
  state.goal = {
    generation_id: content.generation_id,
    revision: content.revision,
    source: content.source,
    reason: content.reason,
    goal,
    evaluation_met: content.evaluation_met,
  };
}

function onBackgroundDelivery(state, event) {
  const content = event.content || {};
  const required = ["job_id", "completion_id", "agent_type", "status", "task", "text", "disposition"];
  if (event.sub_agent_info != null || !required.every((key) => typeof content[key] === "string")) return;
  if (!["job_id", "completion_id", "agent_type"].every((key) => content[key])) return;
  if (!["completed", "failed", "cancelled", "interrupted"].includes(content.status)) return;
  if (!["context", "archived"].includes(content.disposition)) return;
  const epochId = content.epoch_id ?? null;
  const metadata = content.metadata === undefined ? {} : content.metadata;
  if ((epochId !== null && typeof epochId !== "string") || !isRecord(metadata)) return;
  if (state.backgroundCompletions.has(content.completion_id)) return;
  state.backgroundCompletions.add(content.completion_id);
  const delivery = {};
  for (const key of required) {
    if (key !== "text") delivery[key] = content[key];
  }
  delivery.epoch_id = epochId;
  delivery.metadata = JSON.parse(JSON.stringify(metadata));
  append(state, newItem({
    kind: "background_delivery",
    text: content.text,
    status: content.status,
    backgroundDelivery: delivery,
    artifacts: event.artifacts || [],
  }), event);
}

const USAGE_COUNTS = [
  "requests", "responses", "reported", "unreported", "failed", "open", "abandoned", "partial", "missing",
];
const USAGE_TOKENS = [
  "input_tokens", "output_tokens", "total_tokens", "cache_read_input_tokens",
  "cache_write_input_tokens", "reasoning_output_tokens",
];
const USAGE_DECIMAL = /^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$(?![\s\S])/;

function usageCount(value) {
  return Number.isSafeInteger(value) && value >= 0;
}

function usageDecimal(value) {
  if (typeof value !== "string" || !USAGE_DECIMAL.test(value)) return false;
  const mantissa = value.toLowerCase().split("e")[0];
  return !value.startsWith("-") || !mantissa.slice(1).replace(/[.0]/g, "");
}

function validUsageCosts(costs) {
  if (!isRecord(costs) || typeof costs.complete !== "boolean") return false;
  if (!["reported", "estimated", "missing", "partial", "missing_requests", "partial_requests"]
    .every((name) => usageCount(costs[name]))) return false;
  for (const name of ["currencies", "non_monetary"]) {
    if (!isRecord(costs[name])) return false;
    for (const [denomination, row] of Object.entries(costs[name])) {
      if (!denomination || !isRecord(row)) return false;
      if (!["reported", "estimated", "total"].every((field) => usageDecimal(row[field]))) return false;
      if (!["requests", "reported_requests", "estimated_requests"]
        .every((field) => usageCount(row[field]))) return false;
      if ("token_cost_subtotal" in row && !usageDecimal(row.token_cost_subtotal)) return false;
    }
  }
  return true;
}

function validUsage(usage) {
  return isRecord(usage) && typeof usage.complete === "boolean" &&
    [...USAGE_COUNTS, ...USAGE_TOKENS].every((name) => usageCount(usage[name])) &&
    isRecord(usage.known_requests) &&
    USAGE_TOKENS.every((name) => usageCount(usage.known_requests[name])) &&
    validUsageCosts(usage.costs);
}

function subAgent(state, key, event) {
  let activity = state.subAgents.get(key);
  if (!activity) {
    activity = withTerminal({
      key,
      name: key,
      task: "",
      status: "running",
      notice: "",
      steps: [],
      lastText: "",
      context: null,
      label: "",
      phase: "",
      workflowRunId: "",
      agentId: null,
      parentAgentId: null,
      parentToolCallId: null,
      depth: 1,
      background: false,
      requestedRouting: null,
      effectiveRouting: null,
      totalTokens: null,
    });
    state.subAgents.set(key, activity);
  }
  return activity;
}

function mergeSubAgentMetadata(activity, event) {
  const info = event.sub_agent_info || {};
  for (const [source, target] of [
    ["agent_name", "name"],
    ["task", "task"],
    ["label", "label"],
    ["phase", "phase"],
    ["workflow_run_id", "workflowRunId"],
    ["agent_id", "agentId"],
    ["parent_agent_id", "parentAgentId"],
    ["parent_tool_call_id", "parentToolCallId"],
  ]) {
    if (typeof info[source] === "string" && info[source]) activity[target] = info[source];
  }
  if (Number.isInteger(info.depth) && info.depth >= 1) activity.depth = info.depth;
  if (typeof info.background === "boolean") activity.background = info.background;
  for (const [source, target] of [
    ["requested_routing", "requestedRouting"],
    ["effective_routing", "effectiveRouting"],
  ]) {
    if (isRecord(info[source])) activity[target] = { ...activity[target], ...info[source] };
  }
  for (const source of [info, event.content || {}]) {
    if (Number.isInteger(source.total_tokens) && source.total_tokens >= 0) {
      activity.totalTokens = source.total_tokens;
    }
  }
}

function newItem(fields) {
  return {
    kind: fields.kind,
    text: fields.text || "",
    complete: fields.complete === undefined ? true : fields.complete,
    streamId: fields.streamId ?? null,
    toolName: fields.toolName ?? null,
    toolSubject: fields.toolSubject ?? null,
    toolCallId: fields.toolCallId ?? null,
    status: fields.status ?? null,
    tone: fields.tone ?? null,
    subAgent: null,
    seq: null,
    elapsedMs: 0,
    artifacts: fields.artifacts || [],
    imageCount: fields.imageCount || 0,
    editPreview: null,
    backgroundDelivery: fields.backgroundDelivery ?? null,
  };
}

function append(state, item, event) {
  item.seq = event.seq === undefined ? null : event.seq;
  item.elapsedMs = event.elapsed_ms || 0;
  const key = subAgentKey(event);
  if (key !== null) {
    item.subAgent = key;
    const activity = subAgent(state, key, event);
    activity.steps.push(item);
    if (item.text) activity.lastText = item.text;
    return item;
  }
  state.conversation.push(item);
  return item;
}

function sequenceFor(state, event) {
  const key = subAgentKey(event);
  if (key !== null && state.subAgents.has(key)) return state.subAgents.get(key).steps;
  return state.conversation;
}

function indexOf(state, event) {
  return sequenceFor(state, event).length - 1;
}

function resolve(state, event, index) {
  const sequence = sequenceFor(state, event);
  if (index >= 0 && index < sequence.length) return sequence[index];
  return null;
}

function onAssistantDelta(state, event, kind) {
  const text = textOf(event, "text");
  const complete = Boolean(event.content && event.content.complete);
  const streamKey = `${kind}:${subAgentKey(event) || ""}:${event.uuid}`;
  const index = state.streams.get(streamKey);
  const target = index === undefined ? null : resolve(state, event, index);
  if (!target) {
    // A segment that never carried text produces nothing. The agent closes every
    // iteration with a final empty prose delta even when it only called a tool,
    // and rendering that as an entry puts a bare glyph with no content into the
    // transcript.
    if (!text) return;
    append(state, newItem({ kind, text, complete, streamId: event.uuid }), event);
    state.streams.set(streamKey, indexOf(state, event));
    if (complete) state.streams.delete(streamKey);
    return;
  }
  target.text += text;
  target.complete = complete;
  if (complete) state.streams.delete(streamKey);
}

function toolSubjectOf(event) {
  // Already sanitized by the producer. Match Python without coercing objects
  // or letting missing/empty metadata clear an established subject.
  const subject = event.content && event.content.tool_subject;
  return typeof subject === "string" && subject ? subject : null;
}

function onToolMessage(state, event, messageType) {
  const toolCallId = String((event.content && event.content.tool_call_id) || "");
  const toolName = String((event.content && event.content.tool_description) || "");
  const toolSubject = toolSubjectOf(event);
  const text = textOf(event, "text");
  const status = TOOL_STATUS[messageType];
  const key = `${subAgentKey(event) || ""}:${toolCallId}`;
  const index = toolCallId ? state.tools.get(key) : undefined;
  const existing = index === undefined ? null : resolve(state, event, index);
  if (existing && existing.kind === "tool") {
    existing.status = status;
    existing.toolName = existing.toolName || toolName;
    if (toolSubject !== null) existing.toolSubject = toolSubject;
    if (text) existing.text = text;
    if (event.artifacts && event.artifacts.length) existing.artifacts = event.artifacts;
    if (status !== "running") state.tools.delete(key);
    return;
  }
  append(
    state,
    newItem({
      kind: "tool",
      text,
      toolName,
      toolSubject,
      toolCallId: toolCallId || null,
      status,
      artifacts: event.artifacts || [],
    }),
    event,
  );
  if (toolCallId && status === "running") state.tools.set(key, indexOf(state, event));
  state.activity = "running_tool";
}

/**
 * Render a gigacode run's own lifecycle.
 *
 * `workflow_start` and `workflow_end` carry their detail in named fields and
 * leave `text` empty, so the generic path dropped them outright: a workflow
 * appeared as a few unlabelled lines with no name, no phases, and no outcome,
 * while its agents' work was merged into the transcript around them.
 */
function onWorkflowMessage(state, event, messageType) {
  const content = event.content || {};
  const tone = WORKFLOW_TONES[messageType];
  let status = null;
  let text;
  if (messageType === "workflow_start") {
    const name = String(content.name || "workflow");
    const description = String(content.description || "");
    text = description ? `${name} — ${description}` : name;
    status = "running";
  } else if (messageType === "workflow_end") {
    status = String(content.status || "completed");
    const error = String(content.error || "");
    text = error ? `${status}: ${error}` : status;
  } else {
    text = textOf(event, "text");
    if (!text) return;
  }
  append(
    state,
    newItem({
      kind: "workflow",
      text,
      tone,
      status,
      toolCallId: String(content.workflow_run_id || "") || null,
    }),
    event,
  );
}

function onChatMessage(state, event) {
  const messageType = String((event.content && event.content.message_type) || "message");
  if (messageType in TOOL_STATUS) {
    onToolMessage(state, event, messageType);
    return;
  }
  if (messageType in WORKFLOW_TONES) {
    onWorkflowMessage(state, event, messageType);
    return;
  }
  const lifecycle = event.content ? event.content.status : undefined;
  const key = subAgentKey(event);
  if (lifecycle !== undefined && lifecycle !== null && key !== null) {
    // Dispatch lifecycle, not conversation: without it a finished sub-agent keeps
    // reporting "running" for the rest of the replay.
    subAgent(state, key, event).status = SUB_AGENT_LIFECYCLE[String(lifecycle)] || "running";
    return;
  }
  const text = textOf(event, "text");
  if (!text) return;
  const kind = messageType === "message" ? "assistant" : "system";
  append(state, newItem({ kind, text, artifacts: event.artifacts || [] }), event);
}

function onToolStreamingUpdate(state, event) {
  const toolCallId = String((event.content && event.content.tool_call_id) || "");
  const toolSubject = toolSubjectOf(event);
  const key = `${subAgentKey(event) || ""}:${toolCallId}`;
  const index = state.tools.get(key);
  let target = index === undefined ? null : resolve(state, event, index);
  const text = textOf(event, "text");
  if (!target) {
    target = append(
      state,
      newItem({
        kind: "tool",
        text,
        toolName: String((event.content && event.content.tool_name) || ""),
        toolSubject,
        toolCallId: toolCallId || null,
        status: "running",
      }),
      event,
    );
    if (toolCallId) state.tools.set(key, indexOf(state, event));
    return;
  }
  if (toolSubject !== null) target.toolSubject = toolSubject;
  if (String((event.content && event.content.stream_mode) || "") === "replace") {
    target.text = text;
  } else {
    target.text += text;
  }
  if (event.content && event.content.is_complete) target.status = "done";
}

function withTerminal(target) {
  target.terminalTruncated = false;
  target._terminalParts = [];
  target._terminalHead = 0;
  target._terminalOffset = 0;
  target._terminalChars = 0;
  target._terminalText = "";
  Object.defineProperty(target, "terminal", {
    enumerable: true,
    get() {
      if (this._terminalText === null) {
        const text = this._terminalParts.slice(this._terminalHead).join("");
        this._terminalText = this._terminalOffset ? text.slice(this._terminalOffset) : text;
      }
      return this._terminalText;
    },
    set(value) {
      this._terminalParts = value ? [value] : [];
      this._terminalHead = 0;
      this._terminalOffset = 0;
      this._terminalChars = value.length;
      this._terminalText = value;
    },
  });
  return target;
}

function appendTerminal(state, text) {
  if (!text) return;
  const parts = state._terminalParts;
  parts.push(text);
  state._terminalChars += text.length;
  state._terminalText = null;
  let excess = state._terminalChars - TERMINAL_BUFFER_CHARS;
  if (excess > 0) {
    state.terminalTruncated = true;
    while (excess > 0) {
      const available = parts[state._terminalHead].length - state._terminalOffset;
      if (available <= excess) {
        // Release consumed chunks immediately; shifting the array would copy it on every trim.
        parts[state._terminalHead++] = null;
        state._terminalOffset = 0;
        state._terminalChars -= available;
        excess -= available;
      } else {
        state._terminalOffset += excess;
        state._terminalChars -= excess;
        excess = 0;
      }
    }
  }
  if (parts.length > TERMINAL_MAX_PARTS || state._terminalOffset > TERMINAL_MAX_DROPPED_PREFIX) {
    const retained = parts.slice(state._terminalHead).join("").slice(state._terminalOffset);
    state._terminalParts = retained ? [retained] : [];
    state._terminalHead = 0;
    state._terminalOffset = 0;
  }
}

function terminalTarget(state, event) {
  const key = subAgentKey(event);
  return key === null ? state : state.subAgents.get(key);
}

function onStatusUpdate(state, event) {
  const text = textOf(event, "message", "status", "text");
  const key = subAgentKey(event);
  if (key !== null) {
    const activity = subAgent(state, key, event);
    if (text) activity.notice = text;
    return;
  }
  if (text) state.status = text;
}

function onPeerMessage(state, event, status) {
  const messageId = stringContent(event, "message_id");
  const senderId = stringContent(event, "sender_id");
  const senderTitle = stringContent(event, "sender_title");
  const body = textOf(event, "text");
  const record = messageId ? state.peerMessages.get(messageId) : undefined;
  if (record) {
    const wasDelivered = record.delivered;
    if (senderId) record.senderId = senderId;
    if (senderTitle) record.sender = senderTitle;
    else if (senderId && record.sender === "peer") record.sender = senderId;
    if (body) record.body = body;
    record.delivered = record.delivered || status === "delivered";
    const existing =
      record.index >= 0 && record.index < state.conversation.length
        ? state.conversation[record.index]
        : null;
    if (existing && existing.kind === "peer_message") {
      existing.text = peerDisplayText(record.sender, record.body);
      existing.toolName = record.sender;
      existing.status = record.delivered ? "delivered" : "received";
      if (record.delivered && !wasDelivered) state.pendingPeerTurns.push(messageId);
      return;
    }
  }
  const sender = senderTitle || senderId || "peer";
  const item = newItem({
    kind: "peer_message",
    text: peerDisplayText(sender, body),
    toolName: sender,
    toolCallId: messageId || null,
    status,
  });
  item.seq = event.seq === undefined ? null : event.seq;
  item.elapsedMs = event.elapsed_ms || 0;
  state.conversation.push(item);
  if (!messageId) return;
  state.peerMessages.set(messageId, {
    index: state.conversation.length - 1,
    senderId,
    sender,
    body,
    delivered: status === "delivered",
  });
  if (status === "delivered") state.pendingPeerTurns.push(messageId);
}

function peerDisplayText(sender, body) {
  return body ? `← ${sender}\n${body}` : `← ${sender}`;
}

function consumePeerTurn(state, userText) {
  for (let index = 0; index < state.pendingPeerTurns.length; index += 1) {
    const messageId = state.pendingPeerTurns[index];
    const record = state.peerMessages.get(messageId);
    if (!record || !record.senderId || !record.body) continue;
    const prefix = `[Peer message from session '${record.sender}' (id ${record.senderId.slice(0, 8)})]`;
    if (userText.startsWith(prefix) && userText.endsWith(`\n\n${record.body}`)) {
      state.pendingPeerTurns.splice(index, 1);
      return peerDisplayText(record.sender, record.body);
    }
  }
  return null;
}

const HANDLERS = {
  assistant_delta: (state, event) => onAssistantDelta(state, event, "assistant"),
  thinking_delta: (state, event) => onAssistantDelta(state, event, "thinking"),
  chat_message: onChatMessage,
  tool_streaming_update: onToolStreamingUpdate,
  terminal_command: (state, event) => {
    const command = textOf(event, "command");
    if (command) appendTerminal(terminalTarget(state, event), `$ ${command}\n`);
    if (subAgentKey(event) === null) state.activity = "running_tool";
  },
  terminal_output: (state, event) =>
    appendTerminal(terminalTarget(state, event), textOf(event, "display_output", "output")),
  terminal_launched: () => {},
  terminal_closed: () => {},
  log_message: (state, event) => {
    const text = textOf(event, "text", "message");
    if (!text) return;
    const level = String((event.content && event.content.level) || "info");
    state.logs.push({ level, text });
    if (state.logs.length > LOG_BUFFER_LINES) {
      state.logs.splice(0, state.logs.length - LOG_BUFFER_LINES);
    }
  },
  file_edit_preview: (state, event) => {
    const preview = Object.assign({}, event.content);
    preview.seq = event.seq === undefined ? null : event.seq;
    preview.elapsed_ms = event.elapsed_ms || 0;
    const key = subAgentKey(event);
    if (key !== null) preview.sub_agent = key;
    state.editPreviews.push(preview);
    const toolCallId = String((event.content && event.content.tool_call_id) || "");
    if (toolCallId) {
      const index = state.tools.get(`${key || ""}:${toolCallId}`);
      const target = index === undefined ? null : resolve(state, event, index);
      if (target) target.editPreview = preview;
    }
  },
  llm_context_update: (state, event) => {
    const content = event.content || {};
    const key = subAgentKey(event);
    if (key !== null) {
      subAgent(state, key, event).context = Object.assign({}, content);
      return;
    }
    state.context = {
      input_tokens: Number(content.input_tokens || 0),
      max_tokens: Number(content.max_tokens || 0),
      usage_percentage: Number(content.usage_percentage || 0),
      alert_level: String(content.alert_level || ""),
      message: content.message === undefined ? null : content.message,
      will_compress_at: Number(content.will_compress_at || 0),
    };
  },
  compaction_status: (state, event) => {
    if (subAgentKey(event) !== null) return;
    state.compaction = Object.assign({}, event.content);
  },
  llm_status_update: onStatusUpdate,
  status_update: onStatusUpdate,
  credit_alert: onStatusUpdate,
  turn_started: (state, event) => {
    const turnId = String((event.content && event.content.turn_id) || "");
    const userText = String((event.content && event.content.user_text) || "");
    const count = event.content?.image_count;
    const imageCount = Number.isInteger(count) && count > 0 ? count : 0;
    const images = (event.artifacts || []).filter(
      (ref) => ref.purpose === "image" && String(ref.media_type || "").startsWith("image/"),
    );
    // A dispatched agent runs its own turns. They are not session turns: indexing
    // them would put work nobody typed into the turn rail, and letting them drive
    // the activity flag would report the session idle while the main agent is
    // still going. The task text still opens the sub-agent's own trajectory.
    if (subAgentKey(event) === null) {
      const peerText = consumePeerTurn(state, userText);
      state.turns.push({
        turn_id: turnId,
        status: "open",
        user_text: peerText || userText,
        started_seq: event.seq === undefined ? null : event.seq,
        started_ms: event.elapsed_ms || 0,
        ended_ms: null,
      });
      state.activity = "generating";
      if (peerText !== null) return;
    }
    if (userText || images.length || imageCount) {
      append(state, newItem({ kind: "user", text: userText, artifacts: images, imageCount }), event);
    }
  },
  turn_ended: (state, event) => {
    if (subAgentKey(event) !== null) return;
    const turnId = String((event.content && event.content.turn_id) || "");
    const status = String((event.content && event.content.status) || "completed");
    for (let index = state.turns.length - 1; index >= 0; index -= 1) {
      if (state.turns[index].turn_id === turnId) {
        state.turns[index].status = status;
        state.turns[index].ended_ms = event.elapsed_ms || 0;
        break;
      }
    }
    state.activity = "idle";
  },
  browser_launched: (state, event) => {
    const id = String((event.content && event.content.browser_id) || "");
    if (id) state.openBrowsers.add(id);
  },
  browser_closed: (state, event) => {
    state.openBrowsers.delete(String((event.content && event.content.browser_id) || ""));
  },
  stream_truncated: (state, event) => {
    state.recordingTruncated = true;
    append(
      state,
      newItem({
        kind: "system",
        text: "Recording stopped here because this session reached its retention limit.",
        tone: "warning",
      }),
      event,
    );
  },
  system_message: (state, event) => {
    const text = textOf(event, "text", "message");
    if (text) append(state, newItem({ kind: "system", text }), event);
  },
  memory_suggestions: () => {},
  llm_error: () => {},
  llm_request: () => {},
  control_requested: (state, event) => {
    const requestId = String((event.content && event.content.request_id) || "");
    if (!requestId || state.prompts.some((prompt) => prompt.request_id === requestId)) return;
    state.prompts.push({
      request_id: requestId,
      kind: String((event.content && event.content.kind) || "prompt"),
      payload: Object.assign({}, (event.content && event.content.payload) || {}),
      response: null,
      reason: null,
      seq: event.seq === undefined ? null : event.seq,
      elapsed_ms: event.elapsed_ms || 0,
    });
    state.activity = "waiting_for_user";
  },
  control_resolved: (state, event) => {
    const requestId = String((event.content && event.content.request_id) || "");
    for (let index = state.prompts.length - 1; index >= 0; index -= 1) {
      if (state.prompts[index].request_id === requestId) {
        const response = event.content && event.content.response;
        state.prompts[index].response =
          response && typeof response === "object" ? Object.assign({}, response) : null;
        state.prompts[index].reason = String((event.content && event.content.reason) || "answered");
        break;
      }
    }
    if (!state.prompts.some((prompt) => prompt.reason === null)) state.activity = "generating";
  },
  peer_message_received: (state, event) => onPeerMessage(state, event, "received"),
  peer_message_delivered: (state, event) => onPeerMessage(state, event, "delivered"),
};

export function fold(state, event) {
  if (event.seq !== null && event.seq !== undefined) {
    if (event.seq <= state.lastSeq) {
      throw new Error(
        `Event seq ${event.seq} is not after the last applied seq ${state.lastSeq}; ` +
          "the projection requires ascending order",
      );
    }
    state.lastSeq = event.seq;
  }
  state.elapsedMs = Math.max(state.elapsedMs, event.elapsed_ms || 0);

  if (event.event_type === "queue_updated") {
    onQueueUpdated(state, event);
    return state;
  }

  if (event.event_type === "goal_updated") {
    onGoalUpdated(state, event);
    return state;
  }

  if (event.event_type === "background_job_delivered") {
    onBackgroundDelivery(state, event);
    return state;
  }

  if (event.event_type === "usage_updated") {
    const content = event.content || {};
    if (subAgentKey(event) === null && content.scope === "session" &&
      content.semantics === "snapshot" && usageCount(content.source_seq) &&
      validUsage(content.usage) && (state.usage === null || content.source_seq >= state.usage.source_seq)) {
      state.usage = { ...JSON.parse(JSON.stringify(content.usage)), source_seq: content.source_seq };
    }
    return state;
  }

  if (
    event.event_type === "task_list_updated" ||
    (event.event_type === "chat_message" && event.content?.message_type === "task_list_update")
  ) {
    if (subAgentKey(event) === null) {
      const field = event.event_type === "task_list_updated" ? "markdown" : "text";
      const markdown = event.content?.[field];
      if (typeof markdown === "string") {
        state.taskList = { markdown, entries: taskListEntries(markdown) };
      }
    }
    return state;
  }

  // Peer lifecycle is session-scoped even when a producer attaches stale worker metadata.
  if (!["peer_message_received", "peer_message_delivered"].includes(event.event_type)) {
    const key = subAgentKey(event);
    if (key !== null) mergeSubAgentMetadata(subAgent(state, key, event), event);
  }

  const handler = HANDLERS[event.event_type];
  if (!handler) {
    state.unknownEventTypes.add(event.event_type);
    return state;
  }
  handler(state, event);
  return state;
}

export function replay(events) {
  const state = emptyState();
  for (const event of events) fold(state, event);
  return state;
}

/**
 * Drop null/undefined members, matching Pydantic's exclude_none=True.
 *
 * The wire form of an event keeps unset artifact fields as explicit nulls, but
 * the projection omits them, so they must be stripped here or the two folds
 * disagree on artifact shape.
 */
function artifactDict(ref) {
  const payload = {};
  for (const [key, value] of Object.entries(ref)) {
    if (value !== null && value !== undefined) payload[key] = value;
  }
  return payload;
}

function itemDict(item) {
  const payload = {
    kind: item.kind,
    text: item.text,
    complete: item.complete,
    seq: item.seq,
    elapsed_ms: item.elapsedMs,
  };
  const optional = [
    ["stream_id", item.streamId],
    ["tool_name", item.toolName],
    ["tool_subject", item.toolSubject],
    ["tool_call_id", item.toolCallId],
    ["status", item.status],
    ["tone", item.tone],
    ["sub_agent", item.subAgent],
    ["edit_preview", item.editPreview],
    ["background_delivery", item.backgroundDelivery],
  ];
  for (const [key, value] of optional) {
    if (value !== null && value !== undefined) payload[key] = value;
  }
  if (item.artifacts && item.artifacts.length) payload.artifacts = item.artifacts.map(artifactDict);
  if (item.imageCount) payload.image_count = item.imageCount;
  return payload;
}

/** Mirrors PresentationState.to_dict() so both folds can be compared directly. */
export function toDict(state) {
  const subAgents = {};
  for (const [key, activity] of state.subAgents.entries()) {
    subAgents[key] = {
      key: activity.key,
      name: activity.name,
      task: activity.task,
      status: activity.status,
      notice: activity.notice,
      last_text: activity.lastText,
      context: activity.context,
      label: activity.label,
      phase: activity.phase,
      workflow_run_id: activity.workflowRunId,
      agent_id: activity.agentId,
      parent_agent_id: activity.parentAgentId,
      parent_tool_call_id: activity.parentToolCallId,
      depth: activity.depth,
      background: activity.background,
      requested_routing: activity.requestedRouting,
      effective_routing: activity.effectiveRouting,
      total_tokens: activity.totalTokens,
      terminal: activity.terminal,
      terminal_truncated: activity.terminalTruncated,
      steps: activity.steps.map(itemDict),
    };
  }
  return {
    conversation: state.conversation.map(itemDict),
    sub_agents: subAgents,
    turns: state.turns.map((marker) => ({
      turn_id: marker.turn_id,
      status: marker.status,
      user_text: marker.user_text,
      started_seq: marker.started_seq,
      started_ms: marker.started_ms,
      ended_ms: marker.ended_ms,
    })),
    terminal: state.terminal,
    terminal_truncated: state.terminalTruncated,
    logs: state.logs.map((entry) => ({ level: entry.level, text: entry.text })),
    edit_previews: state.editPreviews,
    context: state.context,
    compaction: state.compaction,
    task_list: state.taskList,
    usage: state.usage,
    queue: state.queue === null ? null : JSON.parse(JSON.stringify(state.queue)),
    goal: state.goal === null ? null : JSON.parse(JSON.stringify(state.goal)),
    prompts: state.prompts.map((prompt) => ({
      request_id: prompt.request_id,
      kind: prompt.kind,
      payload: prompt.payload,
      response: prompt.response,
      reason: prompt.reason,
      resolved: prompt.reason !== null,
      seq: prompt.seq,
      elapsed_ms: prompt.elapsed_ms,
    })),
    status: state.status,
    activity: state.activity,
    open_browsers: Array.from(state.openBrowsers).sort(),
    last_seq: state.lastSeq,
    elapsed_ms: state.elapsedMs,
    recording_truncated: state.recordingTruncated,
    unknown_event_types: Array.from(state.unknownEventTypes).sort(),
  };
}
