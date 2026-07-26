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

const TOOL_STATUS = {
  tool_call: "running",
  tool_result: "done",
  tool_error: "failed",
};

export function emptyState() {
  return {
    conversation: [],
    subAgents: new Map(),
    turns: [],
    prompts: [],
    terminal: "",
    terminalTruncated: false,
    logs: [],
    editPreviews: [],
    context: null,
    compaction: null,
    status: "",
    activity: "idle",
    openBrowsers: new Set(),
    lastSeq: 0,
    elapsedMs: 0,
    recordingTruncated: false,
    unknownEventTypes: new Set(),
    streams: new Map(),
    tools: new Map(),
  };
}

function textOf(event, ...keys) {
  for (const key of keys) {
    const value = event.content ? event.content[key] : undefined;
    if (typeof value === "string" && value) return value;
  }
  return "";
}

function subAgentKey(event) {
  const info = event.sub_agent_info;
  if (!info) return null;
  return String(info.dispatch_id || info.agent_name || "sub-agent");
}

function subAgent(state, key, event) {
  let activity = state.subAgents.get(key);
  if (!activity) {
    const info = event.sub_agent_info || {};
    activity = {
      key,
      name: String(info.agent_name || key),
      task: String(info.task || ""),
      status: "running",
      steps: [],
      lastText: "",
      context: null,
    };
    state.subAgents.set(key, activity);
  }
  return activity;
}

function newItem(fields) {
  return {
    kind: fields.kind,
    text: fields.text || "",
    complete: fields.complete === undefined ? true : fields.complete,
    streamId: fields.streamId ?? null,
    toolName: fields.toolName ?? null,
    toolCallId: fields.toolCallId ?? null,
    status: fields.status ?? null,
    tone: fields.tone ?? null,
    subAgent: null,
    seq: null,
    elapsedMs: 0,
    artifacts: fields.artifacts || [],
    editPreview: null,
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
    if (!text && !complete) return;
    append(state, newItem({ kind, text, complete, streamId: event.uuid }), event);
    state.streams.set(streamKey, indexOf(state, event));
    if (complete) state.streams.delete(streamKey);
    return;
  }
  target.text += text;
  target.complete = complete;
  if (complete) state.streams.delete(streamKey);
}

function onToolMessage(state, event, messageType) {
  const toolCallId = String((event.content && event.content.tool_call_id) || "");
  const toolName = String((event.content && event.content.tool_description) || "");
  const text = textOf(event, "text");
  const status = TOOL_STATUS[messageType];
  const key = `${subAgentKey(event) || ""}:${toolCallId}`;
  const index = toolCallId ? state.tools.get(key) : undefined;
  const existing = index === undefined ? null : resolve(state, event, index);
  if (existing && existing.kind === "tool") {
    existing.status = status;
    existing.toolName = existing.toolName || toolName;
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
      toolCallId: toolCallId || null,
      status,
      artifacts: event.artifacts || [],
    }),
    event,
  );
  if (toolCallId && status === "running") state.tools.set(key, indexOf(state, event));
  state.activity = "running_tool";
}

function onChatMessage(state, event) {
  const messageType = String((event.content && event.content.message_type) || "message");
  if (messageType in TOOL_STATUS) {
    onToolMessage(state, event, messageType);
    return;
  }
  const text = textOf(event, "text");
  if (!text) return;
  const kind = messageType === "message" ? "assistant" : "system";
  append(state, newItem({ kind, text, artifacts: event.artifacts || [] }), event);
}

function onToolStreamingUpdate(state, event) {
  const toolCallId = String((event.content && event.content.tool_call_id) || "");
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
        toolCallId: toolCallId || null,
        status: "running",
      }),
      event,
    );
    if (toolCallId) state.tools.set(key, indexOf(state, event));
    return;
  }
  if (String((event.content && event.content.stream_mode) || "") === "replace") {
    target.text = text;
  } else {
    target.text += text;
  }
  if (event.content && event.content.is_complete) target.status = "done";
}

function appendTerminal(state, text) {
  if (!text) return;
  let combined = state.terminal + text;
  if (combined.length > TERMINAL_BUFFER_CHARS) {
    combined = combined.slice(combined.length - TERMINAL_BUFFER_CHARS);
    state.terminalTruncated = true;
  }
  state.terminal = combined;
}

function onStatusUpdate(state, event) {
  const text = textOf(event, "message", "status", "text");
  const key = subAgentKey(event);
  if (key !== null) {
    subAgent(state, key, event).status = text || "running";
    return;
  }
  if (text) state.status = text;
}

const HANDLERS = {
  assistant_delta: (state, event) => onAssistantDelta(state, event, "assistant"),
  thinking_delta: (state, event) => onAssistantDelta(state, event, "thinking"),
  chat_message: onChatMessage,
  tool_streaming_update: onToolStreamingUpdate,
  terminal_command: (state, event) => {
    const command = textOf(event, "command");
    if (command) appendTerminal(state, `$ ${command}\n`);
    state.activity = "running_tool";
  },
  terminal_output: (state, event) => appendTerminal(state, textOf(event, "display_output", "output")),
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
    state.turns.push({
      turn_id: turnId,
      status: "open",
      user_text: userText,
      started_seq: event.seq === undefined ? null : event.seq,
      started_ms: event.elapsed_ms || 0,
      ended_ms: null,
    });
    if (userText) append(state, newItem({ kind: "user", text: userText }), event);
    state.activity = "generating";
  },
  turn_ended: (state, event) => {
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
    ["tool_call_id", item.toolCallId],
    ["status", item.status],
    ["tone", item.tone],
    ["sub_agent", item.subAgent],
    ["edit_preview", item.editPreview],
  ];
  for (const [key, value] of optional) {
    if (value !== null && value !== undefined) payload[key] = value;
  }
  if (item.artifacts && item.artifacts.length) payload.artifacts = item.artifacts.map(artifactDict);
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
      last_text: activity.lastText,
      context: activity.context,
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
