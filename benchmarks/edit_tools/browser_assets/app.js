(function () {
  "use strict";

  const data = window.BENCHMARK_DATA;
  if (!data || !Array.isArray(data.tasks)) {
    document.body.textContent = "Benchmark data could not be loaded.";
    return;
  }

  const elements = {
    description: document.getElementById("suite-description"),
    summary: document.getElementById("summary-strip"),
    resultCount: document.getElementById("result-count"),
    taskList: document.getElementById("task-list"),
    taskDetail: document.getElementById("task-detail"),
    search: document.getElementById("task-search"),
    language: document.getElementById("language-filter"),
    family: document.getElementById("family-filter"),
    difficulty: document.getElementById("difficulty-filter"),
    length: document.getElementById("length-filter"),
    clear: document.getElementById("clear-filters"),
  };

  const state = {
    selectedId: null,
    selectedFile: null,
    view: "diff",
    filtered: [],
  };

  const difficultyOrder = ["easy", "medium", "hard"];
  const lengthOrder = ["short", "normal", "medium", "long", "oversized"];

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function displayName(value) {
    if (!value) return "—";
    return String(value)
      .replaceAll("-", " ")
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function repositoryName(url) {
    if (!url) return "Pinned source";
    const parts = url.split("/").filter(Boolean);
    return parts.slice(-2).join("/");
  }

  function taskSearchText(task) {
    return [
      task.id,
      task.prompt,
      task.language,
      task.family,
      task.difficulty,
      task.target_length,
      task.primary_target,
      ...(task.tags || []),
      ...(task.workspace_files || []),
      ...(task.operations || []).flatMap((operation) => [operation.kind, operation.path]),
      task.snapshot && task.snapshot.repository,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
  }

  for (const task of data.tasks) {
    task._search = taskSearchText(task);
  }

  function addOptions(select, values, order) {
    const unique = [...new Set(values.filter(Boolean))];
    unique.sort((left, right) => {
      if (order) return order.indexOf(left) - order.indexOf(right);
      return left.localeCompare(right);
    });
    for (const value of unique) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = displayName(value);
      select.append(option);
    }
  }

  function renderHeader() {
    const suiteName = data.suite.id === "edit-core" ? "Edit corpus v1" : `${data.suite.id} corpus`;
    document.title = suiteName;
    elements.description.textContent = data.suite.description;
    const items = [
      [data.summary.tasks, "tasks"],
      [data.summary.languages, "languages"],
      [data.summary.operations, "operations"],
      [data.summary.repositories, "repositories"],
    ];
    elements.summary.innerHTML = items
      .map(
        ([value, label]) => `
          <div class="summary-item">
            <span class="summary-value">${escapeHtml(value)}</span>
            <span class="summary-label">${escapeHtml(label)}</span>
          </div>`,
      )
      .join("");
  }

  function currentFilters() {
    return {
      query: elements.search.value.trim().toLowerCase(),
      language: elements.language.value,
      family: elements.family.value,
      difficulty: elements.difficulty.value,
      length: elements.length.value,
    };
  }

  function filterTasks() {
    const filters = currentFilters();
    state.filtered = data.tasks.filter((task) => {
      return (
        (!filters.query || task._search.includes(filters.query)) &&
        (!filters.language || task.language === filters.language) &&
        (!filters.family || task.family === filters.family) &&
        (!filters.difficulty || task.difficulty === filters.difficulty) &&
        (!filters.length || task.target_length === filters.length)
      );
    });
  }

  function taskCard(task) {
    const selected = task.id === state.selectedId;
    return `
      <button
        class="task-card"
        type="button"
        role="option"
        aria-selected="${selected}"
        data-task-id="${escapeHtml(task.id)}"
        title="${escapeHtml(task.id)}"
      >
        <span class="task-number">${String(task.index + 1).padStart(3, "0")}</span>
        <span>
          <span class="task-name">${escapeHtml(task.id)}</span>
          <span class="task-card-meta">
            <span>${escapeHtml(task.language)}</span>
            <span>${escapeHtml(task.operation_count)} ops</span>
            <span>${escapeHtml(task.changed_file_count)} files</span>
          </span>
        </span>
      </button>`;
  }

  function renderTaskList() {
    elements.resultCount.textContent = `${state.filtered.length} of ${data.tasks.length}`;
    elements.taskList.innerHTML = state.filtered.map(taskCard).join("");
    for (const button of elements.taskList.querySelectorAll("[data-task-id]")) {
      button.addEventListener("click", () => selectTask(button.dataset.taskId));
    }
  }

  function renderEmptyState() {
    const template = document.getElementById("empty-state-template");
    elements.taskDetail.replaceChildren(template.content.cloneNode(true));
  }

  function refreshFilters() {
    filterTasks();
    if (!state.filtered.length) {
      state.selectedId = null;
      renderTaskList();
      renderEmptyState();
      return;
    }
    if (!state.filtered.some((task) => task.id === state.selectedId)) {
      state.selectedId = state.filtered[0].id;
      state.selectedFile = null;
    }
    renderTaskList();
    renderDetail();
  }

  function selectedTask() {
    return data.tasks.find((task) => task.id === state.selectedId) || null;
  }

  function selectTask(id, updateUrl = true) {
    const task = data.tasks.find((item) => item.id === id);
    if (!task) return;
    state.selectedId = task.id;
    state.selectedFile = task.primary_target || task.files.find((file) => file.changed)?.path || task.files[0]?.path;
    state.view = "diff";
    if (updateUrl) {
      history.replaceState(null, "", `#${encodeURIComponent(task.id)}`);
    }
    renderTaskList();
    renderDetail();
    const selectedButton = elements.taskList.querySelector('[aria-selected="true"]');
    selectedButton?.scrollIntoView({ block: "nearest" });
  }

  function chip(value, className = "") {
    if (!value) return "";
    return `<span class="chip ${className}">${escapeHtml(displayName(value))}</span>`;
  }

  function sourcePanel(task) {
    const snapshot = task.snapshot;
    const authoring = task.authoring || {};
    const sourceRows = snapshot && snapshot.repository
      ? `
        <div class="source-kv">
          <dt>Repository</dt>
          <dd><a href="${escapeHtml(snapshot.repository)}" target="_blank" rel="noreferrer">${escapeHtml(repositoryName(snapshot.repository))}</a></dd>
        </div>
        <div class="source-kv">
          <dt>Commit</dt>
          <dd><a href="${escapeHtml(snapshot.commit_url)}" target="_blank" rel="noreferrer">${escapeHtml(snapshot.commit.slice(0, 12))}</a></dd>
        </div>
        <div class="source-kv"><dt>License</dt><dd>${escapeHtml(snapshot.license)}</dd></div>`
      : `<div class="source-kv"><dt>Source</dt><dd>${escapeHtml(displayName(task.provenance))}</dd></div>`;
    const authoringRows = authoring.model
      ? `
        <div class="source-kv"><dt>Authored by</dt><dd>${escapeHtml(authoring.model)}</dd></div>
        <div class="source-kv"><dt>Surface</dt><dd>${escapeHtml(authoring.protocol || "—")}</dd></div>`
      : "";
    return `
      <section class="panel">
        <div class="panel-header">
          <h3 class="panel-title">Provenance</h3>
          <span class="panel-count">${escapeHtml(task.snapshot?.id || task.provenance)}</span>
        </div>
        <dl class="source-body">
          ${sourceRows}
          <div class="source-kv"><dt>Primary file</dt><dd>${escapeHtml(task.primary_target || "—")}</dd></div>
          ${authoringRows}
        </dl>
      </section>`;
  }

  function operationCard(operation) {
    let location = "new file";
    if (operation.start_line != null && operation.end_line != null) {
      location = operation.start_line === operation.end_line
        ? `line ${operation.start_line}`
        : `lines ${operation.start_line}–${operation.end_line}`;
    } else if (operation.start_line != null) {
      location = `after line ${operation.start_line}`;
    }
    const preview = operation.new_text
      ? `<pre class="operation-preview">${escapeHtml(operation.new_text)}</pre>`
      : "";
    return `
      <article class="operation-card">
        <div class="operation-card-top">
          <span class="operation-kind">${escapeHtml(operation.kind)}</span>
          <span class="operation-id">${escapeHtml(operation.id)}</span>
        </div>
        <p class="operation-path" title="${escapeHtml(operation.path)}">${escapeHtml(operation.path)}</p>
        <span class="operation-location">${escapeHtml(location)}</span>
        ${preview}
      </article>`;
  }

  function operationsPanel(task) {
    if (!task.operations.length) return "";
    return `
      <section class="panel operation-panel">
        <div class="panel-header">
          <h3 class="panel-title">Exact operation recipe</h3>
          <span class="panel-count">${task.operations.length} operation${task.operations.length === 1 ? "" : "s"}</span>
        </div>
        <div class="operation-grid">${task.operations.map(operationCard).join("")}</div>
      </section>`;
  }

  function fileOption(file) {
    const marker = file.status === "created" ? "A" : file.status === "deleted" ? "D" : file.changed ? "M" : "·";
    return `<option value="${escapeHtml(file.path)}" ${file.path === state.selectedFile ? "selected" : ""}>[${marker}] ${escapeHtml(file.path)}</option>`;
  }

  function filesPanel(task) {
    return `
      <section class="panel files-panel">
        <div class="panel-header">
          <div class="file-toolbar">
            <label>
              <span class="visually-hidden">Select workspace file</span>
              <select class="file-picker" id="file-picker">${task.files.map(fileOption).join("")}</select>
            </label>
            <div class="view-tabs" role="tablist" aria-label="File comparison view">
              ${["diff", "before", "expected"]
                .map(
                  (view) => `<button class="view-tab" type="button" role="tab" data-view="${view}" aria-selected="${state.view === view}">${displayName(view)}</button>`,
                )
                .join("")}
            </div>
          </div>
        </div>
        <div class="file-meta-bar" id="file-meta-bar"></div>
        <div class="code-viewport" id="code-viewport" tabindex="0" aria-label="File content"></div>
      </section>`;
  }

  function renderDetail() {
    const task = selectedTask();
    if (!task) {
      renderEmptyState();
      return;
    }
    if (!task.files.some((file) => file.path === state.selectedFile)) {
      state.selectedFile = task.primary_target || task.files[0]?.path;
    }
    const filteredIndex = state.filtered.findIndex((item) => item.id === task.id);
    const allPosition = task.index + 1;
    const tags = [
      chip(task.language, "chip-language"),
      chip(task.difficulty, `chip-${task.difficulty}`),
      chip(task.family),
      chip(task.target_length),
      chip(`${task.operation_count} operations`),
      chip(`${task.changed_file_count} files`),
    ].join("");
    elements.taskDetail.innerHTML = `
      <div class="detail-shell">
        <div class="detail-topline">
          <span class="task-position">Case ${String(allPosition).padStart(3, "0")} / ${data.tasks.length}</span>
          <div class="detail-nav">
            <button class="nav-button" id="previous-task" type="button" ${filteredIndex <= 0 ? "disabled" : ""}>← Previous</button>
            <button class="nav-button" id="next-task" type="button" ${filteredIndex >= state.filtered.length - 1 ? "disabled" : ""}>Next →</button>
          </div>
        </div>
        <div class="task-heading-row">
          <h2 class="task-heading">${escapeHtml(task.id)}</h2>
          <button class="copy-button" id="copy-task-id" type="button">Copy ID</button>
        </div>
        <div class="chip-row">${tags}</div>
        <div class="overview-grid">
          <section class="panel">
            <div class="panel-header">
              <h3 class="panel-title">Model instruction</h3>
              <button class="quiet-button" id="copy-prompt" type="button">Copy prompt</button>
            </div>
            <pre class="prompt-body">${escapeHtml(task.prompt)}</pre>
          </section>
          ${sourcePanel(task)}
        </div>
        ${operationsPanel(task)}
        ${filesPanel(task)}
      </div>`;
    bindDetailEvents(task, filteredIndex);
    renderFileView(task);
  }

  function sourceLines(text) {
    if (text == null) return [];
    const normalized = text.endsWith("\n") ? text.slice(0, -1) : text;
    return normalized ? normalized.split("\n") : [""];
  }

  function renderSource(text, missingLabel) {
    if (text == null) return `<div class="empty-code">${escapeHtml(missingLabel)}</div>`;
    return sourceLines(text)
      .map(
        (line, index) => `
          <div class="code-line">
            <span class="line-number">${index + 1}</span>
            <span class="line-number"></span>
            <span class="line-marker"></span>
            <span class="line-text">${escapeHtml(line || " ")}</span>
          </div>`,
      )
      .join("");
  }

  function renderDiff(diff) {
    if (!diff) return '<div class="empty-code">No changes in this file</div>';
    const lines = diff.endsWith("\n") ? diff.slice(0, -1).split("\n") : diff.split("\n");
    let oldLine = null;
    let newLine = null;
    return lines
      .map((line) => {
        let oldDisplay = "";
        let newDisplay = "";
        let marker = "";
        let className = "";
        if (line.startsWith("@@")) {
          const match = line.match(/^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/);
          if (match) {
            oldLine = Number(match[1]);
            newLine = Number(match[2]);
          }
          className = "diff-hunk";
        } else if (line.startsWith("---") || line.startsWith("+++")) {
          className = "diff-header";
        } else if (line.startsWith("+") && !line.startsWith("+++")) {
          newDisplay = newLine;
          newLine += 1;
          marker = "+";
          className = "diff-add";
        } else if (line.startsWith("-") && !line.startsWith("---")) {
          oldDisplay = oldLine;
          oldLine += 1;
          marker = "−";
          className = "diff-remove";
        } else if (line.startsWith(" ")) {
          oldDisplay = oldLine;
          newDisplay = newLine;
          oldLine += 1;
          newLine += 1;
        }
        const content = line.startsWith("+") || line.startsWith("-") || line.startsWith(" ") ? line.slice(1) : line;
        return `
          <div class="code-line ${className}">
            <span class="line-number">${oldDisplay ?? ""}</span>
            <span class="line-number">${newDisplay ?? ""}</span>
            <span class="line-marker">${marker}</span>
            <span class="line-text">${escapeHtml(content || " ")}</span>
          </div>`;
      })
      .join("");
  }

  function renderFileView(task) {
    const file = task.files.find((item) => item.path === state.selectedFile) || task.files[0];
    if (!file) return;
    const meta = document.getElementById("file-meta-bar");
    const viewport = document.getElementById("code-viewport");
    meta.innerHTML = `
      <span class="file-status">${escapeHtml(file.status)}</span>
      <span>${escapeHtml(file.before_lines)} → ${escapeHtml(file.expected_lines)} lines</span>
      <span>·</span>
      <span>${file.changed ? "expected change" : "context only"}</span>`;
    if (state.view === "before") {
      viewport.innerHTML = renderSource(file.before, "File does not exist in the before workspace");
      viewport.setAttribute("aria-label", `Before content for ${file.path}`);
    } else if (state.view === "expected") {
      viewport.innerHTML = renderSource(file.expected, "File does not exist in the expected workspace");
      viewport.setAttribute("aria-label", `Expected content for ${file.path}`);
    } else {
      viewport.innerHTML = renderDiff(file.diff);
      viewport.setAttribute("aria-label", `Expected diff for ${file.path}`);
    }
    viewport.scrollTo({ top: 0, left: 0 });
  }

  async function copyText(value, button, successLabel) {
    try {
      await navigator.clipboard.writeText(value);
    } catch (_error) {
      const area = document.createElement("textarea");
      area.value = value;
      area.style.position = "fixed";
      area.style.opacity = "0";
      document.body.append(area);
      area.select();
      document.execCommand("copy");
      area.remove();
    }
    const original = button.textContent;
    button.textContent = successLabel;
    window.setTimeout(() => {
      button.textContent = original;
    }, 1200);
  }

  function bindDetailEvents(task, filteredIndex) {
    document.getElementById("previous-task")?.addEventListener("click", () => {
      if (filteredIndex > 0) selectTask(state.filtered[filteredIndex - 1].id);
    });
    document.getElementById("next-task")?.addEventListener("click", () => {
      if (filteredIndex < state.filtered.length - 1) selectTask(state.filtered[filteredIndex + 1].id);
    });
    document.getElementById("copy-task-id")?.addEventListener("click", (event) => {
      copyText(task.id, event.currentTarget, "Copied");
    });
    document.getElementById("copy-prompt")?.addEventListener("click", (event) => {
      copyText(task.prompt, event.currentTarget, "Copied");
    });
    document.getElementById("file-picker")?.addEventListener("change", (event) => {
      state.selectedFile = event.currentTarget.value;
      renderFileView(task);
    });
    for (const tab of elements.taskDetail.querySelectorAll("[data-view]")) {
      tab.addEventListener("click", () => {
        state.view = tab.dataset.view;
        for (const peer of elements.taskDetail.querySelectorAll("[data-view]")) {
          peer.setAttribute("aria-selected", String(peer === tab));
        }
        renderFileView(task);
      });
    }
  }

  function navigateBy(delta) {
    if (!state.filtered.length) return;
    const index = state.filtered.findIndex((task) => task.id === state.selectedId);
    const nextIndex = Math.max(0, Math.min(state.filtered.length - 1, index + delta));
    if (nextIndex !== index) selectTask(state.filtered[nextIndex].id);
  }

  function initializeControls() {
    addOptions(elements.language, data.tasks.map((task) => task.language));
    addOptions(elements.family, data.tasks.map((task) => task.family));
    addOptions(elements.difficulty, data.tasks.map((task) => task.difficulty), difficultyOrder);
    addOptions(elements.length, data.tasks.map((task) => task.target_length), lengthOrder);
    for (const control of [elements.search, elements.language, elements.family, elements.difficulty, elements.length]) {
      control.addEventListener(control === elements.search ? "input" : "change", refreshFilters);
    }
    elements.clear.addEventListener("click", () => {
      elements.search.value = "";
      elements.language.value = "";
      elements.family.value = "";
      elements.difficulty.value = "";
      elements.length.value = "";
      refreshFilters();
    });
    document.addEventListener("keydown", (event) => {
      const target = event.target;
      const editing = target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement;
      if (event.key === "/" && !editing) {
        event.preventDefault();
        elements.search.focus();
      } else if (!editing && event.key.toLowerCase() === "j") {
        event.preventDefault();
        navigateBy(1);
      } else if (!editing && event.key.toLowerCase() === "k") {
        event.preventDefault();
        navigateBy(-1);
      } else if (event.key === "Escape" && document.activeElement === elements.search) {
        elements.search.blur();
      }
    });
    window.addEventListener("hashchange", () => {
      const id = decodeURIComponent(location.hash.slice(1));
      if (id && id !== state.selectedId) selectTask(id, false);
    });
  }

  renderHeader();
  initializeControls();
  filterTasks();
  const requestedId = decodeURIComponent(location.hash.slice(1));
  const initialTask = data.tasks.find((task) => task.id === requestedId) || data.tasks[0];
  state.selectedId = initialTask?.id || null;
  state.selectedFile = initialTask?.primary_target || initialTask?.files[0]?.path || null;
  renderTaskList();
  renderDetail();
})();
