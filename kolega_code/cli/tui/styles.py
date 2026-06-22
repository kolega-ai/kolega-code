"""CSS for the Textual CLI app."""

APP_CSS = """
    Screen {
        layout: vertical;
    }

    #body {
        height: 1fr;
    }

    #conversation_panel {
        width: 2fr;
        height: 100%;
    }

    #side_panel {
        width: 1fr;
        min-width: 34;
        height: 100%;
    }

    #conversation, #logs, #terminal {
        height: 1fr;
        border: round $surface;
    }

    ConversationEntryWidget {
        height: auto;
        padding-bottom: 1;
    }

    ToolEntryWidget {
        height: auto;
        padding-bottom: 1;
    }

    ToolEntryWidget Collapsible {
        background: transparent;
        border-top: none;
        padding-bottom: 0;
        padding-left: 0;
    }

    ToolEntryWidget Collapsible Contents {
        padding-left: 3;
    }

    ToolEntryWidget .tool-body {
        color: $text-muted;
    }

    #jump_to_bottom {
        display: none;
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
        text-align: center;
    }

    #status_container {
        height: 1fr;
    }

    #status_dashboard {
        height: 1fr;
        min-height: 15;
        border: round $surface;
        padding: 1;
    }

    #settings_form, #planning_form {
        height: 1fr;
        padding: 1;
    }

    .settings-section {
        height: auto;
        border: round $surface;
        border-title-color: $text;
        border-title-style: bold;
        padding: 0 1;
        margin-bottom: 1;
    }

    #settings_status {
        margin-top: 1;
    }

    #settings_form Label {
        margin-top: 1;
    }

    #settings_form Button {
        margin-top: 1;
    }

    .settings-hint {
        color: $text-muted;
        margin-top: 0;
    }

    .agent-model-group {
        height: auto;
        margin-top: 1;
    }

    .agent-model-role {
        text-style: bold;
        color: $text;
    }

    .agent-model-field {
        height: auto;
        margin-top: 1;
    }

    .agent-model-field-label {
        width: 10;
        margin-top: 1;
        color: $text-muted;
    }

    .agent-model-field Select {
        width: 1fr;
    }

    #settings_actions {
        height: auto;
        padding: 0 1;
        margin-top: 1;
    }

    #planning_form Markdown.empty-state {
        color: $text-muted;
    }

    #composer {
        dock: bottom;
        height: 5;
        border: round $surface;
    }

    #turn_status {
        display: none;
        height: 1;
        padding: 0 1;
        color: $text-muted;
        background: $surface;
    }

    #composer_hint_row {
        display: none;
        height: auto;
        max-height: 4;
        background: $surface;
    }

    #composer_hint_row #composer_hint {
        width: 1fr;
        height: auto;
        padding: 0 1;
    }

    #composer_hint_row #detach_btn {
        width: 3;
        height: 1;
        min-height: 1;
        border: none;
        background: transparent;
        color: $text-muted;
        padding: 0 1;
    }

    #composer_hint_row #detach_btn:hover {
        color: $warning;
    }

    #composer_hint_row #detach_btn:focus {
        color: $warning;
        text-style: bold;
    }

    #completion_dropdown {
        display: none;
        height: auto;
        max-height: 10;
        border: round $surface;
        background: $surface;
    }

    #composer_hint.hint-warning {
        color: $warning;
    }

    #composer_hint.hint-info {
        color: $text-muted;
    }

    #composer:disabled {
        opacity: 0.6;
    }

    #plan_actions, #model_actions, #effort_actions, #theme_actions {
        display: none;
        height: auto;
        max-height: 12;
        border: round $surface;
        background: $surface;
    }

    /* Question / approval prompts: the question is folded into the panel header
       (the border title) above its options, so the whole prompt reads as one
       bordered unit. The inner ActionList drops its own border. */
    #question_prompt, #approval_prompt {
        display: none;
        height: auto;
        max-height: 14;
        border: round $surface;
        background: $surface;
        border-title-color: $text;
        border-title-style: bold;
        padding: 0 1;
    }

    #question_prompt > ActionList, #approval_prompt > ActionList {
        border: none;
        background: $surface;
        height: auto;
        max-height: 10;
        padding: 0;
    }

    .prompt-header {
        padding: 0 0 1 0;
        background: $surface;
    }

    /* Neutralize the selected-row highlight on every choice list. Textual paints
       it with $block-cursor-background (= $primary, a saturated brand color),
       which clashes with the otherwise-neutral chrome and looks wrong in 256-color
       Terminal.app. Each theme pins $surface-lighten-2 to a near-neutral gray
       (see theme.build_textual_theme) so the highlight stays subtle across all
       themes — incl. Solarized, whose auto-derived $surface-lighten-2 would
       otherwise quantize to a saturated teal. The OptionList type selector also
       covers its subclasses: ActionList, CompletionDropdown, and the Select's
       SelectOverlay dropdown. */
    OptionList > .option-list--option-highlighted {
        background: $surface-lighten-2;
        color: $text;
    }

    .meta {
        color: $text-muted;
    }

    Footer {
        background: $surface;
    }

    Input {
        border: round $surface;
    }

    Input:focus {
        border: round $surface-lighten-2;
    }

    Select > SelectCurrent {
        border: round $surface;
    }

    Select:focus > SelectCurrent {
        border: round $surface-lighten-2;
    }

    Select > SelectOverlay {
        border: round $surface;
    }

    SubAgentInspectorScreen {
        align: center middle;
    }

    SubAgentInspectorScreen #inspector_body {
        width: 100%;
        height: 1fr;
    }

    SubAgentInspectorScreen #inspector_roster {
        width: 40;
        height: 100%;
        border: round $surface;
        padding: 0 1;
    }

    SubAgentInspectorScreen #inspector_main {
        width: 1fr;
        height: 100%;
    }

    SubAgentInspectorScreen #inspector_header {
        height: auto;
        border: round $surface;
        padding: 0 1;
    }

    SubAgentInspectorScreen #inspector_trajectory {
        height: 1fr;
        border: round $surface;
        padding: 0 1;
    }

    SubAgentRosterRow {
        height: auto;
        padding: 0 0 1 0;
    }

    SubAgentInspectorScreen .inspector-empty {
        color: $text-muted;
        padding: 1;
    }

    SubAgentInspectorScreen #inspector_footer {
        height: 1;
        padding: 0 1;
        background: $surface;
        color: $text-muted;
    }
    """
