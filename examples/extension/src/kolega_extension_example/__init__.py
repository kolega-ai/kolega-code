"""Minimal Kolega Code CLI extension: one prompt section, one harmless tool.

Load with:

    kolega-code --extension kolega_extension_example:create_extension
    kolega-code ask "..." --extension kolega_extension_example:create_extension
"""

from pathlib import Path

from kolega_code import (
    BaseAgent,
    KolegaExtensionBundle,
    KolegaExtensionHost,
    PromptExtension,
    ToolExtension,
)


def create_extension(host: KolegaExtensionHost, config_path: Path | None) -> KolegaExtensionBundle:
    """Called once per top-level agent generation with resolved host metadata."""
    state: dict = {"agent": None, "config_path": config_path}

    async def extension_echo(text: str) -> str:
        """Echo the given text back, with the caller's history length.

        Args:
            text: Text to echo back verbatim.
        """
        # Tool callbacks are awaited by the executor, so they must be async.
        # The model-visible schema is derived from the signature (types from
        # annotations, required from missing defaults) and this docstring's
        # Args: section; tool_schemas= is only needed for shapes a signature
        # cannot express. The bound agent is available through the closure.
        agent: BaseAgent | None = state["agent"]
        history_length = len(agent.history) if agent is not None else 0
        return f"echo: {text} (caller history: {history_length} messages)"

    def bind_agent(agent: BaseAgent) -> None:
        state["agent"] = agent

    def cleanup() -> None:
        state["agent"] = None

    return KolegaExtensionBundle(
        prompt_extensions=[
            PromptExtension(
                id="kolega-extension-example",
                title="Example extension",
                markdown=(
                    "The `extension_echo` tool is provided by a CLI extension for "
                    "demonstration purposes. It echoes its input."
                ),
            )
        ],
        tool_extensions=[
            ToolExtension(
                name="kolega-extension-example",
                tools={"extension_echo": extension_echo},
                # The callback reads the bound top-level agent, which sub-agents
                # are not; do not propagate it to them.
                propagate_to_sub_agents=False,
            )
        ],
        bind_agent=bind_agent,
        cleanup=cleanup,
    )
