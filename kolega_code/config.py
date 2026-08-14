from enum import Enum
from typing import Any, Dict, List, Literal, Optional, cast

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from kolega_code.auth.tokens import OAuthTokens
from kolega_code.llm.specs.custom_endpoints import (
    CUSTOM_PROVIDER_PREFIX,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_OUTPUT_TOKENS,
    REASONING_REPLAY_VALUES,
    custom_endpoint_id,
    is_custom_provider,
    sync_custom_endpoint_specs,
    valid_custom_endpoint_id,
)
from kolega_code.services.lsp.config import LspConfig


class ModelProvider(str, Enum):
    """Supported model providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENAI_CHATGPT = "openai_chatgpt"  # OpenAI via ChatGPT-subscription OAuth (Responses API)
    GOOGLE = "google"
    GROQ = "groq"
    TOGETHER = "together"
    FIREWORKS = "fireworks"
    XAI = "xai"
    LLAMA = "llama"
    DASHSCOPE = "dashscope"
    MOONSHOT = "moonshot"
    DEEPSEEK = "deepseek"
    ZAI = "zai"
    KIMI_CODING = "kimi_coding"
    OLLAMA_CLOUD = "ollama_cloud"
    OPENROUTER = "openrouter"  # Gateway in front of many vendors' models
    THINKING_MACHINES = "thinking_machines"  # Thinking Machines Lab (Tinker API)
    TINKER = "tinker"  # Native Tinker SamplingClient (agentic-RL trajectory source)
    PERPLEXITY_AGENT = "perplexity_agent"

    @classmethod
    def _missing_(cls, value: object) -> Optional["ModelProvider"]:
        # Dynamic members for user-defined endpoints; __members__ iteration stays built-ins only.
        if isinstance(value, str) and value.startswith(CUSTOM_PROVIDER_PREFIX):
            endpoint = value[len(CUSTOM_PROVIDER_PREFIX) :]
            if valid_custom_endpoint_id(endpoint):
                member = cast("ModelProvider", str.__new__(cls, value))
                member._name_ = value  # type: ignore[attr-defined]
                member._value_ = value  # type: ignore[attr-defined]
                cls._value2member_map_[value] = member  # type: ignore[attr-defined]
                return member
        return None


class EditProtocol(str, Enum):
    """Model-facing language used for ordinary file edits."""

    SEARCH_REPLACE = "search_replace"
    CODEX_APPLY_PATCH = "codex_apply_patch"
    CLAUDE_CODE = "claude_code"
    HASHLINE_V2 = "hashline_v2"


class AgentRole(str, Enum):
    """Configurable agent roles that can each run on their own model.

    Keyed off each agent class's stable ``agent_name``. A role with no entry in
    ``AgentConfig.agent_models`` inherits the global ``long_context_config``.
    """

    PLANNING = "planning"
    BUILDING = "building"  # the coder agent
    INVESTIGATION = "investigation"
    GENERAL = "general"
    BROWSER = "browser"


# Maps a BaseAgent.agent_name to its configurable role. Agents whose name is not
# listed (e.g. the abstract base) simply fall back to the global model.
AGENT_ROLE_BY_NAME: Dict[str, AgentRole] = {
    "planning-agent": AgentRole.PLANNING,
    "coder": AgentRole.BUILDING,
    "investigation-agent": AgentRole.INVESTIGATION,
    "general-agent": AgentRole.GENERAL,
    "browser-agent": AgentRole.BROWSER,
}


class RateLimitConfig(BaseModel):
    """Rate limit configuration for a specific LLM."""

    requests_per_minute: int = Field(default=60, description="Maximum number of requests allowed per minute", gt=0)

    tokens_per_minute: int = Field(default=80_000, description="Maximum number of tokens allowed per minute", gt=0)

    max_retries: int = Field(
        default=4,
        description="Retries the underlying SDK client performs per request (exponential backoff + jitter, honors retry-after)",
        ge=0,
    )

    loop_max_retries: int = Field(
        default=3,
        description="Consecutive agent-loop retries on rate-limit/overload after the SDK's own retries are exhausted",
        ge=0,
    )


class ModelConfig(BaseModel):
    """Configuration for a specific model type (long context, fast, or thinking)."""

    provider: ModelProvider = Field(description="Provider to use for this model configuration")

    model: str = Field(description="Model identifier to use")

    rate_limits: RateLimitConfig = Field(default_factory=RateLimitConfig, description="Rate limits for this model")

    thinking_effort: Optional[str] = Field(
        default=None,
        description="Model-specific thinking or reasoning effort level",
    )


class CustomEndpointConfig(BaseModel):
    """A user-defined OpenAI/Responses/Anthropic-compatible server endpoint.

    Referenced by provider value ``custom:<id>`` where ``id`` is the key in
    ``AgentConfig.custom_endpoints``. Any model id is accepted; the endpoint's
    defaults (and optional per-model ``models`` overrides) supply context specs.
    """

    api_style: Literal["openai_chat", "openai_responses", "anthropic"] = Field(
        description="Wire dialect: Chat Completions, Responses API, or Anthropic Messages"
    )
    base_url: str = Field(description="Base URL; OpenAI styles include /v1, Anthropic style is the API root")
    api_key: Optional[str] = Field(default=None, description="Optional Bearer credential")
    label: Optional[str] = Field(default=None, description="Display name in pickers")
    default_model: Optional[str] = Field(default=None, description="Model used by connection probes")
    context_length: int = Field(default=DEFAULT_CONTEXT_LENGTH, gt=0)
    max_output_tokens: int = Field(default=DEFAULT_MAX_OUTPUT_TOKENS, gt=0)
    supports_vision: bool = Field(default=False)
    temperature: Optional[float] = Field(
        default=None,
        gt=0,
        le=2,
        description="Sampling temperature (keep <= 1 for anthropic style; ignored by openai_responses)",
    )
    models: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Per-model spec overrides")
    thinking: Optional[Dict[str, Any]] = Field(default=None, description="Thinking-effort spec for this endpoint")
    reasoning_replay: str = Field(
        default="auto", description="Reasoning replay field: auto|reasoning_content|reasoning|off"
    )

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http(s) URL")
        return value

    @field_validator("reasoning_replay")
    @classmethod
    def _validate_reasoning_replay(cls, value: str) -> str:
        if value not in REASONING_REPLAY_VALUES:
            raise ValueError(f"reasoning_replay must be one of {', '.join(REASONING_REPLAY_VALUES)}")
        return value


class AgentConfig(BaseModel):
    """Configuration for the agent system.

    This class contains all configuration parameters needed to run the agent,
    including API keys for different providers and model configurations for
    various operational modes (long context, fast, and thinking).

    Usage:
        # Create a default configuration
        config = AgentConfig()

        # Create a custom configuration
        config = AgentConfig(
            anthropic_api_key="your_anthropic_key",
            long_context_config=ModelConfig(
                provider=ModelProvider.ANTHROPIC,
                model="claude-opus-5"
            )
        )

        # Get an API key for a specific provider
        api_key = config.get_api_key(ModelProvider.ANTHROPIC)

        # Access model configurations
        long_context_model = config.long_context_config
        fast_model = config.fast_config

    API keys can be set directly or loaded from environment variables.
    Model configurations define which models to use for different operational
    contexts and their respective token limits and rate limits.
    """

    # API Keys for different providers
    anthropic_api_key: Optional[str] = Field(default=None, description="API key for Anthropic")
    openai_api_key: Optional[str] = Field(default=None, description="API key for OpenAI")
    google_api_key: Optional[str] = Field(default=None, description="API key for Google")
    groq_api_key: Optional[str] = Field(default=None, description="API key for Groq")
    together_api_key: Optional[str] = Field(default=None, description="API key for Together.ai")
    fireworks_api_key: Optional[str] = Field(default=None, description="API key for Fireworks.ai")
    xai_api_key: Optional[str] = Field(default=None, description="API key for X.ai")
    dashscope_api_key: Optional[str] = Field(default=None, description="API key for Dashscope (Alibaba Model Studio)")
    moonshot_api_key: Optional[str] = Field(default=None, description="API key for Moonshot.ai")
    deepseek_api_key: Optional[str] = Field(default=None, description="API key for DeepSeek")
    zai_api_key: Optional[str] = Field(default=None, description="API key for Z.AI (GLM Coding Plan)")
    kimi_coding_api_key: Optional[str] = Field(default=None, description="API key for Kimi Coding Plan")
    ollama_cloud_api_key: Optional[str] = Field(default=None, description="API key for Ollama Cloud")
    openrouter_api_key: Optional[str] = Field(default=None, description="API key for OpenRouter")
    thinking_machines_api_key: Optional[str] = Field(default=None, description="API key for Thinking Machines (Tinker)")
    perplexity_api_key: Optional[str] = Field(
        default=None, description="API key for Perplexity (Gateway and Agent API)"
    )

    # ChatGPT-subscription OAuth credentials (used instead of an api key for the
    # OPENAI_CHATGPT provider). The live, refreshing token manager is attached
    # separately via attach_chatgpt_token_manager so refreshes persist to disk.
    openai_chatgpt_tokens: Optional[OAuthTokens] = Field(
        default=None, description="ChatGPT OAuth tokens for the openai_chatgpt provider"
    )
    _chatgpt_token_manager: Optional[Any] = PrivateAttr(default=None)

    # Web search configuration (the web_search tool). Optional: the default backend is
    # keyless, so these must never be required for AgentConfig to be constructable.
    web_search_mode: str = Field(
        default="auto",
        description=(
            "Web tool mode: auto (hosted server-side search when the model supports it, "
            "else the client tools), hosted, client, or off"
        ),
    )
    web_search_backend: str = Field(
        default="duckduckgo", description="Selected web_search backend (duckduckgo, firecrawl, tavily, searxng)"
    )
    web_search_api_key: Optional[str] = Field(
        default=None, description="API key for the selected cloud web-search backend (Firecrawl/Tavily)"
    )
    web_search_base_url: Optional[str] = Field(
        default=None, description="Base URL for the self-hosted SearXNG web-search backend"
    )

    # Langfuse configuration
    langfuse_enabled: bool = Field(default=False, description="Enable Langfuse tracing")
    langfuse_host: Optional[str] = Field(default=None, description="Langfuse host URL")
    langfuse_public_key: Optional[str] = Field(default=None, description="Langfuse public key")
    langfuse_secret_key: Optional[str] = Field(default=None, description="Langfuse secret key")
    environment: Optional[str] = Field(default="development", description="Environment name (development, production)")
    edit_protocol: Optional[EditProtocol] = Field(
        default=None,
        description="Optional session-wide edit protocol override; model catalog preference is used when unset",
    )

    # Model configurations
    long_context_config: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="claude-opus-5", thinking_effort="medium"
        ),
        description="Configuration for long context operations",
    )

    fast_config: ModelConfig = Field(
        default_factory=lambda: ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5-20251001"),
        description="Configuration for fast operations",
    )

    # Per-agent-role model overrides, keyed by AgentRole value (e.g. "investigation").
    # A role with no entry inherits long_context_config, so an empty mapping
    # reproduces the previous single-model behavior.
    agent_models: Dict[str, ModelConfig] = Field(
        default_factory=dict,
        description="Per-agent-role model overrides keyed by AgentRole value",
    )

    # Host-loaded MCP configuration. Excluded from serialized model config because
    # it can contain local paths, headers/env secrets, and non-Pydantic dataclasses.
    mcp_config: Optional[Any] = Field(default=None, exclude=True, description="Loaded MCP server configuration")

    # LSP (Language Server Protocol) configuration. Excluded from serialization
    # (parity with mcp_config): LspConfig.custom_servers.env and
    # workspace_configuration can hold secrets/local paths.
    lsp: LspConfig = Field(default_factory=LspConfig, exclude=True, description="LSP integration configuration")

    # Whether the project's .kolega/lsp.json is trusted to define custom language
    # servers. Excluded from serialization (parity with mcp_config) — it is a
    # runtime-resolved trust flag, not user-editable config.
    lsp_project_trusted: bool = Field(default=False, exclude=True, description="Project LSP config trust flag")

    # Fraction of the model context window above which automatic history
    # compression kicks in (e.g. 0.8 = compress once input tokens exceed 80%).
    # None = the agent's built-in default (BaseAgent.history_compression_threshold).
    history_compression_threshold: Optional[float] = Field(
        default=None,
        gt=0,
        le=1.0,
        description="Context-window fraction that triggers automatic history compression",
    )

    # Strict per-run context budget from the paired CLI flags. Process-local:
    # populated only from the command line, never persisted. When set, the
    # agent budgets against these instead of the catalog and enforces a hard
    # post-compaction preflight.
    context_window_tokens: Optional[int] = Field(
        default=None, gt=0, description="Total input-plus-output context-window limit for this process"
    )
    max_output_tokens: Optional[int] = Field(
        default=None, gt=0, description="Output allowance reserved for a primary model call in this process"
    )

    @model_validator(mode="after")
    def _validate_strict_context_budget(self) -> "AgentConfig":
        window, output = self.context_window_tokens, self.max_output_tokens
        if (window is None) != (output is None):
            raise ValueError("context_window_tokens and max_output_tokens must be supplied together")
        if window is not None and output is not None and output >= window:
            raise ValueError("max_output_tokens must be strictly smaller than context_window_tokens")
        return self

    # eval tool (persistent code kernels with a loopback tool bridge). All fields
    # are additive with defaults that enable the feature; excluded from
    # serialization since they carry local paths (parity with lsp/mcp_config).
    eval_enabled: bool = Field(default=True, exclude=True, description="Enable the eval tool and its kernels")
    eval_python_version: str = Field(
        default="3.12", exclude=True, description="Python version requested for the managed eval environment"
    )
    eval_env_path: Optional[str] = Field(
        default=None, exclude=True, description="Override location of the managed eval environment directory"
    )
    eval_kernel_packages: Optional[List[str]] = Field(
        default=None, exclude=True, description="Extra packages installed into the eval environment at provision time"
    )
    eval_python_path: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Full interpreter override for the eval Python kernel (skips managed-env provisioning)",
    )
    eval_js_runtime: Optional[str] = Field(
        default=None, exclude=True, description="JS runtime for the eval JS kernel: 'bun', 'node', or a path"
    )

    # Sub-agent dispatch (the dispatch_agent tool). Additive with a default that
    # enables the feature; excluded from serialization (parity with
    # eval_enabled) since it is resolved from settings.json / CLI flag at build.
    subagents_enabled: bool = Field(default=True, exclude=True, description="Enable the dispatch_agent sub-agent tool")

    # CLI Agent Skills. This runtime-only switch controls whether CLI hosts add
    # the skill catalog prompt and model-facing activation tool.
    skills_enabled: bool = Field(default=True, exclude=True, description="Enable CLI Agent Skills")

    # Custom model endpoints keyed by endpoint id (provider value "custom:<id>").
    # Excluded from serialization like mcp_config: entries can carry API keys.
    custom_endpoints: Dict[str, CustomEndpointConfig] = Field(
        default_factory=dict, exclude=True, description="User-defined OpenAI/Responses/Anthropic-compatible endpoints"
    )

    def model_config_for_agent(self, agent_name: Optional[str]) -> ModelConfig:
        """Return the model configuration an agent should use for its main loop.

        Resolves the agent's role from its ``agent_name`` and returns the matching
        override, falling back to ``long_context_config`` when the role has no
        override configured.
        """
        role = AGENT_ROLE_BY_NAME.get(agent_name or "")
        if role is not None:
            override = self.agent_models.get(role.value)
            if override is not None:
                return override
        return self.long_context_config

    def custom_endpoint_for(self, model_config: ModelConfig) -> Optional[CustomEndpointConfig]:
        """Return the endpoint definition backing a model config, if it is a custom endpoint."""
        endpoint_id = custom_endpoint_id(model_config.provider)
        if endpoint_id is None:
            return None
        return self.custom_endpoints.get(endpoint_id)

    def resolve_edit_protocol(self, model_config: Optional[ModelConfig] = None) -> EditProtocol:
        """Resolve the model-facing edit protocol for one effective model."""

        return self.resolve_edit_protocol_with_source(model_config)[0]

    def resolve_edit_protocol_with_source(self, model_config: Optional[ModelConfig] = None) -> tuple[EditProtocol, str]:
        """Resolve an edit protocol and report which configuration layer chose it."""

        if self.edit_protocol is not None:
            return self.edit_protocol, "session_override"

        effective_model = model_config or self.long_context_config
        from kolega_code.llm.specs import preferred_edit_protocol

        preferred = preferred_edit_protocol(effective_model.provider, effective_model.model)
        if preferred is not None:
            return EditProtocol(preferred), "model_catalog"
        return EditProtocol.CLAUDE_CODE, "default"

    def get_api_key(self, provider: ModelProvider) -> Optional[str]:
        """Get the API key for a specific provider."""
        if is_custom_provider(provider):
            endpoint_id = custom_endpoint_id(provider)
            endpoint = self.custom_endpoints.get(endpoint_id) if endpoint_id else None
            return endpoint.api_key if endpoint and endpoint.api_key else None
        api_key_map = {
            ModelProvider.ANTHROPIC: self.anthropic_api_key,
            ModelProvider.OPENAI: self.openai_api_key,
            # The OAuth access token doubles as the "api key" for compatibility with
            # call sites; the live provider uses the refreshing token manager instead.
            ModelProvider.OPENAI_CHATGPT: (
                self.openai_chatgpt_tokens.access_token if self.openai_chatgpt_tokens else None
            ),
            ModelProvider.GOOGLE: self.google_api_key,
            ModelProvider.GROQ: self.groq_api_key,
            ModelProvider.TOGETHER: self.together_api_key,
            ModelProvider.FIREWORKS: self.fireworks_api_key,
            ModelProvider.XAI: self.xai_api_key,
            ModelProvider.DASHSCOPE: self.dashscope_api_key,
            ModelProvider.MOONSHOT: self.moonshot_api_key,
            ModelProvider.DEEPSEEK: self.deepseek_api_key,
            ModelProvider.ZAI: self.zai_api_key,
            ModelProvider.KIMI_CODING: self.kimi_coding_api_key,
            ModelProvider.OLLAMA_CLOUD: self.ollama_cloud_api_key,
            ModelProvider.OPENROUTER: self.openrouter_api_key,
            ModelProvider.THINKING_MACHINES: self.thinking_machines_api_key,
            # The native Tinker provider shares the same TINKER_API_KEY credential.
            ModelProvider.TINKER: self.thinking_machines_api_key,
            ModelProvider.PERPLEXITY_AGENT: self.perplexity_api_key,
            ModelProvider.LLAMA: None,  # Local model, no API key needed
        }
        return api_key_map[provider]

    @model_validator(mode="after")
    def validate_provider_api_key(self) -> "AgentConfig":
        """Validates that if a model provider is specified, the corresponding API key is provided."""
        configs = [
            (self.long_context_config, "long context"),
            (self.fast_config, "fast"),
        ]
        configs.extend((override, f"agent '{role}'") for role, override in self.agent_models.items())

        for config, config_name in configs:
            provider = config.provider
            if provider == ModelProvider.LLAMA:
                continue
            if is_custom_provider(provider):
                # Endpoint keys are optional (local servers are usually keyless).
                continue
            if provider == ModelProvider.OPENAI_CHATGPT:
                # OAuth provider: satisfied by stored ChatGPT tokens, not an api key.
                if self.openai_chatgpt_tokens is None:
                    raise ValueError(f"Not signed in to ChatGPT for {config_name}; run /login chatgpt to sign in.")
                continue
            if self.get_api_key(provider) is None:
                raise ValueError(f"Missing API key for {config_name} provider '{provider.value}'")

        return self

    @model_validator(mode="after")
    def _sync_custom_endpoint_specs(self) -> "AgentConfig":
        """Register runtime model specs so library hosts don't need a separate sync call."""
        sync_custom_endpoint_specs(
            {
                endpoint_id: endpoint.model_dump(mode="python", exclude_none=True)
                for endpoint_id, endpoint in self.custom_endpoints.items()
            }
        )
        return self

    def attach_chatgpt_token_manager(self, manager: Any) -> None:
        """Attach a live, persisting ChatGPT token manager (wired by the CLI)."""
        self._chatgpt_token_manager = manager

    def get_chatgpt_token_manager(self) -> Optional[Any]:
        """Return the ChatGPT token manager, building an in-memory one if needed.

        The CLI attaches a manager whose refreshes persist to settings.json. When
        none is attached (e.g. programmatic use), fall back to a manager built from
        the stored tokens that refreshes in-memory only.
        """
        if self._chatgpt_token_manager is None and self.openai_chatgpt_tokens is not None:
            from kolega_code.auth.tokens import ChatGPTTokenManager

            self._chatgpt_token_manager = ChatGPTTokenManager(self.openai_chatgpt_tokens)
        return self._chatgpt_token_manager
