from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from kolega_code.auth.tokens import OAuthTokens


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
                model="claude-opus-4-8"
            )
        )

        # Get an API key for a specific provider
        api_key = config.get_api_key(ModelProvider.ANTHROPIC)

        # Access model configurations
        long_context_model = config.long_context_config
        fast_model = config.fast_config
        thinking_model = config.thinking_config

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

    # ChatGPT-subscription OAuth credentials (used instead of an api key for the
    # OPENAI_CHATGPT provider). The live, refreshing token manager is attached
    # separately via attach_chatgpt_token_manager so refreshes persist to disk.
    openai_chatgpt_tokens: Optional[OAuthTokens] = Field(
        default=None, description="ChatGPT OAuth tokens for the openai_chatgpt provider"
    )
    _chatgpt_token_manager: Optional[Any] = PrivateAttr(default=None)

    # Web search configuration (the web_search tool). Optional: the default backend is
    # keyless, so these must never be required for AgentConfig to be constructable.
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

    # Model configurations
    long_context_config: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="claude-opus-4-8", thinking_effort="medium"
        ),
        description="Configuration for long context operations",
    )

    fast_config: ModelConfig = Field(
        default_factory=lambda: ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-haiku-4-5-20251001"),
        description="Configuration for fast operations",
    )

    thinking_config: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            provider=ModelProvider.ANTHROPIC, model="claude-opus-4-8", thinking_effort="medium"
        ),
        description="Configuration for thinking operations",
    )

    # Per-agent-role model overrides, keyed by AgentRole value (e.g. "investigation").
    # A role with no entry inherits long_context_config, so an empty mapping
    # reproduces the previous single-model behavior.
    agent_models: Dict[str, ModelConfig] = Field(
        default_factory=dict,
        description="Per-agent-role model overrides keyed by AgentRole value",
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

    def get_api_key(self, provider: ModelProvider) -> Optional[str]:
        """Get the API key for a specific provider."""
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
            ModelProvider.LLAMA: None,  # Local model, no API key needed
        }
        return api_key_map[provider]

    @model_validator(mode="after")
    def validate_provider_api_key(self) -> "AgentConfig":
        """Validates that if a model provider is specified, the corresponding API key is provided."""
        configs = [
            (self.long_context_config, "long context"),
            (self.fast_config, "fast"),
            (self.thinking_config, "thinking"),
        ]
        configs.extend((override, f"agent '{role}'") for role, override in self.agent_models.items())

        for config, config_name in configs:
            provider = config.provider
            if provider == ModelProvider.LLAMA:
                continue
            if provider == ModelProvider.OPENAI_CHATGPT:
                # OAuth provider: satisfied by stored ChatGPT tokens, not an api key.
                if self.openai_chatgpt_tokens is None:
                    raise ValueError(f"Not signed in to ChatGPT for {config_name}; run /login chatgpt to sign in.")
                continue
            if self.get_api_key(provider) is None:
                raise ValueError(f"Missing API key for {config_name} provider '{provider.value}'")

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
