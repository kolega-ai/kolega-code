from enum import Enum
from typing import Dict, Optional

from pydantic import BaseModel, Field, model_validator


class ModelProvider(str, Enum):
    """Supported model providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
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
            if config.provider != ModelProvider.LLAMA and self.get_api_key(config.provider) is None:
                raise ValueError(f"Missing API key for {config_name} provider '{config.provider.value}'")

        return self
