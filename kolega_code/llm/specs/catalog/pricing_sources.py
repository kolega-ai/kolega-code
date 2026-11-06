"""Public evidence/policy index, not an automatic price scraper.

Per-model records carry the authoritative exact-identity references. Registry
entries cover billing surfaces, including surfaces without universal cash rates.
"""

CHECKED_AT = "2026-11-01"

_SOURCES = {
    "anthropic": ("https://platform.claude.com/docs/en/about-claude/pricing", "first_party_api"),
    "openai": ("https://developers.openai.com/api/docs/pricing", "first_party_api"),
    "openai_chatgpt": ("https://developers.openai.com/codex/pricing", "subscription_or_provider_credits"),
    "google": ("https://ai.google.dev/gemini-api/docs/pricing", "gemini_developer_api"),
    "mistral": ("https://docs.mistral.ai/inference/pricing", "first_party_api"),
    "groq": ("https://console.groq.com/docs/models", "hosted_api"),
    "together": ("https://docs.together.ai/docs/serverless-models", "hosted_serverless_or_dedicated"),
    "fireworks": ("https://docs.fireworks.ai/serverless/pricing", "hosted_serverless_or_dedicated"),
    "xai": ("https://docs.x.ai/developers/pricing", "first_party_api"),
    "llama": ("https://github.com/ggml-org/llama.cpp", "self_hosted"),
    "dashscope": ("https://www.alibabacloud.com/help/en/model-studio/model-pricing", "regional_model_studio_api"),
    "moonshot": ("https://platform.kimi.ai/docs/pricing/chat", "first_party_api"),
    "minimax": ("https://platform.minimax.io/docs/guides/pricing-paygo", "first_party_api"),
    "deepseek": ("https://api-docs.deepseek.com/quick_start/pricing", "first_party_api"),
    "zai": ("https://docs.z.ai/devpack/overview", "coding_plan"),
    "kimi_coding": ("https://www.kimi.com/code/docs/kimi-code/membership.html", "subscription"),
    "ollama_cloud": ("https://ollama.com/pricing", "included_credits_or_pay_as_you_go"),
    "cloudflare_workers_ai": ("https://developers.cloudflare.com/workers-ai/platform/pricing/", "workers_ai"),
    "openrouter": ("https://openrouter.ai/api/v1/models", "gateway"),
    "thinking_machines": ("https://tinker-docs.thinkingmachines.ai/tinker/serverless.json", "tinker_serverless"),
    "tinker": ("https://tinker-docs.thinkingmachines.ai/tinker/models.json", "native_tinker_sampling"),
    "perplexity_agent": ("https://docs.perplexity.ai/docs/getting-started/pricing", "gateway"),
}

PROVIDER_SOURCES = {
    provider: {
        "checked_at": CHECKED_AT,
        "url": url,
        "billing_surface": surface,
        "policy": "Require exact model/route evidence; no sibling, upstream-vendor or replacement-rate inference.",
    }
    for provider, (url, surface) in _SOURCES.items()
}
PROVIDER_SOURCES["llama"]["policy"] = (
    "Self-hosted resource costs require operator configuration; absence of a list rate is not USD zero."
)
PROVIDER_SOURCES["openai_chatgpt"]["policy"] = (
    "OAuth subscription limits are not API USD prices. Published purchased-credit rates remain non-monetary."
)
PROVIDER_SOURCES["custom:*"] = {
    "checked_at": CHECKED_AT,
    "billing_surface": "operator_configured",
    "policy": "Require an exact custom-provider/model override; never assume compatible API billing identity.",
    "url": "https://www.openapis.org/",
}
PROVIDER_SOURCES["tinker://*"] = {
    "checked_at": CHECKED_AT,
    "billing_surface": "native_tinker_sampling",
    "policy": "Saved checkpoints are unpriced by default, even with a resolved base; require an exact-route override.",
    "url": "https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/index.md",
}
PROVIDER_SOURCES["refuseless"] = {
    "checked_at": "2026-11-05",
    "billing_surface": "prepaid_credits",
    "policy": (
        "Use current exact-route USD input, cached-input, and output rates from Refuseless's model/pricing pages."
    ),
    "url": "https://refuseless.com/pricing",
}

# Names are aliases for consumers auditing coverage; there is one registry.
PRICING_SOURCES = PROVIDER_SOURCES
SOURCE_REGISTRY = PROVIDER_SOURCES
