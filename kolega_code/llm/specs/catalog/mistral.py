from kolega_code.llm.specs.types import ThinkingEffortSpec


MISTRAL_REASONING = ThinkingEffortSpec(
    options=("none", "high"),
    default="none",
    mode="mistral_effort",
)


def _chat_spec(
    *,
    context_length: int,
    default_temperature: float,
    supports_vision: bool,
    supports_function_calling: bool = True,
    thinking_effort: ThinkingEffortSpec | None = None,
) -> dict:
    spec = {
        "context_length": context_length,
        # Mistral model metadata publishes context length, but not a hard native
        # output ceiling. This is Kolega's reserved output budget/policy so
        # window_minus_output context accounting leaves room for completions.
        "max_completion_tokens": 32768,
        "input_budget": "window_minus_output",
        "default_temperature": default_temperature,
        "supports_vision": supports_vision,
        "supports_function_calling": supports_function_calling,
        "supports_hosted_web_search": False,
    }
    if thinking_effort is not None:
        spec["thinking_effort"] = thinking_effort
    return spec


# First-party Mistral catalog. Insertion order is load-bearing: the first entry
# is the Settings picker/provider-switch default and must match
# PROVIDER_DEFAULT_MODEL[MISTRAL].
MISTRAL_SPECS = {
    ("mistral", "mistral-medium-3-5"): _chat_spec(
        context_length=262144,
        default_temperature=1.0,
        supports_vision=True,
        thinking_effort=MISTRAL_REASONING,
    ),
    ("mistral", "mistral-medium-latest"): _chat_spec(
        context_length=262144,
        default_temperature=1.0,
        supports_vision=True,
        thinking_effort=MISTRAL_REASONING,
    ),
    ("mistral", "mistral-medium"): _chat_spec(
        context_length=262144,
        default_temperature=1.0,
        supports_vision=True,
        thinking_effort=MISTRAL_REASONING,
    ),
    ("mistral", "mistral-medium-3.5"): _chat_spec(
        context_length=262144,
        default_temperature=1.0,
        supports_vision=True,
        thinking_effort=MISTRAL_REASONING,
    ),
    ("mistral", "mistral-medium-3"): _chat_spec(
        context_length=262144,
        default_temperature=1.0,
        supports_vision=True,
        thinking_effort=MISTRAL_REASONING,
    ),
    ("mistral", "mistral-medium-2604"): _chat_spec(
        context_length=262144,
        default_temperature=1.0,
        supports_vision=True,
        thinking_effort=MISTRAL_REASONING,
    ),
    ("mistral", "mistral-small-2603"): _chat_spec(
        context_length=262144,
        default_temperature=0.3,
        supports_vision=True,
        thinking_effort=MISTRAL_REASONING,
    ),
    ("mistral", "mistral-small-latest"): _chat_spec(
        context_length=262144,
        default_temperature=0.3,
        supports_vision=True,
        thinking_effort=MISTRAL_REASONING,
    ),
    ("mistral", "mistral-large-4"): _chat_spec(
        # Authenticated GET /v1/models reports 1,048,576 for Large 4, and the
        # same field agrees with the published windows for every other family.
        context_length=1048576,
        default_temperature=1.0,
        supports_vision=True,
        thinking_effort=MISTRAL_REASONING,
    ),
    ("mistral", "mistral-large-4-0"): _chat_spec(
        context_length=1048576,
        default_temperature=1.0,
        supports_vision=True,
        thinking_effort=MISTRAL_REASONING,
    ),
    ("mistral", "ministral-3b-2512"): _chat_spec(
        context_length=131072,
        default_temperature=0.3,
        supports_vision=True,
    ),
    ("mistral", "ministral-3b-latest"): _chat_spec(
        context_length=131072,
        default_temperature=0.3,
        supports_vision=True,
    ),
    ("mistral", "ministral-8b-2512"): _chat_spec(
        context_length=262144,
        default_temperature=0.3,
        supports_vision=True,
    ),
    ("mistral", "ministral-8b-latest"): _chat_spec(
        context_length=262144,
        default_temperature=0.3,
        supports_vision=True,
    ),
    ("mistral", "ministral-14b-2512"): _chat_spec(
        context_length=262144,
        default_temperature=0.3,
        supports_vision=True,
    ),
    ("mistral", "ministral-14b-latest"): _chat_spec(
        context_length=262144,
        default_temperature=0.3,
        supports_vision=True,
    ),
    ("mistral", "codestral-2508"): _chat_spec(
        context_length=256000,
        default_temperature=0.3,
        supports_vision=False,
    ),
    ("mistral", "codestral-latest"): _chat_spec(
        context_length=256000,
        default_temperature=0.3,
        supports_vision=False,
    ),
    # Appended after the reviewed set so the picker default and every existing
    # ordering stay put. Large 3 (v25.12) is still a current generalist model in
    # Mistral's catalog; the live model list reports no reasoning capability.
    ("mistral", "mistral-large-2512"): _chat_spec(
        context_length=262144,
        default_temperature=0.3,
        supports_vision=True,
    ),
    ("mistral", "mistral-large-latest"): _chat_spec(
        context_length=262144,
        default_temperature=0.3,
        supports_vision=True,
    ),
    # ``mistral-code-latest`` and ``mistral-code-fim-latest`` are aliases of the
    # Codestral billing model. Only the chat alias is listed; the FIM endpoint is
    # a fill-in-the-middle surface, not an agent chat model.
    ("mistral", "mistral-code-latest"): _chat_spec(
        context_length=256000,
        default_temperature=0.3,
        supports_vision=False,
    ),
}
