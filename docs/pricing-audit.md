# Catalogue pricing audit

Checked November 1, 2026. This is a pricing-only audit; no bundled capability rows,
model ordering, defaults, context windows, or availability were changed.

## Coverage

All **534 pricing cards live directly in the existing provider catalogue
entries** under `kolega_code/llm/specs/catalog/`: 533 bundled models and the
Tinker checkpoint wildcard. There is no parallel rate registry or import-time
pricing injection. `pricing_feeds.py` contains shared parsing and validation;
reviewed fallbacks deep-copy the actual provider card. Provider policies also
cover `llama` self-hosting and dynamic `custom:<id>` endpoints without inventing
selectable models or flat rates.

| Provider | Records | Status |
| --- | ---: | --- |
| `anthropic` | 13 | 13 priced |
| `openai` | 11 | 11 priced API routes |
| `openai_chatgpt` | 12 | 8 priced in credits; 4 subscription gaps |
| `google` | 7 | 7 priced |
| `mistral` | 21 | 15 priced; 6 unpublished exact-route rates |
| `groq` | 6 | 2 priced; 4 retired |
| `together` | 2 | 2 retired |
| `fireworks` | 9 | 3 priced; 4 retired serverless routes; 2 configuration gaps |
| `xai` | 5 | 5 priced, with unobserved usage-violation fee quantities |
| `dashscope` | 2 | 1 priced; 1 unpublished international-route rate |
| `moonshot` | 6 | 6 priced |
| `minimax` | 9 | 2 priced; 7 unverified historical routes |
| `deepseek` | 2 | 2 priced, holiday-aware schedules and dated tool fees |
| `zai` | 4 | 4 priced in Coding Plan credits, not USD |
| `kimi_coding` | 6 | 6 subscription/Extra Usage monetary gaps |
| `ollama_cloud` | 22 | 22 priced at published USD resource/list value |
| `cloudflare_workers_ai` | 13 | 13 priced at published resource/list value |
| `openrouter` | 296 | 291 priced, including 13 verified zero-price variants; 5 unverified |
| `thinking_machines` | 2 | 2 priced serverless alias estimates, with mapping assumptions |
| `tinker` | 35 | 31 priced native bases; 3 retired bases; 1 base-resolution policy |
| `perplexity_agent` | 51 | 25 priced exact public rows; 26 unverified historical routes |
| `llama` | 0 | Provider-level self-hosted policy; operator pricing required |

Totals: **463 priced, 38 unverified, 13 retired, 10 subscription,
7 unpublished and 3 requires-configuration**. The 71 non-priced cards are
model-specific sourced exceptions. Priced includes non-monetary credit cards
and cards that yield only a known subtotal without all billing quantities.
There are no blanket zero prices for subscriptions or unknown routes.

## Sources and billing decisions

Every executable card includes its source URL and exact source model identifier.
Machine-readable source excerpts also carry a SHA-256 digest. Primary sources:

| Billing surface | Official source |
| --- | --- |
| Fireworks serverless | `https://docs.fireworks.ai/serverless/pricing` |
| Fireworks retirements | `https://docs.fireworks.ai/updates/changelog` |
| Fireworks router selection | `https://docs.fireworks.ai/serverless/serverless-modes` and `https://docs.fireworks.ai/serverless/us-only-serverless` |
| OpenRouter conditional override contract | `https://openrouter.ai/docs/api-reference/models/get-models` |
| MiniMax pay-as-you-go | `https://platform.minimax.io/docs/guides/pricing-paygo` |
| DeepSeek API | `https://api-docs.deepseek.com/quick_start/pricing` |
| Chinese public holidays | `https://www.beijing.gov.cn/cs/gncs/zcwj/202603/t20260327_4568275.html` |
| Z.AI Coding Plan | `https://docs.z.ai/devpack/overview` |
| Z.AI API, separate non-applied comparison | `https://docs.z.ai/guides/overview/pricing` |
| Kimi Code membership | `https://www.kimi.com/code/docs/kimi-code/membership.html` |
| Ollama Cloud | `https://ollama.com/pricing` and the exact official library/tag pages |
| Cloudflare Workers AI | `https://developers.cloudflare.com/workers-ai/platform/pricing/` |
| OpenRouter gateway | `https://openrouter.ai/api/v1/models` |
| Tinker native | `https://tinker-docs.thinkingmachines.ai/tinker/models.json` |
| Tinker serverless | `https://tinker-docs.thinkingmachines.ai/tinker/serverless.json` |
| Tinker retirements | `https://tinker-docs.thinkingmachines.ai/tinker/model-deprecations/` |
| Perplexity Agent API | `https://docs.perplexity.ai/docs/agent-api/models` |

- **OpenRouter:** 291 of 296 exact bundled IDs matched the public feed. Rates
  convert from dollars per token to dollars per million using Decimal arithmetic.
  Raw extra fields and structured context/UTC overrides survive. Supported
  override conditions are represented as schedules. All 278 published paid
  cards are enabled by default, not configuration-gated. They estimate advertised
  resource/list value rather than claiming a pinned endpoint, tier or BYOK bill;
  that limitation is disclosed in assumptions. Prefer actual `usage.cost`,
  including zero. The API contract specifies strictly greater-than prompt
  thresholds and later matching overrides winning per price key; missing keys
  inherit their earlier/base values. UTC windows and weekdays remain conditional.
  Extra billed quantities, including override-only audio meters, must not
  silently disappear from coverage. Unmetered extras conservatively keep the
  result partial. A `:free`
  suffix is insufficient: all published token, override, and extra charges must
  be zero before the record is treated as verified free. A final public-feed
  comparison matched all 291 exact pricing payloads; two changed rows,
  `moonshotai/kimi-k3` and `openai/gpt-oss-120b`, were refreshed with new hashes.
- **Tinker:** all 34 exact native bases matched `models.json`. Native
  prefill/cached-prefill/sample rates are not serverless rates. Temporary
  discounts retain regular rates and their unpublished-expiration limitation.
  `Qwen/Qwen3.6-27B`, `deepseek-ai/DeepSeek-V3.1`, and
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` are marked retired on October 23,
  2026 even if still returned by the feed. Their capabilities remain untouched.
  Saved checkpoints are intentionally unpriced by default, even with a resolved
  native base. They never inherit the base card automatically. Their token
  consumption still counts; exact-checkpoint operator overrides and valid
  provider-reported charges remain supported.
- **Thinking Machines serverless:** the public feed prices suffixed
  `:peft:262144:sampling-nvfp4` billing IDs. The supported unsuffixed Inkling
  aliases use those published serverless rates by default, with the exact
  alias-to-billing-ID assumption disclosed. Inkling input/cache-read/output is
  USD 1.00/0.17/4.05 per million; Inkling-Small is 0.30/0.06/1.20. Cache reads
  are separately priced, not included at the full input rate. Cache creation
  is interpreted as ordinary published input pricing; this is recorded as an
  assumption, not a separately documented zero write fee. Unknown cache-read
  counts therefore prevent a complete total. Native Tinker rates are never
  substituted.
- **Fireworks:** three of nine exact bundled IDs have current public serverless
  rates: `kimi-k3`, `minimax-m3`, and `ember-1`. Input/cached-input/output USD per
  million is respectively 3/0.30/15, 0.30/0.06/1.20, and 3/0.30/15 in Standard;
  Priority is 3.75/0.375/18.75, 0.45/0.09/1.80, and 3.75/0.375/18.75.
  Batch retains the published half-price input/output rule, without inventing
  an unlisted cached Batch discount. Fast and US-only use separate router IDs:
  `accounts/fireworks/routers/kimi-k3-fast` and `.../kimi-k3-us`. Their rates
  must not be selected as service tiers of `accounts/fireworks/models/kimi-k3`.
  The changelog retires `glm-5p2` and `kimi-k2p7-code` on September 25, 2026,
  unsuffixed `deepseek-v4-pro` on August 27, 2026, and unsuffixed
  `deepseek-v4-flash` on August 14, 2026. These are serverless retirements;
  dedicated deployments are not thereby retired. The exact `glm-5p1` page
  marks serverless unsupported; `qwen3p7-plus` marks Serverless/On-demand
  unsupported and says Contact Us for dedicated deployment. Neither establishes
  a public token rate. Successor prices and GPU-second prices are not substitutes.
  Capabilities and ordering were intentionally left untouched; no new models
  or live capability probes were added.
- **Perplexity:** 25 exact bundled IDs matched public Agent API tables, including
  cache reads, applicable whole-request length bands, and documented tier
  multipliers. The eight GPT routes with service-tier schedules use the
  documented `default` tier when omitted; `auto` uses the same standard rates.
  Hosted Search/Fetch charges require their own tool-call
  quantities; tool fees are not service-tier multiplied. The public `/models`
  reference documents ID-only entries. An unauthenticated live request returned
  401, so no authenticated pricing response schema was independently verified.
  Only explicitly USD-per-million denominated feed pricing is accepted; absent,
  malformed, or undocumented-unit pricing falls back to the exact reviewed card,
  preserving its real source-check date. No native-vendor price substitution.
- **DeepSeek:** default off-peak cards plus Beijing weekday 09:00–12:00 and
  14:00–18:00 peak schedules; Chinese public holidays are excluded. Make-up
  weekend workdays remain weekends under the published Monday–Friday wording.
  The sourced calendar ends December 31, 2026; later dates require review rather
  than guessed holiday applicability. Successful Search and Fetch requests
  change from free to USD 0.00075 each at October 31, 2026 16:00 UTC, which is
  November 1, 2026 00:00 Beijing time.
- **Ollama:** its independent weekday 12:00–18:00 UTC peak rule has no Chinese
  holiday exception. Official library tag pages establish the tagged-to-price-row
  mappings, including `:0813`, `:1017`, `:31b`, `:675b`, and `:30b`.
  Separate cache-read rates are retained when published. A dash for cached input
  on `mistral-large-3:675b` or `nemotron-3-nano:30b` does **not** establish a free
  or included cache-read rate; those components stay explicitly unknown.
  Explicit zero cache reads need no cache price and do not create a blanket
  request gap. Nonzero or unknown cache reads still produce a partial estimate.
  Cache-write inclusion is a stated interpretation of the listed input pricing.
  Beam's official temporary free promotion has no published end date or regular
  rate, and is explicitly labeled as requiring review.
- **Z.AI:** current supported Coding Plan aliases use documented credit weights
  and Singapore weekday 14:00–18:00 peak scheduling. Credits have no universal
  monetary conversion. The separately documented API USD rates are retained
  only under non-applied `alternative_billing_surface` metadata.
- **Kimi Code:** membership quota and account-priced Extra Usage are not exact
  Moonshot API prices. The documentation's approximate comparison does not
  justify a USD token-rate substitution.
- **MiniMax:** M3's standard/priority whole-request bands are retained. The
  source's `512k` boundary is explicitly interpreted as 512,000 input tokens.
  The Anthropic-compatible API reference documents omitted `service_tier` as
  `standard`, which is retained as the M3 card's default billing context.
  Cache writes have no verified separate current rate in this card; an absent
  charge is not fabricated as zero.
- **Workers AI:** exact official per-model input/output/cache rates are USD
  resource/list value, not the session's incremental cash after account-wide free
  neuron allocation or credits.

Taxes, negotiated discounts, training, storage, subscription amortization,
external tools/MCP billing, and unknown account allowances are excluded.
Provider-reported request charges should take precedence over estimates.

## Focused default-request correction

Rechecked November 1, 2026 against official primary sources:

- `https://platform.minimax.io/docs/api-reference/text-anthropic-api`: MiniMax's
  supported Anthropic-compatible route documents `service_tier` as optional and
  defaulting to `standard`; `priority` is explicit. The M3 card now records that
  `default_context` and the API reference as provenance.
- `https://docs.perplexity.ai/docs/agent-api/models`: Perplexity documents `auto`
  and `default` as standard processing; omission selects default processing.
  Its eight public GPT rows with tier multipliers now record `default_context`
  with `service_tier: default`. Standard bands match only `default`/`auto`, while
  Flex/Priority/Fast bands retain their published multipliers. MiniMax's standard
  bands likewise match only `standard`, preventing ambiguous overlap with
  explicit Priority requests.
- `https://ollama.com/pricing`: the Mistral Large 3 and Nemotron 3 Nano rows still
  show a dash for cached input. Their rate and cache-policy fields remain absent.
  The source caveat is an assumption, not unconditional `missing_components`;
  the existing resolver requires a rate only for nonzero/unknown cache reads.
  Their existing cache-write-included policy is unchanged.

No unknown cache charge is invented as zero. Regressions cover all nine
documented tier-default cards and both Ollama rows
with November 1, 2026 requests, `tools=[]`, input 1,000/output 100, and explicit
zero cache counts. They also cover explicit tier overrides, missing documented
defaults remaining gaps, nonzero/unknown unpriced cache reads, and nonzero
unpriced cache writes remaining partial. xAI `usage_violation` gaps
are deliberately untouched.

## Exact unresolved rate rows

These 38 model-specific exceptions are deliberately visible, not silently
filled with a sibling model or another provider's price.

### MiniMax: 7

The current live API price and supported-model references list M3.1-Flash and M3.
Older indexed official pages and launch material contain legacy observations,
but do not establish a currently applicable exact cache/tier card for these
Anthropic-compatible routes. Some indexed URLs redirect to updated pages.

```text
MiniMax-M2.7
MiniMax-M2.7-highspeed
MiniMax-M2.5
MiniMax-M2.5-highspeed
MiniMax-M2.1
MiniMax-M2.1-highspeed
MiniMax-M2
```

### OpenRouter: 5

These exact variants are absent from the checked public feed; an unavailable
`:free` variant is not assigned its sibling's rate.

```text
poolside/laguna-s-2.1:free
poolside/laguna-xs-2.1:free
poolside/laguna-s-2.1
poolside/laguna-xs-2.1
upstage/solar-pro-3
```

### Perplexity Agent API: 26

These exact routes are absent from current public price tables; authenticated
account access or a verifiable exact historical Agent API rate card is needed.

```text
anthropic/claude-haiku-4-5
anthropic/claude-opus-4-5
anthropic/claude-opus-4-6
anthropic/claude-opus-4-7
anthropic/claude-opus-4-8
anthropic/claude-sonnet-4-5
anthropic/claude-sonnet-4-6
google/gemini-3-flash-preview
google/gemini-3.1-flash-lite
google/gemini-3.1-pro-preview
google/gemini-3.5-flash
google/gemini-3.5-flash-lite
openai/gpt-5
openai/gpt-5-mini
openai/gpt-5-nano
openai/gpt-5.1
openai/gpt-5.2
openai/gpt-5.3-codex
openai/gpt-5.4
openai/gpt-5.4-mini
openai/gpt-5.4-nano
perplexity/nemotron-3-ultra-550b-a55b
xai/grok-4.20-multi-agent
xai/grok-4.20-non-reasoning
xai/grok-4.20-reasoning
xai/grok-4.3
```

## Refresh preservation and validation

Six runtime catalog transforms and all four corresponding generators preserve
JSON-safe pricing. Older capability-only caches remain readable. Malformed cached
pricing is removed without discarding valid capabilities. Invalid/absent live
prices do not overwrite an exact known reviewed card. Unknown overlay-only routes
receive an explicit sourced `unverified` card. The generators retain pricing
provenance, structured overrides, native retirement status, and checkpoint policy.

No startup/inference price fetch or new dependency was introduced. Checkpoint
base-model metadata resolution is part of the existing native sampling-client
setup, not a price-feed request or an additional inference.

Deterministic tests in `test_pricing_cards.py`, `test_pricing_feeds.py`,
`test_pricing_coverage.py`, `test_pricing.py`, and `test_tinker_provider.py`
cover inline cards, default estimates, schedules, cache/component gaps,
retirement/router boundaries, feed/cache/generator round-trips, and unpriced
checkpoint requests, including known bases and subsequent calls. All 534 inline
cards were independently shown equivalent to their pre-refactor pricing;
non-pricing values, field/model order, and shared
objects were unchanged. The subsequent documented pricing corrections are
intentional, not catalogue capability changes.

The zero-cache, no-tools default probe on November 1, 2026 gives **438 complete
estimates and 25 partial estimates** across all 463 priced cards. Five partial
cards are xAI usage-violation quantity gaps; 20 are OpenRouter cards with
unmetered extra billing components. Published token subtotals survive without
being mislabeled complete charges. See `acp-accounting-verification.md` for the
final hermetic test, type-check, lint and documentation-build evidence and the
boundary on earlier paid verification.
