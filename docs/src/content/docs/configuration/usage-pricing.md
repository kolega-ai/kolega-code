---
title: Usage and pricing
description: Session consumption, catalogue pricing, exact-route overrides, and cost coverage.
---

Kolega Code tracks context occupancy, inference consumption, and cost independently.
Context is the latest root-agent context measurement or estimate, not the sum of
tokens billed over a session. Consumption includes observed root, delegate, and
helper inference. Clearing or rewinding conversation history does not refund
previous consumption.

## How a request is priced

For each request, Kolega uses:

1. A valid provider-reported request charge, including an explicitly reported zero.
2. An applicable exact-route user override.
3. An applicable exact-route catalogue pricing card.
4. An explicit unavailable reason.

Catalogue estimates are **enabled by default**. They represent documented
resource/list value, not necessarily newly debited money after account-wide free
allowances, subscriptions, promotions, credits, or negotiated discounts. They are
not an invoice. Currency totals remain separate; provider credits are
non-monetary denominations, with no guessed USD conversion. Monthly subscription
fees are never amortized across guessed token volumes.

Prices, sources, and request-start billing context are pinned before inference.
Later model switches, refreshed catalogues, settings changes, and resumed
sessions do not reprice previously recorded requests. A provider charge replaces
an estimate; the two are not added.

Known token costs may be available even when a tool fee or cache partition is
unknown. Such a result includes component subtotals and missing-component
reasons, not a purported complete request charge. Missing counts are not zero.
Output already includes billed reasoning, and cache details are subsets of
inclusive input; neither is added again.

SDK-internal retries, unobserved historical work, account-wide cache storage,
taxes, and external tool/MCP charges cannot be reconstructed from request token
counts. These remain explicit limitations rather than fabricated amounts.

## Inspect and refresh prices

```sh
kolega-code models list --pricing
kolega-code models list --provider openai --json
kolega-code models refresh --provider openrouter
```

The default human model table is unchanged. `--pricing` adds billing status,
rates, schedule warnings, source identifiers, and check dates. JSON model rows
include a structured `pricing` object, or `null` if no valid exact-route card
exists. Scheduled prices are labelled as such, not presented as a universal
minimum.

Refresh is explicit. Startup and inference never fetch prices. Refreshable
catalogues can add models and replace validated **newer pricing only** for a
bundled model. Reviewed capabilities, order, defaults, and featured status are
not replaced. Old caches without prices remain useful for capabilities; invalid
cached pricing does not remove an otherwise valid model.

Sources older than 30 days are marked due for review, but their rates do not
silently become zero. Published expiration dates are respected. An expired
promotion needs a pinned documented regular-rate card or produces a pricing gap.
Sources are recorded per exact billing route: another provider serving the same
model name is not a pricing source for that route.

## Exact-route overrides

Add optional `usage_pricing` entries to your CLI `settings.json`. Keys are the
exact `provider/model` pair; everything after the first slash belongs to the
model ID. Nested `provider: {model: card}` mappings are also accepted.

```json
{
  "usage_pricing": {
    "custom:example/operator-model": {
      "currency": "USD",
      "unit": "per_million_tokens",
      "rates": {
        "input": "1.25",
        "output": "5",
        "cache_read": "0.125"
      },
      "cache_policy": {
        "cache_write": "unsupported"
      },
      "conditions": {
        "region": "us",
        "service_tier": "standard"
      }
    },
    "openai/exact-model-id": false
  }
}
```

This is illustrative operator-supplied pricing, not a claim about a real
provider. Rates must be finite, nonnegative **decimal strings**. A zero string
is an authoritative free rate; an absent field is not.

`false` (or `{"enabled": false}`) disables estimation only for that exact
route. It does not suppress a valid provider-reported charge. Invalid overrides
warn and fall back to valid catalogue pricing; they do not block inference.
An override with unmatched conditions similarly falls back. Request context
must actually establish required region/tier values; a missing value is not
silently guessed.

Optional full-card fields include `schema_version: 1`, `pricing_id`, `provider`,
`model`, `status`, `billing_mode`, `checked_at`, and `sources`. Omitted identity
fields are filled from the exact route; omitted provenance is marked as
operator configuration. To document an operator's source, supply
`sources: [{"url": "https://example.org/rate-card", "model_id": "operator-model"}]`.
Do not include credentials, private endpoint URLs, or account secrets.
Source URL query strings and fragments are not retained.

Custom endpoints require an exact operator override. Self-hosted routes have no
universal hardware/electricity rate.

## Tinker checkpoints and serverless aliases

A `tinker://...` identifier is a user's saved sampling checkpoint, not a shared
public model name. The wildcard catalogue entry lets Kolega accept arbitrary
checkpoint identifiers; it does **not** assign every checkpoint one flat price.
Saved checkpoints are intentionally **unpriced by default**, even when the
provider resolves a known base model. They never automatically inherit a
native base-model or serverless-alias price. Token consumption is still counted;
missing cost is not treated as USD zero and does not block valid inference.

The native provider still resolves the checkpoint's base for rendering and
context handling, but accounting does not initialize the sampling client just
to select a price. A valid provider-reported charge remains authoritative, and
an operator may supply an exact-checkpoint pricing override. Ordinary native
base models retain their published `models.json` prefill, cached-prefill, and
sample rates. Pinned pricing preserves recorded work across resume or later
refresh.

Native `tinker` and Anthropic-compatible `thinking_machines` are different
billing surfaces. The latter's Inkling aliases use published `serverless.json`
rates by default, explicitly recording the assumption that the supported alias
maps to its corresponding `:peft:262144:sampling-nvfp4` billing ID. Separately
published cache reads are discounted; cache creation at the normal input rate
is a disclosed interpretation. Native checkpoint training, forward passes and
storage are not added to sampling-request cost.

## Gateway and Fireworks boundaries

Published OpenRouter model-list rates work without an override. They estimate
list value, not a guaranteed endpoint/BYOK invoice. A valid reported
`usage.cost`, including zero, always wins. Prompt-length overrides are strictly
greater-than thresholds; later matching entries win per price key. Extra
unobserved image/audio/tool billing components leave a subtotal instead of a
fabricated complete total.

Fireworks cards use the exact supported serverless model ID, not GPU-second
dedicated-deployment rates. Standard/Priority and documented Batch token
schedules are distinct from Fast/US-only **router IDs**; choosing a tier on a
base model does not select a router. Retired IDs retain their capability entries
and sourced retirement dates rather than inheriting a successor's price.
Dedicated-only/contact-sales routes require their own applicable billing
configuration. This pricing audit does not add or probe new model capabilities.

## Card and schedule format

Version 1 cards use decimal strings and JSON-native objects. They include
`pricing_id`, exact `provider`/`model`, `status`, `billing_mode`, `checked_at`,
and source references (`url`, exact `model_id`, optional `checked_at`).
Non-priced statuses require a precise `reason`: `subscription`, `self_hosted`,
`requires_configuration`, `unpublished`, `retired`, or `unverified`.

Priced cards contain either an ISO currency code such as `USD` **or** a
provider-credit `denomination`. Rate units are `per_token`,
`per_thousand_tokens`, or `per_million_tokens`; component fees may use
`per_request`.

For disjoint cache pricing the formula is:

```text
((input - cache_read - cache_write) * input_rate
 + cache_read * read_rate
 + cache_write * write_rate
 + output * output_rate) / 1_000_000
```

`cache_policy` distinguishes `unsupported`, `included` (billed at base input),
`separate` (the default when a cache rate is documented), and `supplemental`
(the cache charge is additional to base input). Unknown cache quantities are
required when they change the result. Cache-write TTL rates may be named
`cache_write_5m` and `cache_write_1h`; calculation requires their exact observed
quantities.

`conditions` constrain an entire card. Each entry in `schedules` has a `when`
condition and replaces only its listed `rates`, or adds its listed
`components`. Relevant condition keys include:

- `service_tier`, `region`, `modality`, `cache_ttl`, `billing_surface`, `route`,
  and `byok`: exact values or lists of accepted values.
- `min_input_tokens` and `max_input_tokens`: inclusive whole-request length
  bounds, evaluated against the pricing-tier input count when provided.
- `effective_from` inclusive and `effective_to` exclusive: ISO timestamps.
- `timezone`: IANA timezone, default `UTC`; `weekdays`: Monday `0` to Sunday `6`.
- `start_time` inclusive and `end_time` exclusive: local `HH:MM`; an overnight
  interval may cross midnight.
- `dates`, `excluded_dates`, `holidays`, and `working_dates`: explicit ISO dates.
  Holidays count as Sunday and exceptional working dates as Monday for weekday
  selection. `calendar_complete_through` prevents using an expired holiday
  calendar as if its future exceptions were known.

Unknown condition keys produce a coverage gap instead of being silently
ignored. A schedule may declare `priority` to resolve overlapping rate choices;
conflicting overlapping schedules without distinct priorities are ambiguous.
`requires_schedule: true` makes a matching schedule mandatory.

Cards can declare a documented `default_context`, such as the standard service
tier or a provider's default cache TTL. Explicit request context takes
precedence, and any defaults used are disclosed in the estimate's assumptions.
These are published billing defaults, not guesses about account eligibility.

Whole-request tiers use the selected rate for the entire component.
Marginal schedules instead declare `mode: "marginal"`, `min_tokens` (exclusive
consumed-prefix bound, starting at zero), and optional `max_tokens` (inclusive
prefix bound). Each band bills only its segment. Bands must cover the quantity
without overlap. Marginal cache allocation needs explicit disjoint evidence;
it is not guessed from aggregate cache totals.

Tool and other attributable fees use `components`:

```json
{
  "name": "search",
  "quantity": "search_calls",
  "rate": "0.01",
  "unit": "per_request",
  "when": {"effective_from": "2026-11-01T00:00:00+08:00"}
}
```

Here `search_calls` must be an observed pricing quantity, not the count of
locally displayed tool messages. Distinct providers retain their own calendars;
serving another vendor's model does not inherit that vendor's off-peak terms.
Schedules use the pinned aware request-start timestamp. Provider billing-time
differences are disclosed; explicit uncertainty can prevent a complete amount
near a boundary.

Known-disabled hosted tools contribute zero only when the pinned request
explicitly establishes that they were not enabled. An enabled tool with an
unknown billable count leaves a pricing gap. Fractional non-token meters, such
as execution hours, use decimal-string quantities. A modality component can
declare `covers: "input"` or `covers: "output"`; its observed token quantities
must cover that normalized partition without also charging the ordinary token
rate. Missing modality splits do not turn into zero.

## Editor and session display

The TUI and `ask` combine an immutable loaded-history baseline with the live
ledger once. They identify incomplete coverage and keep currencies and credits
separate. Older uninstrumented history is not retroactively priced.

ACP publishes complete single-currency cumulative monetary costs when coverage
permits, including catalogue-backed estimates by default. Detailed source,
basis, partial subtotals, and pricing gaps are carried in
`_meta["dev.kolega/usage"]`; generic editors need not display that metadata.
An editor may display an estimate simply as cost, and may retain an older
complete amount when a later incomplete snapshot omits a native amount.
Consumers replace cumulative snapshots; they never sum repeated updates or add
a prompt snapshot to the session lifetime.
