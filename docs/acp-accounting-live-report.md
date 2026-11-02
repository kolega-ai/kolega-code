# Bounded live ACP accounting verification — November 1, 2026

**Final campaign: 11 actual inference requests / 18 allowed; USD 0.04917550
conservative list-value safety debit / USD 1 allowed.** The initial full
selection failed for the explicitly retained harness/bootstrap reasons below;
the final two-request Coding Plan-only selection **passed**. No paid requests
were repeated to repair the initial child-origin false negative.

## Scope and execution

The probe launches the real worktree interpreter with
`sys.executable -m kolega_code.cli.main acp`, isolated settings/home/project/state,
the real ACP factory and provider SDKs, and no user pricing overrides. A local
TLS forwarding guard passes real provider request/response payloads unchanged,
counts actual inference POSTs, prevents identical-payload SDK retries, stops a
route after an HTTP failure, and reserves maximum output/list charges before
forwarding. It does not replace the factory, agent client, or provider responses.
The only served fixture is a small public-style document used by `web_fetch`;
its answering request uses the live provider.

The campaign limit is 18 actual inference requests and USD 1 conservative
published list value. Short prompts request one-word replies except the single
explicit child/helper tool prompt. ACP currently does not forward its generic
CLI context/output-budget flags into the factory, and OpenRouter deliberately
omits generic output caps. The selected DeepSeek gateway route re-adds an actual
64,000-token cap; native DeepSeek has a 384,000-token cap. These are not small
custom limits: the guard reserves their full maximum charge. No source changes
were made to alter these behaviors.

Credentials were parsed using `dotenv_values` and passed only as the selected
provider's environment variable. No shell sourcing, dotenv copying, credential
logging, account-balance queries, or invoice deltas were used. Saved evidence
contains whitelisted usage, costs, pricing snapshots, SDK versions and lineage;
request bodies and authentication headers are not saved.

SDK versions: ACP **0.12.0**, OpenAI **2.46.0**, Anthropic **0.101.0**.
The saved default was `openai_chatgpt/gpt-6.1-sol`; the probe explicitly selected
current catalog routes to cover the required billing surfaces and cleared
role/fast-slot overrides in its isolated settings.

## Verified live: OpenRouter gateway/BYOK

Route: `openrouter/deepseek/deepseek-v4.1-flash-1017`.

- **7 actual inference POSTs**, all HTTP 200.
- First prompt: one root request.
- Second prompt: root tool selection, real `web_fetch` helper inference, real
  general-agent child inference, and final root inference. Both tools were
  requested once; the paid tool prompt was never repeated.
- Child evidence is a `history` origin with `depth=1`, child `agent_id` and
  `parent_tool_call_id`; helpers have `kind=helper, helper=web_fetch`. Requiring
  a literal `kind=sub_agent` would incorrectly reject this recorder path.
- A later root stream was held immediately after forwarding its real provider
  usage frame, before stream EOF; ACP cancellation preserved **10,967 input +
  7 output = 10,974 tokens**, with terminal `failed/CancelledError` and no
  duplicate settlement.
- The subprocess was stopped, restarted, the same session loaded, and another
  prompt completed. The run ID changed; frozen lifetime usage/cost persisted
  without duplicate counting.
- Final lifetime **60,076 input + 205 output = 60,281 tokens**, including
  **42,496 cached-read** and **25 reasoning** tokens as subsets.
- Final root context occupancy was **9,600**, not cumulative consumption.
- Provider wire usage/charges, compact journal facts and prompt/session ACP
  metadata independently reconciled.

This account routed BYOK: all seven authoritative `usage.cost` values were
explicit **USD 0**, scoped to `gateway_request_charge`. Separate
`cost_details.upstream_inference_cost` values totaled **USD 0.007474368**.
They are retained as breakdown evidence and **not added to the gateway total**.
This verifies a real reported monetary zero and BYOK scope, not a nonzero
gateway charge or a free upstream service.

Pinned catalog source:

- Public source: `https://openrouter.ai/api/v1/models`.
- Pricing ID:
  `openrouter:deepseek/deepseek-v4.1-flash-1017:2026-11-01:9ae93d6e305c1a43`.
- Source SHA-256:
  `9ae93d6e305c1a43a4f0c2cf820bb5576bd44e2ef399e4dea4eb971e06e5ae65`.
- Listed off-peak USD/million: input **0.15**, output **0.60**, cache read
  **0.003**. The card explicitly does not treat gateway model-list rates as an
  exact routed charge; provider-reported gateway cost took precedence.

## Verified live: native DeepSeek catalog-only estimate

Route: `deepseek/deepseek-flash`, native `/responses`.

- **2 actual inference POSTs**, both HTTP 200; no user rate overrides.
- Wire and compact facts reconcile with ACP lifetime **21,334 input + 4 output
  = 21,338 tokens**, including **10,496 cache-read tokens**.
- Each request pins catalog pricing at begin; provider wire usage supplies no
  monetary total, so costs are `estimated`, `pricing_source=catalogue`.
- Independent `Decimal` arithmetic selects the request-time Shanghai
  calendar/time band from the pinned public card, not the current headline
  model or an account-balance difference.

Applicable off-peak USD/million: uncached input **0.15**, cache read **0.003**,
output **0.60**. Cache writes are included rather than fabricated as known zero.
There were no hosted Search/Fetch requests.

```text
request 1 = (10,659 × 0.15 + 2 × 0.60) / 1,000,000
          = USD 0.001600050

request 2 = ((10,675 − 10,496) × 0.15
             + 10,496 × 0.003 + 2 × 0.60) / 1,000,000
          = USD 0.000059538

lifetime catalog estimate = USD 0.001659588
```

Pinned pricing ID: `deepseek:deepseek-flash:2026-11-01`, source model identifier
`DeepSeek-V4.1-Flash-1017`, checked November 1, 2026. Official sources retained
in the request snapshot:

- `https://api-docs.deepseek.com/quick_start/pricing`
- `https://www.gov.cn/zhengce/content/202511/content_7047090.htm`

This is published resource/list value, not a verified invoice or net debit.

## Harness failures and bounded continuation

The first complete live selection ended **failed**, not green:

1. Its child-origin assertion incorrectly required `kind=sub_agent`; the real
   recorded child has full `history` lineage/depth. This assertion was corrected
   and the captured evidence adjudicated offline, with **zero repeated paid
   tool requests**. The original failure remains in the JSON evidence.
2. Z.AI failed before any inference because its local token counter needed a
   public vocabulary file blocked by the allowlisted proxy.

A Z.AI-only supplemental preflight also made **zero inference requests**:
warming `cl100k_base` and `o200k_base` was insufficient because this compatible
provider uses **`p50k_base`**. After warming that exact public vocabulary in
session scratch, the native local counter was verified using a fake key before
selecting the still-unattempted live Coding Plan leg.

The original forwarding guard released OpenRouter reservations using gateway
reported zero, omitting separate upstream/list value from its conservative
debit. This was a **probe-budget accounting error**, not a product ledger
arithmetic failure. Offline review preserved all original wire facts and
corrected the conservative debit from **USD 0.00640500** to
**USD 0.04294260**. For each gateway request it uses the maximum of gateway
charge, separate upstream charge, and inclusive uncached tokens at doubled
peak rates; native DeepSeek uses peak rates without cache discounts. Each
corrected request debit is below its original reserved maximum, and replaying
these corrected debits plus every next reservation remains below USD 1.
The gateway guard was corrected before subsequent selections. Supplemental runs inherit
the prior nine-request count and corrected conservative debit.

## Checks and evidence

- Real deterministic native-route prerequisite, selected foreground subset:
  **8 passed, 1 background test deselected** before live spending.
- Full deterministic subprocess harness after its owner corrected synchronization:
  **9 passed**. Earlier cache/bootstrap and background-harness failures were
  observed and not treated as green prerequisites.
- New live module carries all `integration`, `live_api`, `e2e` markers and
  requires `KOLEGA_RUN_LIVE_ACP_ACCOUNTING=1`; default collection skips it.
- Ruff and focused Pyright pass.

Artifacts:

- `tests/acp/test_accounting_live.py` — explicitly selected bounded runner.
- `docs/acp-accounting-live-evidence.json` — initial nine real requests,
  original failed-selection status and clearly labeled offline adjudication.
- `docs/acp-accounting-live-subscription-evidence.json` — zero-inference
  tokenizer preflight failure.
- Coding Plan supplemental evidence/result is recorded separately; see the
  final section below.

## Verified live: Z.AI Coding Plan non-monetary surface

Final supplement: **1 test passed in 22.26s**, **2 actual inference POSTs** to
`api.z.ai/api/anthropic/v1/messages`, HTTP 200, route `zai/glm-5.3-flash`.
There were no retries, guard refusals, outstanding reservations, or wire errors.

- Wire tokens independently reconcile with compact persisted facts and ACP:
  **20,703 input + 22 output = 20,725 tokens**. Cache reads were known zero.
- Each request pins `zai:glm-5.3-flash:2026-11-01`, billing mode `coding_plan`,
  denomination `zai_coding_plan_credit`, **currency absent**.
- Official source pins:
  `https://docs.z.ai/devpack/overview` and
  `https://docs.z.ai/guides/overview/pricing`.
- Independent `Decimal` arithmetic using the pinned credit rates (115 input,
  400 output, 28 cache-read credits/million) matches both recorded estimates:

```text
(10,344 × 115 + 3 × 400) / 1,000,000  = 1.190760 credits
(10,359 × 115 + 19 × 400) / 1,000,000 = 1.198885 credits
total                               = 2.389645 credits
```

- Both authoritative prompt/session cost metadata and all **17 retained native
  `UsageUpdate`s** omit monetary cost. The last native update carries lifetime
  20,725 tokens and root context occupancy 11,796. Credit estimates are present
  under `non_monetary`; there is no misleading USD zero or fabricated conversion.
- `docs/acp-accounting-live-coding-plan-evidence.json` retains the successful
  live result and clearly labeled subsequent offline verification.

The audit found the original guard also reserved **zero** for this
non-monetary route. Product accounting is correct to keep credits out of USD,
but the safety ledger must still bound its conservative list value. For the
campaign ceiling only, double the card's explicitly separate API input/output
USD prices (0.15/0.50 per million) supplies a conservative **counterfactual API
list-value proxy**, without cache discounts. This is **not** the Coding Plan's
actual monetary charge or a credit conversion. Corrected maximum reservations
are **USD 0.14491250** and **USD 0.14497130**; replaying each against prior
corrected debits stays below USD 1. Corrected settled proxy values are
**USD 0.00310620 + 0.00312670 = 0.00623290**. Added to the original nine-request
conservative debit, final campaign value is **USD 0.04917550**. The original
zero-reservation wire rows and original total remain preserved beside an
explicit offline correction. The runner now applies this safety proxy; that
last guard adjustment was checked offline, not with a paid replay.

## Limits / not claimed

The 11-request / USD 1 guard covers **only this explicitly selected campaign**,
not every command run during repository verification. A separate broad repository
test run inadvertently included legacy credential-gated live tests: test-module
dotenv discovery reached the ancestor checkout's environment despite `CI=1`.
That run finished with 7,933 passed, 279 skipped and one live DeepSeek exact-text
assertion failure (`vision-阅读器\nvision-ok` rather than `vision-ok`). An overlapping
integrated run was stopped when this was identified. These legacy runs did not
use the campaign's forwarding guard, and their actual request count and charges
were not retained; the campaign's ceiling cannot honestly be claimed for them.
Subsequent offline verification disables dotenv with
`PYTHON_DOTENV_DISABLED=1`, starts without provider-key environment variables,
and excludes `integration` and `live_api` from the broad repository selection.
No paid retry was used to reproduce or repair that unrelated live assertion.

No editor automation or display smoke test was performed within this paid
campaign. The separate `acp-editor-smoke.md` records the completed actual Zed
context/cost display check and VS Code's display limitation, using loopback
provider responses and no paid requests. The initial
live report captures authoritative prompt/session metadata, not a retained
native `UsageUpdate.cost` trace; native monetary update serialization was
verified in the deterministic subprocess harness. The runner now retains and
asserts sanitized native usage-update summaries; the final Z.AI supplement
exercised the non-monetary branch. The subsequent stronger Z.AI token-pair and
nonempty-update assertions were verified against its saved real evidence
offline, not with additional provider calls.

Nonzero OpenRouter gateway charges, xAI ticks conversion, every other provider,
SDK-hidden remote retries, live background settlement during another prompt,
account-specific subscription-to-USD conversion, taxes, external tool bills,
and actual invoices remain **unverified by this live campaign**. A real BYOK
reported zero is not evidence for nonzero monetary-field behavior.

No product source defect was demonstrated by the successful wire/journal/ACP
reconciliations. The harness failures and budget-audit correction above remain
visible rather than being hidden by a paid rerun.
