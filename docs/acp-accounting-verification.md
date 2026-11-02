# ACP accounting implementation verification

Verified November 1, 2026, on branch `feat/acp-usage-accounting`, based on
`7582c17e`. PR publication checks were rerun on November 2, 2026.

## Implemented contract

- **Independent context occupancy:** latest root measurement/estimate and full
  effective capacity, separate from lifetime inference consumption. Context
  generations are fenced across rebuilds; delegates do not overwrite root context.
- **Request-owned lifetime accounting:** root, delegate and helper inference;
  captured logical prompt/lineage; retained cancellation/failure evidence; late
  evidence refinement without duplicate requests; background settlement attributed
  to its launch prompt. Rebuild, rewind and resume do not subtract consumption.
- **Shared costs:** valid reported charge, exact override, exact catalogue, then
  explicit unavailable status. Decimal strings, begin-time pricing snapshots,
  distinct requested/resolved models and billing surfaces, no historical repricing,
  no currency/credit conversion, and no accounting-induced inference retry.
- **Honest aggregates:** known partial/token-cost subtotals survive ledger,
  journal, ACP and CLI folding without becoming complete native totals. Explicit
  reported/estimated zero retains provenance. Missing legacy source-count splits
  remain unknown rather than being inferred from amounts.
- **Durability and protocol:** v2 request facts, deduplicated baseline-plus-live
  folding and persistence coverage; ACP SDK stays at `0.12.0`; authoritative
  `_meta["dev.kolega/usage"]`; native prompt usage requires `disjoint-v1`.
- **User surfaces and maintenance:** CLI/TUI/ask use the shared resolver;
  `models list --pricing` and additive JSON expose rates/status/provenance;
  explicit refresh preserves reviewed capabilities while updating validated
  pricing. No startup/inference price fetching was added.

See `src/content/docs/configuration/usage-pricing.md` for configuration,
protocol semantics and pricing limitations.

## Verification results

| Check | Result |
| --- | --- |
| Broad hermetic repository selection before checkpoint-pricing exemption | **8,224 passed, 22 skipped**, 49 warnings |
| Final pricing cards/feeds/resolver/coverage/native Tinker selection | **551 passed, 1 skipped** |
| Final client/ledger/accounting/ACP selection after checkpoint-pricing exemption | **157 passed, 4 skipped**, including the nine real ACP subprocess scenarios |
| PR publication selection: ACP, pricing, native Tinker, request accounting and CLI persistence/configuration | **859 passed, 1 skipped**, with credentials removed and dotenv disabled |
| Unpriced checkpoint regressions | **6 passed**, included above: generation/streaming × known/unknown/unavailable base, first and subsequent requests |
| Real ACP subprocess scenarios rerun after follow-up | **9 passed** |
| Independent inline/source comparison | **534 literal cards**, identical non-pricing graph/order/sharing; **291 exact OpenRouter price payloads** match public feed |
| Import-time network block | **0 attempted connections** while importing catalogue and six refresh transforms |
| Full Pyright | **0 errors, 0 warnings** |
| Ruff lint and formatting | Passed; **888 files already formatted** |
| PR publication all-files pre-commit hooks | All nine hooks passed, including Pyright, private-key detection and closed-source host-internals checks |
| `git diff --check` | Passed |
| Documentation production build | Passed; **40 pages** |
| Actual CLI follow-up inspection | OpenRouter JSON: 296 rows, 291 enabled public prices independently matched; Fireworks scheduled pricing/status inspected |
| Earlier actual editor wire reconciliation | Two turns each in Zed and VS Code independently reconciled; not rerun for pricing follow-up |

These selections overlap; their counts must not be added as unique tests.
The broad-suite result predates the accepted checkpoint-pricing exemption.
The final focused pricing and client/accounting/ACP selections, type/lint checks
and documentation build were rerun after that change; checkpoints remain an explicit cost gap rather
than inheriting a resolved base-model price.
The 49 broad-suite warnings include existing SDK/deprecation, pytest-marker and
unawaited-coroutine warnings; this is not a warning-free repository run.

The earlier broad command, from the isolated worktree with no provider credential
environment variables, was:

```sh
env -i PATH="$PATH" HOME="$HOME" CI=1 PYTHON_DOTENV_DISABLED=1 \
  .venv/bin/python -m pytest -q -n 4 -m 'not integration and not live_api'
```

`CI=1` alone is insufficient to prevent legacy live tests from running.
Test-module dotenv discovery can reach an ancestor checkout's `.env`.
The relevant cross-layer and real-subprocess selections were also rerun with
dotenv disabled. Loopback subprocess fixtures use isolated state, fake keys and
an external-socket block rather than a substituted agent factory.

## Pricing coverage

The audit covers **22 providers, 533 bundled models and one wildcard policy**:
**463 priced records**, **71 sourced exceptions**, **zero validation/coverage
errors**, and zero cards older than 30 days at the verification date.

This does **not** mean every route has a verified usable rate. **38 exact rows
remain unverified**; other exceptions explicitly describe subscription,
account-configured, retired or unpublished billing surfaces. Priced records also
include documented credit denominations, not just monetary USD estimates.
Do not infer a sibling or native-vendor price for them.

The deterministic default-rate probe produced **438 complete estimates and
25 partial estimates** among the 463 priced cards, using November 1, 2026 at
12:00 UTC, input 1,000/output 100, explicit zero cache counts and no tools.
Five partial cards are xAI routes whose `usage_violation_requests` quantity is
not established by ordinary usage evidence; 20 are OpenRouter cards with
unobserved extra image/audio or other billing components. Their known token
subtotals are retained, but they are not
misrepresented as complete request totals.

Exact model-specific statuses, source references and reasons are in
`pricing-coverage-report.json`; supplementary research is in
`pricing-audit.md`. All 534 cards are inline in existing provider entries;
the A/B modules and injection layer were removed. Non-pricing values, field
and model order, and shared object relationships still match the saved
pre-refactor catalogue graph after the follow-up.

The 278 paid OpenRouter cards and both Thinking Machines serverless alias cards
now estimate by default, with route/alias assumptions rather than configuration
gates. Cache reads use published discounts; unsupported component quantities
remain coverage gaps. Saved Tinker checkpoints are deliberately unpriced by
default, even with a known base; token accounting and ordinary base-model pricing
remain intact. Exact-checkpoint overrides and provider-reported charges remain
supported. Fireworks retains only exact applicable serverless rates,
with four dated retirements and two dedicated/contact-sales exceptions; Fast
and US-only router prices are not base-model service tiers. No capabilities
were refreshed and no new paid inference was run for this follow-up.

## Actual editor verification

`acp-editor-smoke.md` records versions, interactions, wire values, UI rounding,
fixture boundaries and reproduction instructions.

- **Zed passed actual native context and cumulative-cost display.** Two prompts
  produced 110 then 270 lifetime inference tokens, independent context occupancy,
  and catalogue estimates of USD `0.000021` then `0.000054`. The native tooltip
  rounded the second amount to `0.0001 USD`. Reopening an already-visible tooltip
  was needed to refresh its displayed value.
- **VS Code ACP Client 0.2.0 passed two-turn operation and safe ignored metadata,
  but does not display context/cost in the tested UI.** It is not counted as a
  display success.
- Both editors used the real ACP CLI/factory/provider pipeline with loopback
  provider responses, not a live paid provider or a hand-authored ACP usage
  object. Neither advertised native end-turn usage opt-in.

The parent verification independently reconciled both editors' captured prompt
responses, native updates and provider-request counts against the fixture and
independent Decimal expectations, and inspected the actual Zed tooltip screenshot.

## Live verification and important boundary

The explicitly selected accounting campaign exercised **11 actual inference
requests** against OpenRouter/BYOK, native DeepSeek and Z.AI Coding Plan.
Its corrected conservative safety value was **USD 0.04917550** against its
18-request / USD 1 ceiling. It verified real child/helper inference, cancellation,
restart/load, reported monetary zero, catalogue estimates and non-monetary credits.

**The campaign ceiling is not a bound on every verification command.** An earlier
broad run inadvertently executed legacy credential-gated live tests after ancestor
dotenv discovery. It finished with 7,933 passed, 279 skipped and one live DeepSeek
exact-text assertion failure; a separate overlapping integrated run was stopped.
Those calls did not use the campaign guard, and their request count and charges
were not captured. No paid replay was used to resolve that failure. The clean
hermetic reruns above replace those runs as the offline regression evidence,
but do not erase the unintended calls.

`acp-accounting-live-report.md` and its three sanitized JSON evidence files retain
the original harness failures, offline adjudications, budget-guard corrections
and exact route/token/rate provenance.

Not claimed: a nonzero authoritative gateway charge, retained native monetary
updates from the initial live routes, invoices/account debits, SDK-hidden remote
attempts, live verification of every provider, or a subscription-to-USD conversion.
Native monetary update serialization is covered by real subprocesses and the
actual editor test; the final live Coding Plan selection verifies its monetary
omission branch.
