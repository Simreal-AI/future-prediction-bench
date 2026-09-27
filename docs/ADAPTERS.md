# Model and research adapters

The model chooses actions; a trusted research provider retrieves evidence; the environment validates and stores observations and submissions. A source collector creates and resolves questions through a separate administrative interface. These roles should not share access to private outcomes or baseline records.

## Included live providers

`ChatCompletionModel` supports a nonstreaming Chat Completions-compatible endpoint configured through `FPB_MODEL_BASE_URL`, `FPB_MODEL_API_KEY`, and `FPB_MODEL_NAME`. The endpoint must support the request fields and function-calling shape the adapter sends; provider compatibility still needs a real smoke run.

`BraveResearchProvider` uses `FPB_BRAVE_API_KEY` for search snippets and a separate `PublicPageReader` for text/HTML. The reader does not execute JavaScript. It limits response bytes, text length, redirects, and time; its transport restricts agent-selected URLs to public HTTP(S) destinations on ports 80/443, checks every redirect, and pins a validated DNS address for the request. These mechanisms do not replace source-policy review or audit of the content returned.

The model runner adds step, call, output-token allowance, prompt-character, and wall-time limits, and stores credential-free configuration plus model turns and reported token usage. Its output-token allowance reserves the requested amount on each call, even when the provider omits usage. Token usage is not a price calculation; complete cost accounting remains deployment work.

The fixed fallback baseline uses separate `FPB_BASELINE_BASE_URL`, `FPB_BASELINE_API_KEY`, and `FPB_BASELINE_MODEL_NAME` values. Its current prompt uses only public question fields and existing model knowledge, with no search tools or evaluated-agent predictions. Its model/prompt configuration and actual output are stored privately.

See [`.env.example`](../.env.example) for the complete environment variable list. The CLI reads process environment variables and does not automatically load an `.env` file. Source collection and offline tests do not require model or search credentials.

## Model action contract

```json
{"action": "search", "query": "official outlook and relevant historical observations"}
{"action": "open", "url": "https://example.org/report"}
{"action": "submit", "probabilities": {"option_a": 0.25, "option_b": 0.75}}
```

A model adapter passes its chosen action to `PredictionEnv.step`. Every option must appear in the final distribution and probabilities must sum to one within an absolute tolerance of `1e-6`. The runner supplies identity and time information; it does not trust a model-reported submission time or reduce a distribution to its most likely option.

Only public question fields and tool observations belong in the model input. Do not expose the store, provider internals, injected test clock, resolver, private baselines, or future outcomes. Page content is evidence, not a source of execution instructions.

## Research provider contract

```python
from future_prediction_bench.research import ResearchSession

class ExampleProvider:
    def search(self, query: str) -> list[dict]:
        return [{"url": "https://example.org/report", "title": "Report",
                 "text": "A stored search snippet.",
                 "published_at": "2026-09-21T10:00:00Z"}]

    def open(self, url: str) -> dict:
        return {"url": url, "title": "Report", "text": "A stored page snapshot."}

# question must be an eligible, validated question.
# session = ResearchSession(question, ExampleProvider(), max_calls=8)
# observation = session.search("relevant primary-source evidence")
```

`search` returns a list of dictionaries. Each search result and the `open` result must contain string fields `url`, `title`, and `text`; `published_at` is optional. `open.url` must be the final URL after redirects so the session can filter it again. The trusted provider is responsible for the truthfulness of the URL and publication metadata. Extra fields are not automatically passed to the model.

A publication timestamp must include a timezone and cannot be later than server observation time. Missing publication time is allowed and labeled `published_at_status="unknown"`. An old publication date does not prove that the page has remained unchanged.

## Time, budget, and failure behavior

The trusted clock must satisfy `issued_at <= now < forecast_deadline`. Research calls are checked both before retrieval and after the provider returns. Results arriving at or after the deadline are discarded and a blocked observation is recorded.

`max_calls` limits total tool calls. Failed calls within the open window and budget still consume a call; excess attempts are audited. A successful search must return at least one permitted result. Empty, fully filtered, or late results do not meet the self-research requirement.

The session does not enforce token, cost, response-size, per-query result-count, or concurrency limits, and cannot cancel arbitrary synchronous providers. Network providers and the outer model runner must enforce their own request timeout and resource limits. The runner must also control other browser, terminal, and network capabilities available to the model.

## Market information conditions

- `no_consensus`, the default, filters known prediction-market and odds domains plus heuristic patterns for explicit consensus probabilities. Searches omit filtered records and retain only counts/reasons. Opens check both the requested URL and the final returned URL and text.
- `market_aware` permits these observations and records detected market-source and consensus-exposure flags. Report this condition separately.

Known domains and text patterns are incomplete. Mirrors, quotations, new domains, and unrecognized language can evade the rules; legitimate analysis can also be filtered. No-consensus results must not be described as guaranteed free of market information. A real provider must additionally handle redirects, network access controls, request limits, and source permissions.

A baseline used for reward calculation is stored privately regardless of its origin. Enabling a baseline must not append its probabilities, provenance, or source response to a no-consensus agent's question or research observations. See [baseline design](PROJECT_DESIGN.md#private-baselines-and-rewards).

## Observation storage and replay

Each string-argument tool call produces an action and an observation, including failures:

```json
{"type": "action", "tool": "search", "at": "2026-09-22T03:00:00Z",
 "observed_at": "2026-09-22T03:00:00Z", "payload": {"query": "outlook"},
 "loss_mask": 1, "sha256": "..."}
{"type": "observation", "tool": "search", "at": "2026-09-22T03:00:01Z",
 "observed_at": "2026-09-22T03:00:01Z",
 "payload": {"success": true, "results": [{"url": "https://example.org/report", "title": "Report", "text": "Observed evidence."}], "filtered_count": 0, "filter_reasons": {}},
 "loss_mask": 0, "sha256": "..."}
```

The example shows the event shape; real observations also contain snapshot metadata. Search observations store full permitted result dictionaries. A search with no permitted results does not count as successful research. Successful open observations use `result`. Blocked observations have `success=false`, `status`, and `reason`. Provider exceptions expose only their type, preventing raw exception messages from leaking retrieved content.

An event hash covers the full event with its top-level `sha256` field removed, serialized as UTF-8 JSON with `ensure_ascii=False`, `sort_keys=True`, `separators=(",", ":")`, and `allow_nan=False`. Result hashes use the same rule. Hashes detect content changes; they are not signed timestamps or proof of historical availability. Stored, returned, and provider-owned objects are deep-copied apart.

Replay uses the persisted observations, including the exact snippet or page text, timestamps, filtering records, and hashes. It must not refresh a page to replace the original evidence. Tool observations have event-level `loss_mask=0`; a real trainer must translate that policy into token-level masks.

Historical evaluation requires evidence actually captured at the historical cutoff with auditable provenance. Moving a test clock into the past, reading a currently available old URL, or trusting a displayed publication date does not create a valid historical snapshot.
