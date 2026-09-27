# Analyst toolkit

The environment supports an evidence-gathering workflow: interpret the resolution rules, identify reference classes and base rates, read independent sources, seek counterevidence, calculate quantities, keep source-linked notes, revise a probability draft, and submit once. The model chooses the sequence. The workflow is guidance, not an extra reasoning-quality reward.

`analyst_tools_v1` uses the same schemas for model discovery and host argument validation. `PredictionEnv.reset()` returns the toolkit version and available action names. The autonomous runner includes the full tool schemas in every model request and its configuration identity.

| Action | Arguments besides `action` | Result and limits |
| --- | --- | --- |
| `search` | `query` | Public web snippets; existing Brave provider requires its own credential. At least one successful permitted web search is required in `self_research`. |
| `open` | `url` | Bounded public HTML/text extraction, with validated DNS and redirects. No authenticated browser, PDF parsing, or JavaScript execution. |
| `calculator` | `expression` | Bounded arithmetic with parentheses, `+`, `-`, `*`, `/`, and `**`. No arbitrary code, functions, files, or imports. |
| `notebook` | `claim`, `source_hashes` | Appends an agent-authored claim referencing sources observed successfully in this episode. At least one source hash is required. |
| `draft` | `probabilities`, `rationale`, `source_hashes` | Records a full tentative distribution, revision number, and previous distribution. Does not submit or reveal an outcome reward. |
| `market_search` | `query` | Up to five public Polymarket candidates. Available only in `market_aware`. |
| `market_snapshot` | `market_id` | Public market identity, rules, reported outcome prices, timestamps and selected quote fields. Numeric IDs only; `market_aware` only. |
| `submit` | `probabilities` | The single immutable final forecast. Valid probabilities wait for outcome settlement. |

Every research action, including notes and drafts, consumes the shared call budget; submission is separate. Preflight and post-request deadline checks prevent late source responses from entering observations. Actions and observations retain timestamps and hashes. Agent-generated actions have text-level loss mask 1, while tool observations and host-generated terminal actions have mask 0.

## Evidence and revision example

```python
result = env.step({"action": "search", "query": "relevant official bulletin"})
source = result["observation"]["results"][0]
env.step({"action": "notebook", "claim": "A provisional interpretation of the bulletin.",
          "source_hashes": [source["sha256"]]})
env.step({"action": "calculator", "expression": "(7 + 1) / (12 + 2)"})
env.step({"action": "draft", "probabilities": {"yes": 0.6, "no": 0.4},
          "rationale": "Updated after reviewing contrary evidence.",
          "source_hashes": [source["sha256"]]})
env.step({"action": "submit", "probabilities": {"yes": 0.6, "no": 0.4}})
```

The example assumes an active binary episode and a successful search. A source hash proves which snapshot was referenced, not whether it supports the claim. Notes remain marked `support_verified=false`. Notes and drafts are local to the episode and reset between questions; they do not provide an unreviewed cross-split memory channel. Drafts do not grant extra final submissions or extra outcomes.

## Public prediction-market connector

The public connector uses documented [search](https://docs.polymarket.com/api-reference/search/search-markets-events-and-profiles) and [market-by-ID](https://docs.polymarket.com/api-reference/markets/get-market-by-id) GET endpoints. The host is fixed to `gamma-api.polymarket.com`; paths, query names, query limits and numeric IDs are allowlisted. TLS hostname verification remains enabled and API redirects are rejected. Because the destination is a fixed trusted service, it follows system DNS routing, including an operator's network proxy. The generic model-selected `open` URL still requires public-address DNS validation.

This connector has no trading, wallet, position-management, private-account, or order-placement capability. It does not query the private baseline store. A market may have different wording, settlement rules, option coverage, or timing from the benchmark event; returned candidates explicitly have `match_verified=false`. Public reported prices are preserved without automatic normalization and are not a guaranteed executable quote or fair probability. Market update time and host observation time are recorded separately.

The private baseline scorer continues to require its own reviewed exact mapping. Permitting the agent to inspect public markets is a declared information condition, independent of which private baseline is used for reward. `no_consensus` rejects market tools before network access and retains the existing heuristic web-content filter.

## Next capability candidates

Structured authoritative data queries, PDF/table reading, bounded statistical analysis, independent analyst subagents, and time-series market history may be useful. They are not implemented by the current tool list. Add each through a versioned schema, bounded read-only provider, explicit information policy, and auditable observations. A broad unrestricted shell or arbitrary personal connector is not required for the current forecast task.

For RL, a stored trace is an audit of one interaction, not a replayable historical web. Counterfactual post-settlement research would need an independently frozen evidence environment; an unknown query must not silently fall through to today's web. See [RL_RESEARCH.md](RL_RESEARCH.md).
