# Autonomous model research runner

The optional live runner gives a model the public question and a versioned analyst toolkit: web search, webpage reading, bounded arithmetic, source-linked notes, probability drafts, and final submission. The market-aware condition also exposes public market discovery and snapshots. The model chooses what to research and returns a probability for every option. Binary and categorical questions use the same protocol. The host validates deadlines, shared research budgets, full option coverage, and probability sums through `PredictionEnv` and `Store`. See [ANALYST_TOOLS.md](ANALYST_TOOLS.md) for exact capabilities.

The implementation uses the Python standard library. Running the test suite does not call a model or a search API. Configuring credentials enables real network requests that may incur provider charges.

## Configure providers

Set these environment variables in the scheduler's environment or a local shell. Do not commit their values. The Python library reads the process environment; it does not automatically load a `.env` file.

| Variable | Meaning |
| --- | --- |
| `FPB_MODEL_BASE_URL` | Trusted operator-configured API base URL, including its version path, such as `https://your-model-service.example/v1`. The adapter appends `/chat/completions`. |
| `FPB_MODEL_API_KEY` | Model API bearer token. A local server without authentication can use a dummy value. |
| `FPB_MODEL_NAME` | Exact deployed model/checkpoint name accepted by the configured service. |
| `FPB_BRAVE_API_KEY` | Brave Web Search subscription token. Required for `self_research`; unnecessary for `no_search`. |

The model endpoint must support nonstreaming Chat Completions requests with `messages`, `tools`, `tool_choice="auto"`, `parallel_tool_calls=false`, `temperature`, and `max_tokens`. Responses must contain one assistant choice with text or one function tool call. Providers with different protocols or reasoning/tool transcript requirements need a separate adapter. A local tool-capable [vLLM server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server/) is one implementation of this protocol; its [tool calling configuration](https://docs.vllm.ai/en/latest/features/tool_calling/) depends on the served model.

The configured model endpoint requires HTTPS, except for loopback development servers. Endpoint configuration is trusted operator input and is never controllable by model tools. Credentials are sent only in headers, are excluded from public configuration hashes and transcripts, and are never forwarded on redirects. Model API redirects are rejected.

Search uses the [Brave Web Search API](https://api-dashboard.search.brave.com/app/documentation/web-search), with five results and optional extra snippets per query by default. Search ages such as “two hours ago” are not promoted to verified publication timestamps. Source snapshots retain the host's actual observation time. The existing no-consensus filter still applies before snippets or pages reach the model.

## Python API

```python
from future_prediction_bench.providers import BraveResearchProvider, ChatCompletionModel
from future_prediction_bench.runner import run_questions
from future_prediction_bench.store import Store

store = Store("data/live.sqlite", mode="live")
try:
    reports = run_questions(
        store,
        ["an-already-published-question-id"],
        model=ChatCompletionModel.from_env(),
        provider=BraveResearchProvider.from_env(),
        track="benchmark",
        market_mode="no_consensus",
        research_mode="self_research",
        reward_mode="baseline_improvement",
        max_steps=12,
        max_calls=8,
        max_tokens_per_step=1024,
        max_output_tokens=8192,
        max_wall_seconds=300,
    )
finally:
    store.close()
```

Question IDs must already exist in the store and remain within their forecast windows. The default baseline-improvement reward requires the trusted host to seal a private baseline before assigning an episode. The runner never generates a substitute baseline from the evaluated forecast or exposes private baseline fields to the agent. Use `reward_mode="negative_brier"` explicitly for an experiment without baseline-relative reward.

For the no-search ablation, set `research_mode="no_search"` and omit `provider`. Only `submit` is advertised to the model. For delayed-feedback RL trajectories, set `track="rl"` and supply training-split questions. Evaluation questions cannot enter RL episodes.

## Budgets, reproducibility, and audit records

Each question has independent limits:

| Limit | Default | Enforcement |
| --- | --- | --- |
| Model requests | 12 | No further model request after the step limit. |
| Analyst tool calls | 8 combined | Search, open, calculator, notebook, draft, and market tools share one budget. `ResearchSession` records rejections. Final submit is separate. |
| Requested output tokens per model call | 1,024 | Sent as the endpoint's `max_tokens`. |
| Sum of requested output-token allowances | 8,192 | The full requested allowance is reserved before each call, even if the server omits usage. |
| Wall time | 300 seconds | Live model and research requests receive the remaining time; the built-in transport closes sockets at its deadline. |
| Serialized prompt characters per request | 160,000 | Stop before issuing an oversized model request. No invisible transcript truncation. |

The output reservation is intentionally conservative: eight calls of 1,024 tokens exhaust an 8,192-token allowance even if the model emits fewer tokens. API-reported prompt, completion, and total token usage is recorded separately, including how many calls supplied usage. These controls bound requests and requested generation allowances; they do not calculate currency cost or guarantee provider billing/token semantics. Input text is sent again on later turns, so cumulative input billing can exceed the size of a single prompt.

Every assignment uses a SHA-256 configuration identity covering the adapter and endpoint, checkpoint, generation settings, search/reader settings, system-prompt version and digest, all budgets, track, research mode, market mode, and reward mode. Changing these settings creates a different comparison group. A provider reusing the same mutable model alias is still an external reproducibility limitation; use versioned checkpoints where possible.

The store records each model-visible request and assistant response, selected usage fields, and the configuration hash. Tool evidence and actions remain in the environment's existing event log. Private question metadata, hidden baselines, resolved labels, credentials, and server-private reasoning fields are not added to model messages. These are local audit records, not cryptographic evidence of a provider's actual inference or a tokenized training batch.

## Failure handling

- Valid submissions become `pending_reward` and wait for the external result. Prediction success is separate from eventual scoring.
- Invalid probability distributions or malformed final JSON become invalid submissions. Values are never silently normalized or completed.
- Unknown tools and malformed tool arguments receive a bounded error observation; arbitrary execution is unavailable.
- A model that exhausts its step, output-allowance, or prompt-size budget without a submission receives an invalid submission. The host does not invent a forecast for it.
- Host-generated terminal actions are marked `origin="host", loss_mask=0`; only actual assistant generations are eligible for policy loss.
- Model transport failures or wall-time exhaustion return an error report. The assigned episode remains active until its forecast deadline and then becomes missed. No prediction or immediate reward is fabricated.
- The same benchmark question/configuration is assigned once. Repeating a scheduled command skips an existing assignment, including interrupted or failed attempts. Automatic continuation of a partially executed model transcript is not implemented. Repeated RL rollouts are permitted only on the training split.
- A model response or page arriving after the forecast deadline cannot be submitted as a valid forecast. Late model responses can be retained in the audit table without becoming accepted evidence or predictions.

A provider error includes only a fixed error category or HTTP status. API response bodies, credential-bearing headers, and exception diagnostics are not surfaced to the model or task logs. Missing credentials are configuration errors, not simulated forecasts.

## Public webpage reader

The `open` tool accepts only public HTTP(S) pages on ports 80 or 443. Every DNS answer must be globally routable. The connection uses the validated IP directly while retaining the original hostname for TLS verification; it does not resolve the hostname again during connection. Loopback, private, link-local, multicast, and IPv6 translation/tunnel destinations are blocked. Every redirect is validated independently, with at most four redirects.

Pages are limited to 1 MB and 16,000 extracted characters by default. The reader accepts HTML, XHTML, and plain text; it excludes scripts, styles, noscript, and template content and executes no page code. It has no cookies, authentication, browser state, proxy support, or local-file access. DNS uses bounded daemon workers with a timeout; stalled lookups cannot create an unbounded number of threads. Socket closure enforces an absolute request time limit in addition to socket timeouts. Compressed responses are rejected rather than decompressed without bounds.

This is a deliberately limited text reader. JavaScript-only pages, PDFs, authentication walls, and sites requiring a browser can fail or yield little text. The no-consensus filter is heuristic: public pages can still contain unattributed consensus or incorrect information. Live retrieval does not establish a historical information cutoff or prove absence of leakage.
