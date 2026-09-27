# Future Prediction Bench

A probability forecasting benchmark and verifiable real-world agent environment for research on delayed feedback, evidence, and rollout collection. Forecasting agents research unresolved events and submit probabilities; coding agents change an isolated workspace and receive independently checked outcomes.

**Status: alpha research prototype.** The forecasting protocol, offline demos, live source adapters, and scripted environment integrations are implemented. There are no formal model forecasting results, hosted leaderboard, or complete LLM training pipeline. Text trajectory exports remain `trainer_ready=false`.

| Track | Agent task | Implemented output |
| --- | --- | --- |
| Research Benchmark | Research a future event and submit a probability for every option | Frozen forecasts, evidence traces, and outcome-based scores |
| Live RL Environment | Collect repeated train-only forecasting rollouts | Delayed rewards, policy-revision checks, and grouped text samples with advantages |
| Real-World Task RL | Inspect and change an isolated software workspace | Hidden verification, auditable task rewards, and train-only text trajectories |

The intended forecasting scope is multi-domain events targeted within seven days after the prediction deadline. Current automated coverage is **MLB baseball and USGS earthquakes**. The software track currently provides bounded coding tasks through Docker and experimental QEMU backends.

## Five-minute offline demo

Use **Python 3.10 or newer**. The core runtime has no third-party Python dependencies and uses the standard library. From a repository checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m future_prediction_bench demo --output runs/my-first-demo
python -m future_prediction_bench validate examples/questions.fixture.jsonl
python -m future_prediction_bench status --db runs/my-first-demo/bench.sqlite --mode fixture
```

Choose a new or empty output directory. The demo requires no API keys, network calls, model, or Docker. It creates **three synthetic questions, five submissions, and four exported train trajectories**: two train questions receive two scripted rollouts each, and one test question receives one benchmark submission. All five submissions first await outcomes; simulated resolution then scores them. The test submission never enters training export.

Inspect `report.json`, `bench.sqlite`, `questions.jsonl`, `pending_receipts.json`, and `training_trajectories.jsonl` in the output directory. The fixture clock and database mode keep this synthetic exercise distinct from live forecasting. Local `runs/` artifacts are ignored by Git.

## Forecasting contract

Binary and categorical questions use the same distribution format:

```json
{"action": "submit", "probabilities": {"A": 0.50, "B": 0.30, "C": 0.20}}
```

Every option must appear exactly once. Probabilities must be finite, lie in `[0, 1]`, and sum to one within `1e-6`; validation never silently normalizes. Categorical options must be mutually exclusive and collectively exhaustive. Question payloads, deadlines, resolution rules, and submissions are frozen.

The primary score is normalized Brier: `0.5 * sum((p_i - y_i)^2)`, where `y_i` marks the observed outcome. Valid forecasts remain pending until an authoritative outcome arrives. Missing, delayed, canceled, and void events follow their preregistered rules.

The live runner defaults to improvement over a private baseline sealed before the deadline and before evaluation. It skips assignments with missing or invalid baselines. A separately selected `negative_brier` reward supports raw-score experiments. Train/dev/test event isolation and condition-specific reporting support reproducible study design; formal studies still need preregistered cohorts and statistical uncertainty reporting. See the [project design](docs/PROJECT_DESIGN.md).

## Optional live sources and model runs

Collect events and resolve due questions from the original sources:

```bash
python -m future_prediction_bench collect --config configs/sources.json --db runs/live.sqlite
python -m future_prediction_bench resolve-due --config configs/sources.json --db runs/live.sqlite
```

These commands make public-source requests but need no model or search credentials. The default configuration creates **test** questions; RL collection requires a separately registered, nonoverlapping train cohort. See [source rules](docs/SOURCES.md) and the [automation guide](docs/AUTOMATION_PLAN.md) for retries and workers.

For model-driven forecasting, configure a Chat Completions-compatible endpoint, Brave Search for `self_research`, and an independently configured fixed fallback baseline. Calls can incur provider charges. The CLI reads process environment variables and **does not automatically load `.env`**:

```bash
cp .env.example .env
cp configs/baselines.example.json configs/baselines.local.json
# Fill in endpoint, model IDs, and credentials before continuing.
set -a
. ./.env
set +a
python -m future_prediction_bench forecast --db runs/live.sqlite --limit 20 \
  --baseline-config configs/baselines.local.json
```

The example has no active market mappings. Market baselines require reviewed exact matches. Analyst tools provide search, webpage reading, a calculator, evidence notes, drafts, and submission. `no_search` is an explicit ablation. The default `no_consensus` filter is heuristic; `market_aware` records permitted market exposure separately. Step/call, token, prompt-size, and wall-time limits are enforced; provider usage is recorded, but currency pricing is not. See the [runner guide](docs/MODEL_RUNNER.md), [analyst tools](docs/ANALYST_TOOLS.md), and [adapter contracts](docs/ADAPTERS.md).

## Software environments and RL preparation

With Docker running and a locally cached image containing Python 3, run a scripted offline repair:

```bash
python -m future_prediction_bench realworld-code-smoke \
  --image sha256:<your-local-image-id> --output runs/my-code-smoke
```

This command never pulls an image or calls a model. It repairs one broken Python function in a network-disabled container, freezes the submission, and checks two hidden cases against host-only expectations. It demonstrates the environment protocol. Pinned [Boltons](examples/realworld_boltons26/README.md) and [Humanize](examples/realworld_humanize/README.md) fixtures extend integration to public repositories; their scripted repairs are already public and are not held-out model evaluations.

RL preparation supports grouped rollouts, delayed outcome joins, RLOO or alternative advantages, and revision checks. It lacks tokenizer integration, behavior-policy token log probabilities, and an integrated optimizer. Upstream CPU numerical probes are separate experiments. See the [RL contract](docs/RL_TRAINING.md) and [real-world environment contract](docs/REALWORLD_ENV.md).

## Evidence, documentation, and development

The [detailed research record](docs/RESEARCH_RESULTS.md) preserves experiment descriptions, commands, and measurement links, including negative results. The [performance comparison](docs/PERFORMANCE_COMPARISON.md) separates environment timings from model or training throughput. The [source integration audit](docs/OFFICIAL_CODE_INTEGRATION_AUDIT.md) distinguishes upstream execution from independently implemented mechanisms; the [validation record](docs/VALIDATION.md) records checks and limits.

Run the core test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

Advanced Docker, microVM, and upstream-code experiments require their documented host dependencies and prepared artifacts. Start with [Contributing](CONTRIBUTING.md), the [microVM guide](docs/MICROVM_ENV.md), and [backend support](docs/MICROVM_BACKENDS.md). `pilot_config.json` is a study-planning draft; `configs/sources.json` is the executable source configuration.

## License

Project-owned code is distributed under the [MIT license](LICENSE). Third-party code, data, and source services retain their independent licenses and terms; this license does not grant rights to redistribute them. Repository: [Simreal-AI/future-prediction-bench](https://github.com/Simreal-AI/future-prediction-bench).
