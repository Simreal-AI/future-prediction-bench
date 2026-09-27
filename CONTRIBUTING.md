# Contributing

Future Prediction Bench is an early-stage project. Documentation, source templates, model prompts, fixtures, comments, and user-facing messages should be written in English.

## Development

Use Python 3.10 or newer from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python -m future_prediction_bench demo --output runs/contributor-demo
```

Use a new or empty output directory for each demo. The core package has no required third-party runtime dependencies. Tests should use deterministic offline fixtures; do not add live network or paid model calls to ordinary CI.

## Changes to sources and scoring

A source adapter should define stable event IDs, a frozen question template, timezone-aware deadlines, exhaustive options, exact resolution rules, and deterministic handling for missing, canceled, or revised records. Include representative offline source payloads and tests for both successful and pending/error outcomes. Document source access and data-retention requirements in [docs/SOURCES.md](docs/SOURCES.md).

Protocol changes must preserve the meaning of already published questions and submitted predictions. Do not silently normalize probability outputs, overwrite conflicting outcomes, turn unavailable data into No, mix fixture and live databases, or move an event across dataset splits.

Private baselines must be recorded before the deadline and remain absent from agent inputs. Keep raw Brier scores independent of reward choice. Changes affecting baseline selection or reward availability need coverage for missing, mismatched, and late baseline data.

## Pull requests

Describe the concrete behavior changed, the reason for it, and the validation performed. State any source, model, or live behavior that was not exercised. Keep generated runs, API credentials, raw research snapshots, local settings, and private planning material out of commits.

Report reproducible problems through the [issue tracker](https://github.com/Simreal-AI/future-prediction-bench/issues). Include the command, Python version, platform, and a minimal sanitized example. Do not include credentials, private baselines, or raw operational records.

Contributions are made under the project's [MIT License](LICENSE). Before proposing a release, follow [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) and review the [third-party notices](THIRD_PARTY_NOTICES.md).
