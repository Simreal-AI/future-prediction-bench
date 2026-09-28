# Changelog

## 0.19.2

Compatibility fix for the initial public alpha release.

- Make the deeply nested JSON rejection test portable across Python builds:
  valid JSON may reach schema rejection before a decoder recursion limit.
  Both paths must reject the input and keep training readiness false.
- Add a deterministic check that decoder recursion errors still produce a
  machine-readable CLI error.
- Include the fix in the versioned source and materials archives and refresh
  the published-version and validation documentation.

Runtime input validation and training-readiness behavior are unchanged.

## 0.19.1

Public alpha release of the forecasting protocol and real-world agent
environment prototype.

- Add the MIT license, third-party source notices, maintainer identity, and
  repository and issue links.
- Make public source and materials packaging independent of the optional
  Chinese stakeholder briefing, including clean-checkout regression coverage.
- Put the offline demo and current forecasting coverage near the top of a
  concise English README; preserve detailed commands and experiment records in
  the [research results](docs/RESEARCH_RESULTS.md).

This release contains infrastructure tests and scripted integration
measurements. It does not introduce formal forecasting model results, a GPU
policy trainer, or a hosted leaderboard.

## 0.19.0

Local research candidate with forecasting, rollout preparation, Docker and
microVM task environments, upstream component probes, and curated measurement
evidence. The prior full overview is retained in the research results document.
