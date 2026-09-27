# Public release checklist

Use this checklist for public source releases. The v0.19.1 alpha release is maintained by Simreal-AI at `https://github.com/Simreal-AI/future-prediction-bench` under the MIT license. Publishing source does not start a hosted worker, invoke paid models, or publish an event dataset.

## Decisions before publication

- [x] Confirm the GitHub owner, repository name, visibility, and release version: Simreal-AI/future-prediction-bench, public alpha, v0.19.1.
- [x] Add the MIT license file and package metadata.
- [x] Document separate third-party rights in `THIRD_PARTY_NOTICES.md`; exclude downloaded third-party sources, raw source snapshots, and live question datasets from this source release.
- [x] Add project URLs and Simreal-AI maintainer identity; use repository issues for public contact.
- [x] Set the supported contribution and issue-reporting process in `CONTRIBUTING.md`.

## Reproducibility and data review

- [x] Run the offline tests, fixture validation, demo, source-distribution build, and installed-wheel import/CLI smoke test.
- [x] Build the English materials ZIP with `python3 scripts/package_materials.py`, then run `python3 scripts/audit_materials.py <materials-zip> --check-distributions`; it verifies the nested source archive, exact public manifest, wheel and sdist module lists, checksums, relative links, privacy scan, measurement digests, and tests from an extracted checkout. This public path does not require `reports/`. To create a separate internal bundle with the reviewed Chinese stakeholder briefing, explicitly pass `--briefing reports/project-highlights-zh.md` to both commands and keep it separate from the English release assets.
- [ ] Compare curated measurement JSON with its raw operational report. Only documented path redaction and asset-provenance metadata should differ; verify asset digests against the local source archive, seed workspace, and pristine VM disk before release.
- [x] Confirm that README commands work from a clean checkout with no personal absolute paths.
- [x] Confirm that shipped documentation, prompts, fixtures, and diagnostics are English.
- [x] Inspect staged files for credentials, local configuration, raw runs/databases, scraped payloads, personal identifiers, and copied private conversations.
- [x] Clearly label synthetic fixtures and avoid publishing them as live forecasting results.
- [ ] Record exactly which live source endpoints and cohorts have been exercised, including failures and unresolved events.
- [ ] Keep baseline probabilities and held-out outcomes outside agent-visible and public question exports.
- [ ] Ensure a formal study has frozen assignments, splits, model/baseline configuration, budgets, and missing-data rules.

## Automation and operations

- [ ] Configure real source settings and contact headers where required; exercise adapters before relying on an unattended worker.
- [ ] Store operational state and snapshots in persistent private storage with backups.
- [ ] Schedule one worker per state database, define failure alerts, and document restart/recovery behavior.
- [ ] Confirm that missing data and failed requests stay pending or enter review instead of becoming fabricated outcomes.
- [ ] Keep model and provider credentials in the hosting platform's secret store.
- [ ] Enable scheduled network/model jobs only after reviewing their configuration and operating costs.

The CI workflow runs deterministic checks and package builds. It does not publish packages or repositories and does not perform live forecasting or parameter training.
