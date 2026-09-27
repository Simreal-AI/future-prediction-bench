# Third-party sources and assets

The project's original code and documentation are licensed under the [MIT
License](LICENSE). Third-party projects, event data, and downloaded assets retain
their own licenses and terms. The project license does not relicense them.

The public source archive excludes downloaded upstream checkouts, source
archives, VM images, provider credentials, operational databases, and raw runs.
Examples fetch or accept separately supplied upstream assets and verify their
recorded revisions and digests. Keep the applicable upstream notices when
building or redistributing an environment that includes those assets.

## Source attribution

- Forecasting adapters use MLB and USGS interfaces. Their templates, collection
  rules, and source references are in [SOURCES.md](docs/SOURCES.md). This release
  includes synthetic question fixtures rather than a redistributed live event
  dataset.
- Repository fixtures use separately obtained Boltons and Humanize releases.
  Their source pins and attribution are in the [Boltons example](examples/realworld_boltons26/README.md)
  and [Humanize example](examples/realworld_humanize/README.md).
- Optional upstream integrations reference SWE-MiniSandbox, ROLL, Crab,
  Tree-GRPO, VIP, KLPO, AReaL, Waypoint, Pilot-Commit, and CubeCoW. Pinned
  revisions, integration boundaries, and upstream links are documented in the
  [official code audit](docs/OFFICIAL_CODE_INTEGRATION_AUDIT.md), individual
  source reviews, and example READMEs.

Curated measurement reports describe bounded experiments and preserve their
source and asset identities. They are distinct from bundled upstream software
or model evaluation datasets. Review the relevant source terms before making a
separate data or environment-image release.
