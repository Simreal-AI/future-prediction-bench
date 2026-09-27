# Automated sources

The first live adapters cover sports and geophysics. They create English questions
from deterministic, preregistered templates; they do not ask an LLM to invent
options or decide the label. Binary and categorical questions both require a
probability for every option.

The source configuration is [`configs/sources.json`](../configs/sources.json).
Both sources are enabled by default and publish into the `test` split. Use a
separate, non-overlapping cohort for training. A source's event and cluster IDs
remain stable across collection time and split settings. The pipeline must never
overwrite a published question when the upstream source changes.

## Sources and templates

| Adapter | Discovery | Question | Settlement |
| --- | --- | --- | --- |
| `mlb` | MLB's schedule API, at most seven days ahead | Does the scheduled home team win? Options: `yes`, `no` | MLB's official Final status and unequal final scores, including extra innings |
| `usgs` | USGS capability check plus future UTC calendar days | Worldwide daily earthquake count with preferred magnitude at least 5.0. Options: `zero`, `one`, `two_or_more` | USGS aggregate count, first successful response after the day ends plus a 24-hour settling delay |

Two adapters provide an initial multi-domain pilot, not broad coverage of the
world. MLB is seasonal. USGS daily bins can be imbalanced; measure the observed
distribution before changing thresholds or adding templates. Threshold changes
must create new question IDs, preserve the day-level cluster, and be applied only
to future cohorts.

## Timing and exception rules

MLB deadlines fall one hour before the scheduled start, and discovery requires at
least 30 minutes of remaining forecast time. Games with unknown start times,
conditional schedules, or resumed-game metadata are excluded. The target is start
plus six hours; settlement begins at start plus eight hours. A postponed or
cancelled game, a start moved outside the original eight-hour window, or a start
moved to/before the locked forecast deadline is void. A live game can wait until
start plus 24 hours. Finals first fetched after that window, changed participants,
resumed games, or malformed/tied final scores require review. The adapter does not
infer who lost from missing or unfinished results.

USGS deadlines fall one hour before the UTC day starts, before any part of the
target day is observable. The default generates the next five calendar days, with
24 hours of settlement delay. All earthquakes matching `eventtype=earthquake`,
the global region, and the minimum-magnitude filter are counted, including
automatic and reviewed catalog records (the default when `reviewstatus` is omitted;
the API rejects an explicit value of `all`). This is explicitly a
prediction of a catalog snapshot, not a claim to count every physical earthquake.
Later catalog revisions do not change a finalized label.

The USGS API's `endtime` is inclusive. The adapter queries through 23:59:59.999 UTC
to represent a half-open day and avoid counting an event at the following midnight
twice. It calls the **aggregate count endpoint**, not the length of a limited event
list. Thus a 20,000-record query cap cannot silently truncate the count. A valid
`{"count": 0, "maxAllowed": 20000}` means zero; absent JSON, a timeout, or an
unexpected schema does not. `maxAllowed` describes the catalog query limit and is
not an upper bound on the aggregate count.

The pipeline owns retries, idempotency, raw response snapshots, review, and reward
delivery. Every adapter proposal includes evidence URLs, evidence text, and the
HTTP snapshot metadata. Network exceptions propagate for retry; unexpected result
structures either fail collection or require review. Neither is a resolved
negative outcome.

## First-party interfaces and live verification

The [USGS FDSN API documentation](https://earthquake.usgs.gov/fdsnws/event/1/)
defines `application.json`, the aggregate `count` method, ISO timestamps, inclusive
time bounds, event type, review status, magnitude filters, and the query limit.

The [MLB schedule endpoint](https://statsapi.mlb.com/api/v1/schedule?sportId=1)
returns game IDs, schedules, participants, statuses, and scores. This is an
MLB-hosted interface, but this project does not claim a supported public API
contract or an upstream availability guarantee. Recheck its schema when adding
new templates or updating the adapter. Fetching data does not grant permission to
republish third-party data; review source terms before distributing a dataset.

Read-only live checks on **2026-09-22** confirmed:

- MLB's [2026-09-22 through 2026-09-29 schedule](https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate=2026-09-22&endDate=2026-09-29)
  returned JSON with `totalGames: 92` and scheduled games with `gamePk`, `gameDate`,
  team IDs/names, and the expected status fields.
- The [USGS daily M5+ count for 2026-09-20](https://earthquake.usgs.gov/fdsnws/event/1/count?format=geojson&starttime=2026-09-20&endtime=2026-09-21&minmagnitude=5&eventtype=earthquake)
  returned `{"count":10,"maxAllowed":20000}`. This historical request checked the
  API response shape; it is not a benchmark prediction or score. Its inclusive
  midnight bound differs from the production template's half-open UTC day.
- [USGS `application.json`](https://earthquake.usgs.gov/fdsnws/event/1/application.json)
  returned the `eventtypes` array containing `earthquake`.
- A separate check of the exact production count parameters (half-open UTC day
  represented by `endtime=2026-09-20T23:59:59.999000Z`, `minmagnitude=5.0`, and
  `eventtype=earthquake`) returned count 10. Sending `reviewstatus=all` initially
  produced HTTP 400; omitting it, as the adapter now does, uses the all-status
  default successfully.

The checks used HTTPS with certificate verification enabled. Their quoted results
are observations at verification time and can change upstream. Future benchmark
labels must use their own archived snapshots, not these documentation examples.

## Adding another source

Implement `SourceAdapter.discover(now, config, client)` and
`SourceAdapter.resolve(question, now, client)`, then register its class in
`SOURCE_REGISTRY`. `build_sources(config)` returns `(adapter, source_config)` pairs
for enabled entries. Use the injected client's `get_json(url)` and archive its
`last_snapshot`; do not perform hidden network calls.

Question metadata must include `source_id`, `source_event_id`, `template_version`,
`target_at`, `discovery_provenance`, and immutable `resolution_parameters`. Check
future chronology before returning a question. Use stable IDs and a cluster broad
enough to keep correlated variants out of different splits. Result proposals use
`pending`, `resolved`, `void`, or `needs_review`, with an option ID only for a
resolved result. Require an explicit, testable rule for cancellations, unavailable
data, revisions, inclusive boundaries, and any pagination.
