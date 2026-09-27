"""Synthetic clock and data only: exercise both tracks without network or model costs."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .env import PredictionEnv
from .store import Store


class FixtureClock:
    def __init__(self):
        self.value = datetime(2030, 1, 1, 12, tzinfo=timezone.utc)

    def __call__(self):
        return self.value


class FixtureProvider:
    """Handwritten observations, explicitly not a live search engine."""

    def search(self, query):
        return [{"url": "https://example.org/fixture/bulletin", "title": "Synthetic bulletin (fixture only)",
                 "text": "A handwritten tool observation for testing snapshot persistence and delayed scoring. This is not real forecasting evidence.",
                 "published_at": "2030-01-01T09:00:00+00:00"}]

    def open(self, url):
        return {"url": url, "title": "Synthetic source (fixture only)",
                "text": "Static fixture page text. This example uses scripted probabilities, not a model adapter.",
                "published_at": "2030-01-01T09:00:00+00:00"}


def fixture_questions():
    common = {"schema_version": "0.1", "is_fixture": True,
              "issued_at": "2030-01-01T00:00:00Z", "forecast_deadline": "2030-01-01T23:00:00Z",
              "outcome_not_before": "2030-01-02T00:00:00Z", "resolve_after": "2030-01-03T08:00:00Z"}
    specifications = [
        ("fixture-weather-binary", "train", "weather", "binary",
         "[SYNTHETIC FIXTURE] Will the maximum temperature in the fictional city of Clearford reach 30 degrees Celsius on 2030-01-02?",
         [("yes", "At least 30 degrees Celsius"), ("no", "Below 30 degrees Celsius")],
         "Use the daily maximum temperature in the synthetic bulletin: yes if at least 30 degrees Celsius, otherwise no. Void if no valid observation is available."),
        ("fixture-port-categorical", "train", "logistics", "categorical",
         "[SYNTHETIC FIXTURE] How many cargo ships will arrive at the fictional port of Cape Harbor on 2030-01-02?",
         [("low", "0-4 ships"), ("medium", "5-9 ships"), ("high", "10 or more ships")],
         "Use the integer count of cargo-ship arrivals in the synthetic daily port bulletin. Void if the bulletin is unavailable."),
        ("fixture-energy-categorical", "test", "energy", "categorical",
         "[SYNTHETIC FIXTURE] What will be the service status of the fictional Greenleaf electricity grid on 2030-01-02?",
         [("normal", "Normal service throughout the day"), ("partial", "Rationing without a grid-wide outage"), ("outage", "A grid-wide outage occurs")],
         "Use the synthetic grid bulletin: outage takes precedence if a grid-wide outage occurs; otherwise partial if rationing occurs; otherwise normal. Void if the bulletin is unavailable."),
    ]
    return [{**common, "question_id": qid, "event_id": qid + "-event", "cluster_id": qid + "-cluster",
             "split": split, "kind": kind, "domain": domain, "prompt": prompt,
             "options": [{"id": key, "text": label} for key, label in options],
             "resolution": {"criteria": criteria, "source_urls": ["https://example.org/fixture/bulletin"], "on_ambiguous": "void"}}
            for qid, split, domain, kind, prompt, options, criteria in specifications]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    with Path(path).open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def run_demo(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Demo output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    clock = FixtureClock()
    store = Store(output / "bench.sqlite", mode="fixture", clock=clock)
    try:
        questions = fixture_questions()
        for question in questions:
            store.add_question(question)
        write_jsonl(output / "questions.jsonl", questions)
        runs = [
            (questions[0], "rl", {"yes": 0.7, "no": 0.3}),
            (questions[0], "rl", {"yes": 0.6, "no": 0.4}),
            (questions[1], "rl", {"low": 0.2, "medium": 0.5, "high": 0.3}),
            (questions[1], "rl", {"low": 0.25, "medium": 0.45, "high": 0.3}),
            (questions[2], "benchmark", {"normal": 0.65, "partial": 0.25, "outage": 0.1}),
        ]
        submissions = []
        for question, track, probabilities in runs:
            env = PredictionEnv(store, FixtureProvider())
            env.reset(question["question_id"], "fixture-script-v1", track=track)
            env.step({"action": "search", "query": question["prompt"]})
            env.step({"action": "open", "url": "https://example.org/fixture/bulletin"})
            receipt = env.step({"action": "submit", "probabilities": probabilities})
            assert receipt["reward"] is None and receipt["info"]["awaiting_outcome"]
            submissions.append(receipt)
        write_json(output / "pending_receipts.json", submissions)
        assert store.export_training() == []
        clock.value = datetime(2030, 1, 3, 9, tzinfo=timezone.utc)
        for question, outcome in zip(questions, ("yes", "medium", "normal")):
            store.resolve(question["question_id"], outcome=outcome,
                          evidence_urls=["https://example.org/fixture/bulletin"],
                          evidence_text="A manually specified synthetic fixture outcome; it must not be reported as evidence of real forecasting ability.")
        records = store.export_training()
        assert len(records) == 4 and all(record["question"]["split"] == "train" for record in records)
        write_jsonl(output / "training_trajectories.jsonl", records)
        report = {"notice": "Synthetic fixture only. No live research, real model inference, or RL optimizer was run.",
                  "pending_before_resolution": len(submissions), "exported_train_trajectories": len(records),
                  "summary": store.summary()}
        write_json(output / "report.json", report)
        return report
    finally:
        store.close()
