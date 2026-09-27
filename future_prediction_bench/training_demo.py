"""Offline analyst/RL smoke exercise. Every forecast and outcome is scripted."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .demo import FixtureClock, FixtureProvider, fixture_questions, write_json, write_jsonl
from .store import Store
from .training import calibration_probe, collect_rollout_group, prepare_training_groups


class SmokeResearchProvider(FixtureProvider):
    def public_config(self):
        return {"adapter": "scripted_fixture_research_v1", "is_fixture": True}


class SmokeAnalystModel:
    """Exercise tool plumbing; this is not learned reasoning or real inference."""

    def __init__(self):
        self.samples = 0

    def public_config(self):
        return {"adapter": "scripted_fixture_analyst_v1", "is_fixture": True}

    def complete(self, messages, tools, **kwargs):
        step = sum(message["role"] == "assistant" for message in messages)
        if step == 0:
            self.samples += 1
        question = json.loads(messages[1]["content"])["question"]
        p = (.55, .65, .75, .85)[(self.samples - 1) % 4]
        probabilities = ({"yes": p, "no": 1 - p} if question["kind"] == "binary"
                         else {"low": (1 - p) * .4, "medium": p, "high": (1 - p) * .6})
        source_hashes = []
        for message in messages:
            if message["role"] == "tool":
                observed = json.loads(message["content"])
                source = observed.get("result")
                if isinstance(source, dict) and "sha256" in source:
                    source_hashes = [source["sha256"]]
        actions = [
            ("search", {"query": "synthetic fixture bulletin"}),
            ("open", {"url": "https://example.org/fixture/bulletin"}),
            ("calculator", {"expression": "(6 + 1) / (10 + 2)"}),
            ("notebook", {"claim": "The source is explicitly a synthetic fixture, not live evidence.", "source_hashes": source_hashes}),
            ("draft", {"probabilities": probabilities, "rationale": "Scripted distribution for an integration test.", "source_hashes": source_hashes}),
            ("submit", {"probabilities": probabilities}),
        ]
        name, arguments = actions[step]
        return {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"fixture-call-{step}", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}}]}


def run_rl_smoke(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Smoke output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    clock = FixtureClock()
    store = Store(output / "bench.sqlite", mode="fixture", clock=clock)
    try:
        questions = fixture_questions()[:2]
        reports = []
        for index, question in enumerate(questions):
            store.add_question(question)
            count = len(question["options"])
            store.seal_baseline(question["question_id"], {option["id"]: 1 / count for option in question["options"]},
                                kind="internal_model", identity="synthetic-fixture-baseline-only", metadata={"is_fixture": True})
            reports.extend(collect_rollout_group(store, question["question_id"], group_id=f"fixture-group-{index}",
                                                group_size=4, policy_revision="fixture-checkpoint-v1", model=SmokeAnalystModel(),
                                                provider=SmokeResearchProvider(), reward_mode="baseline_improvement"))
        if any(report["status"] != "pending_reward" for report in reports):
            raise RuntimeError("Synthetic analyst failed to submit all forecasts")
        before = len(store.export_training())
        if before:
            raise RuntimeError("Unresolved probability forecasts leaked into training")
        clock.value = datetime(2030, 1, 3, 9, tzinfo=timezone.utc)
        for question, outcome in zip(questions, ("yes", "medium")):
            store.resolve(question["question_id"], outcome=outcome, evidence_urls=["https://example.org/fixture/bulletin"],
                          evidence_text="Scripted synthetic smoke outcome, not evidence of forecasting ability.")
        trajectories = store.export_training()
        prepared = prepare_training_groups(trajectories, current_policy_revision="fixture-checkpoint-v1",
                                           available_at=clock().isoformat(), run_mode="fixture")
        stale = prepare_training_groups(trajectories, current_policy_revision="fixture-checkpoint-v2",
                                        available_at=clock().isoformat(), run_mode="fixture")
        probe = calibration_probe()
        write_jsonl(output / "trajectories.jsonl", trajectories)
        write_json(output / "prepared_groups.json", prepared)
        write_json(output / "calibration_probe.json", probe)
        report = {"notice": "Synthetic scripted smoke only; no network, real model inference, or parameter training.",
                  "pending_before_resolution": len(reports), "training_records_before_resolution": before,
                  "prepared": prepared["summary"], "stale_policy_check": stale["summary"],
                  "tool_calls": sum(report["research_calls"] for report in reports), "trainer_ready": False,
                  "calibration_probe": probe}
        write_json(output / "report.json", report)
        return report
    finally:
        store.close()
