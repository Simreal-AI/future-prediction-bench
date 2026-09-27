"""Compare bounded hidden-case parallelism on the public integration task."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from future_prediction_bench.realworld_demo import load_coding_actions, run_coding_task


def benchmark(task_root, image, output, *, repetitions=3):
    if type(repetitions) is not int or not 1 <= repetitions <= 10:
        raise ValueError("repetitions must be between 1 and 10")
    task_root, output = Path(task_root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    task = json.loads((task_root / "task.json").read_text(encoding="utf-8"))
    actions = load_coding_actions(task_root / "actions.solution.jsonl")
    runs = []
    for index in range(repetitions * 2):
        workers = (1, 4)[(index + index // 2) % 2]
        report = run_coding_task(
            task=task, seed_dir=task_root / "seed", verifier_dir=task_root / "verifier",
            image=image, actions=actions, output=output / f"episode-{index + 1}",
            registry_path=output / "registry.sqlite", verifier_workers=workers,
            visible_check=("python3", "-B", "-c", "from boltons.strutils import singularize"),
        )
        if report["status"] != "graded" or report["reward"] != 1.0:
            raise ValueError("Cannot compare verifier speed because the task did not pass")
        adapter_metrics = json.loads((Path(report["output"]) / "report.json").read_text(encoding="utf-8"))["adapter_metrics"]
        runs.append({"workers": workers, "verifier_seconds": adapter_metrics["verify_seconds"],
                     "total_seconds": report["elapsed_seconds"], "reward": report["reward"]})
    medians = {
        str(workers): {field: statistics.median(row[field] for row in runs if row["workers"] == workers)
                       for field in ("verifier_seconds", "total_seconds")}
        for workers in (1, 4)
    }
    result = {"scope": "One public real-repository integration task on this host; no policy inference or optimizer.",
              "repetitions_per_condition": repetitions, "runs": runs, "median_seconds": medians,
              "training_speed_claim": False}
    (output / "benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    arguments = parser.parse_args()
    print(json.dumps(benchmark(arguments.task_root, arguments.image, arguments.output,
                               repetitions=arguments.repetitions), indent=2))
