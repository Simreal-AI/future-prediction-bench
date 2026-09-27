"""Derive a bounded-edit Humanize task with the original 14-case reward.

This public, solved fixture uses the same pristine sdist, prompt, verifier,
and reward contract as ``examples.realworld_humanize.make_task``. Its v2 tool
manifest adds the already reviewed exact ``replace_text`` action so the
cooperative and prepared full-VM arms can run one identical action script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from examples.realworld_humanize.make_task import SOURCE_FILE, _repair, make_task
from future_prediction_bench.microvm_coding import replace_text_helper_binding


TASK_ID = "humanize-4150-naturalsize-rounding-v2"
OLD_TEXT = "    exp = int(min(log(abs_bytes, base), len(suffix)))\n"
GUARD = (
    "    # Carry a rounded mantissa into the next available unit.\n"
    "    if exp < len(suffix) and abs(float(format % (abs_bytes / (base**exp)))) >= base:\n"
    "        exp += 1\n"
)
NEW_TEXT = OLD_TEXT + GUARD


def _jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                            for row in rows), encoding="utf-8")


def make_task_v2(output, *, sdist=None):
    original = make_task(output, sdist=sdist)
    root = Path(output).resolve()
    task_path = root / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if (task.get("task_id") != "humanize-4150-naturalsize-rounding-v1"
            or task.get("reward_contract", {}).get("id")
               != "humanize4150-naturalsize-host-cases-v1"):
        raise ValueError("Humanize v1 fixture changed")
    source_bytes = (root / "seed" / SOURCE_FILE).read_bytes()
    source = source_bytes.decode("utf-8")
    if (source.count(OLD_TEXT) != 1 or len(OLD_TEXT.encode()) > 256
            or len(NEW_TEXT.encode()) > 256
            or _repair(source) != source.replace(OLD_TEXT, NEW_TEXT, 1)):
        raise ValueError("Pinned Humanize repair anchor changed")
    v1_actions = [json.loads(line) for line in
                  (root / "actions.solution.jsonl").read_text(
                      encoding="utf-8").splitlines() if line.strip()]
    if ([row.get("action") for row in v1_actions]
            != ["read_file", "write_file", "run_visible_checks", "submit"]
            or v1_actions[0]["path"] != SOURCE_FILE
            or v1_actions[1]["path"] != SOURCE_FILE
            or v1_actions[1]["content"] != _repair(source)):
        raise ValueError("Pinned Humanize v1 actions changed")
    task["task_id"] = TASK_ID
    task["event_id"] = TASK_ID
    task["metadata"]["fixture_variant"] = "full_write_and_replace_text_v2"
    task["metadata"]["replace_text_helper_binding"] = replace_text_helper_binding()
    tools = task["tool_manifest"]
    write_index = next(index for index, item in enumerate(tools)
                       if item["name"] == "write_file")
    tools.insert(write_index + 1, {
        "name": "replace_text",
        "description": "Replace one unique exact UTF-8 fragment when the file SHA-256 matches.",
    })
    task_path.write_text(json.dumps(task, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    replace_action = {
        "action": "replace_text", "path": SOURCE_FILE,
        "expected_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "old_text": OLD_TEXT, "new_text": NEW_TEXT,
    }
    _jsonl(root / "actions.solution.replace_text.jsonl",
           [v1_actions[0], replace_action, v1_actions[2], v1_actions[3]])
    _jsonl(root / "actions.solution.paired.jsonl",
           [v1_actions[0], replace_action, v1_actions[3]])
    _jsonl(root / "actions.baseline.paired.jsonl",
           [v1_actions[0], v1_actions[3]])
    return {
        **original, "task_id": TASK_ID,
        "host_case_count": 14,
        "source_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "repaired_source_sha256": hashlib.sha256(_repair(source).encode()).hexdigest(),
        "paired_repair_actions": str(root / "actions.solution.paired.jsonl"),
        "paired_baseline_actions": str(root / "actions.baseline.paired.jsonl"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sdist", help="Local SHA-pinned Humanize 4.15.0 sdist")
    args = parser.parse_args()
    print(json.dumps(make_task_v2(args.output, sdist=args.sdist), indent=2))
