"""Build the versioned Boltons fixture with full-write and bounded text-edit actions.

Both scripted repairs produce identical final bytes from the same pinned sdist.
This is a public, solved integration fixture, not an unseen SWE evaluation task.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from future_prediction_bench import prepared_microvm as prepared_pins
from future_prediction_bench.microvm_coding import replace_text_helper_binding

try:
    from .make_task import make_task
except ImportError:  # Direct `python3 examples/.../make_task_v2.py` invocation.
    from make_task import make_task


TASK_ID = "boltons-26-singularize-ss-v2"
PATH = "boltons/strutils.py"
OLD_TEXT = "    else:\n        singular = word[:-1]\n    return _match_case(orig_word, singular)"
NEW_TEXT = "    elif word.endswith('ss'):\n        singular = word\n" + OLD_TEXT


def _jsonl(path, actions):
    path.write_text("".join(json.dumps(action, ensure_ascii=False) + "\n"
                            for action in actions), encoding="utf-8")


def make_task_v2(output, *, sdist=None):
    """Keep v1 untouched while deriving a distinct, pinned v2 task contract."""
    helper = replace_text_helper_binding()
    prepared_pins._check_v2_replace_helper({
        "task_id": TASK_ID, "metadata": {"replace_text_helper_binding": helper}})
    original = make_task(output, sdist=sdist)
    root = Path(output).resolve()
    task_path = root / "task.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    source_bytes = (root / "seed" / PATH).read_bytes()
    source = source_bytes.decode("utf-8")
    v1_actions = [json.loads(line) for line in
                  (root / "actions.solution.jsonl").read_text(encoding="utf-8").splitlines()
                  if line.strip()]
    if (len(v1_actions) != 4
            or [item["action"] for item in v1_actions]
               != ["read_file", "write_file", "run_visible_checks", "submit"]
            or v1_actions[1]["path"] != PATH
            or source.count(OLD_TEXT) != 1
            or v1_actions[1]["content"] != source.replace(OLD_TEXT, NEW_TEXT)):
        raise ValueError("Pinned v1 source and v2 edit differ")
    task["task_id"] = TASK_ID
    task["event_id"] = TASK_ID
    task["reward_contract"]["id"] = "boltons26-singularize-host-cases-v2"
    task["metadata"]["fixture_variant"] = "full_write_and_replace_text_v2"
    task["metadata"]["replace_text_helper_binding"] = helper
    tools = task["tool_manifest"]
    write_index = next(index for index, item in enumerate(tools)
                       if item["name"] == "write_file")
    tools.insert(write_index + 1, {
        "name": "replace_text",
        "description": "Replace one unique exact UTF-8 fragment when the file SHA-256 matches.",
    })
    task_path.write_text(json.dumps(task, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    replace_actions = [v1_actions[0], {
        "action": "replace_text", "path": PATH,
        "expected_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "old_text": OLD_TEXT, "new_text": NEW_TEXT,
    }, *v1_actions[2:]]
    _jsonl(root / "actions.solution.full_write.jsonl", v1_actions)
    _jsonl(root / "actions.solution.replace_text.jsonl", replace_actions)
    return {
        **original, "task_id": TASK_ID,
        "full_write_actions": str(root / "actions.solution.full_write.jsonl"),
        "replace_text_actions": str(root / "actions.solution.replace_text.jsonl"),
        "repaired_source_sha256": hashlib.sha256(
            v1_actions[1]["content"].encode("utf-8")).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sdist", help="Optional already downloaded pinned source distribution")
    arguments = parser.parse_args()
    print(json.dumps(make_task_v2(arguments.output, sdist=arguments.sdist), indent=2))
