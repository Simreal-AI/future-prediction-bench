"""Fault-inject and stress the pinned cooperative checkpoint in real QEMU.

The first branch per repository is a trusted diagnostic, not a reward-bearing
task episode. It tests a detached candidate grandchild, out-of-workspace
symlink write, per-case filesystem disposal, and a stdout-close timeout. It
is followed by alternating, normally graded RealWorldEnv branches from the
same frozen template. No result is a full-VM security equivalence claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from examples.cooperative_realworld.benchmark import _actions as boltons_actions
from examples.cooperative_realworld.benchmark import _episode
from examples.cooperative_realworld.host_runtime import (
    CooperativePrivateCases, connect_booted_vm, make_env as boltons_env,
    verify_assets,
)
from examples.cooperative_realworld_humanize.benchmark import _actions as humanize_actions
from examples.cooperative_realworld_humanize.host_runtime import (
    HumanizePrivateCases, make_env as humanize_env,
)
from examples.realworld_boltons26.check_microvm_branch_env import _fixture_task
from examples.realworld_boltons26.microvm_benchmark import _boot, _runtime
from examples.resident_guest_candidate.smoke_qemu import _outer_workspace_sha
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2
from wire import MAX_BATCH_REQUEST, VERSION, frame, validate_request


PINNED = {
    "boltons": ("boltons-26-singularize-ss-v2",
                "98877b015eb530e022e15480d5c2f788a6f4a94587b92e318c3d66b4a2640545",
                7),
    "humanize": ("humanize-4150-naturalsize-rounding-v2",
                 "7d5fe2be77059694b546988a5cbebda0d1fe9dc00ffa00b5fb038f72843bbeba",
                 5),
}

GRANDCHILD = (
    "import os,time,ctypes\n"
    "r,w=os.pipe()\n"
    "p=os.fork()\n"
    "if p:\n"
    " os.close(w); assert os.read(r,1)==b'1'; print('spawned')\n"
    "else:\n"
    " os.close(r); os.setsid(); os.close(1); os.close(2)\n"
    " assert ctypes.CDLL(None).prctl(15,b'fpbcasechild')==0\n"
    " os.write(w,b'1'); os.close(w)\n"
    " while True: time.sleep(1)\n"
)
SYMLINK_ESCAPE = (
    "import os\n"
    "os.symlink('/fpb-stress-outside','/workspace/outside-link')\n"
    "try: open('/workspace/outside-link','w').write('escaped')\n"
    "except OSError: print('denied')\n"
    "else: print('escaped')\n"
)
FOLLOW_ON = "import os; print(os.path.lexists('/workspace/outside-link'))"
STDOUT_CLOSE_TIMEOUT = "import os,time; os.close(1); time.sleep(30)"
BENIGN = "print('ok')"
FAULT_CODES = [GRANDCHILD, SYMLINK_ESCAPE, FOLLOW_ON, STDOUT_CLOSE_TIMEOUT] + [BENIGN] * 10


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha(path):
    return _sha(Path(path).read_bytes())


def _validate_fault_codes():
    for code in FAULT_CODES:
        compile(code, "<trusted-fault-probe>", "exec")
        if not 0 < len(code.encode("utf-8")) <= 512:
            raise ValueError("fault_case_outside_guest_bound")
    request = {"v": VERSION, "seq": 0, "op": "case_batch",
               "args": {"codes": FAULT_CODES}}
    validate_request(request)
    return len(frame(request, limit=MAX_BATCH_REQUEST))


def _required(runtime, command, *, timeout=20):
    answer = runtime.run_shell(command, timeout=timeout)
    if answer.get("return_code") != 0:
        raise RuntimeError("guest_stress_attestation_failed")
    return answer.get("stdout", "")


def _census(runtime):
    _required(runtime, "test -r /proc/1/comm && test -d /mnt/root")
    descendants = _required(runtime,
        "grep -l '^fpbcasechild$' /proc/[0-9]*/comm 2>/dev/null || true")
    debris = _required(runtime,
        "ls -d /mnt/root/.fpb-resident-* /mnt/root/.fpb-case-* 2>/dev/null || true")
    if descendants.strip() or debris.strip():
        raise RuntimeError("candidate_descendant_or_branch_pool_survived")
    return {"named_candidate_descendants": 0, "root_side_branch_pool_debris": 0}


def _helper_digest(runtime, expected):
    line = _required(runtime, "sha256sum /mnt/root/fpb_coop_guest.py")
    match = re.fullmatch(r"([0-9a-f]{64})\s+\S+\n?", line)
    if match is None or match.group(1) != expected:
        raise RuntimeError("cooperative_service_helper_changed")


def _fault_branch(client, private, runtime, *, label):
    _validate_fault_codes()
    _required(runtime, "test ! -e /mnt/root/fpb-stress-outside")
    started = client.reset("stress-" + label, "baseline")
    original = client.action("read_file", {"path": private.source_path})
    if original.get("accepted") is not True or original.get("sha256") != private.seed_file_sha256:
        raise RuntimeError("stress_branch_seed_unreadable")
    client.submit()
    results = client.run_case_batch(FAULT_CODES,
                                    expected_branch_sha256=private.seed_file_sha256)
    expected = [(0, b"spawned\n"), (0, b"denied\n"), (0, b"False\n"),
                (124, b"")] + [(0, b"ok\n")] * 10
    if (len(results) != 14 or any(
            result["return_code"] != code or result["stdout_bytes"] != output
            or result["truncated"] or result["output_over_batch_cap"]
            for result, (code, output) in zip(results, expected, strict=True))):
        raise RuntimeError("stress_candidate_fault_result_mismatch")
    client.close(completed=True)
    if client.state != "idle":
        raise RuntimeError("stress_branch_did_not_close")
    _required(runtime, "test ! -e /mnt/root/fpb-stress-outside")
    _helper_digest(runtime, private.guest_service_sha256)
    census = _census(runtime)
    return {"fault_case_count": len(results), "detached_grandchild_handshake": True,
            "symlink_outside_write_denied": True,
            "next_case_workspace_copy_clean": True,
            "stdout_close_timeout_return_code": results[3]["return_code"],
            "cleanup_after_faults": census,
            "branch_process_nonce": started["namespaces"]["process_nonce"],
            "case_result_digest": _sha(json.dumps([
                [row["return_code"], _sha(row["stdout_bytes"])] for row in results],
                separators=(",", ":")).encode("ascii"))}


def _run_repository(label, task_dir, assets_dir, output, *, branches):
    task_dir, assets_dir = Path(task_dir).resolve(), Path(assets_dir).resolve()
    task_id, disk_sha, baseline_cases = PINNED[label]
    manifest = json.loads((assets_dir / "manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("task_id") != task_id
            or manifest.get("rootfs_qcow2_sha256") != disk_sha
            or _file_sha(assets_dir / "rootfs.qcow2") != disk_sha):
        raise ValueError("stress_fixture_image_pin_mismatch")
    source_task = _fixture_task(json.loads((task_dir / "task.json").read_text()))
    time_window = {key: source_task[key] for key in
                   ("issued_at", "action_deadline", "outcome_not_before", "verify_after")}
    if label == "boltons":
        private = CooperativePrivateCases(
            task_dir / "task.json", task_dir / "verifier/verify.json",
            assets_dir / "manifest.json", task_window=time_window)
        actions, repaired_sha = boltons_actions(task_dir, private)
        make_env = boltons_env
    else:
        private = HumanizePrivateCases(
            task_dir / "task.json", task_dir / "verifier/verify.json",
            assets_dir / "manifest.json", task_window=time_window)
        actions, repaired_sha = humanize_actions(task_dir, private)
        make_env = humanize_env
    verify_assets(private, assets_dir)
    disk = output / (label + ".qcow2")
    _clone_or_copy_qcow2(assets_dir / "rootfs.qcow2", disk)
    runtime = _runtime(assets_dir, disk)
    report = {"repository": label, "task_id": task_id, "status": "incomplete",
              "graded_branches": [], "source_sdist_sha256":
              private.source_task["metadata"]["source_sdist_sha256"],
              "verifier_sha256": private.file_digests["verifier"],
              "rootfs_qcow2_sha256": disk_sha,
              "guest_service_sha256": private.guest_service_sha256,
              "guest_case_runner_sha256": private.guest_case_runner_sha256}
    try:
        _boot(runtime)
        outer_sha = _outer_workspace_sha(runtime)
        client = connect_booted_vm(runtime, private)
        if _outer_workspace_sha(runtime) != outer_sha:
            raise RuntimeError("outer_tree_changed_during_cooperative_setup")
        report["outer_workspace_sha256"] = outer_sha
        report["fault_branch"] = _fault_branch(client, private, runtime, label=label)
        seen_nonces = {report["fault_branch"]["branch_process_nonce"]}
        by_mode = {}
        for index in range(branches):
            mode = "repair" if index % 2 == 0 else "baseline"
            if _outer_workspace_sha(runtime) != outer_sha:
                raise RuntimeError("outer_workspace_changed_before_branch")
            env = make_env(client, private, mode=mode)
            row = _episode(env, actions[mode],
                           expected_reward=1.0 if mode == "repair" else 0.0,
                           condition="cooperative", mode=mode)
            case_result_digest = _sha(json.dumps(row["case_results"], sort_keys=True,
                                                 separators=(",", ":")).encode("ascii"))
            if (row["passed_cases"] != (14 if mode == "repair" else baseline_cases)
                    or env.adapter.submitted_source_sha !=
                       (repaired_sha if mode == "repair" else private.seed_file_sha256)
                    or client.state != "idle"):
                raise RuntimeError("graded_branch_reward_source_or_cleanup_changed")
            if mode in by_mode and by_mode[mode] != case_result_digest:
                raise RuntimeError("graded_hidden_case_results_drifted_across_branches")
            by_mode.setdefault(mode, case_result_digest)
            nonce = env.adapter.branch_process_nonce
            if nonce in seen_nonces:
                raise RuntimeError("branch_process_nonce_reused")
            seen_nonces.add(nonce)
            _helper_digest(runtime, private.guest_service_sha256)
            census = _census(runtime)
            if _outer_workspace_sha(runtime) != outer_sha:
                raise RuntimeError("outer_workspace_changed_after_branch")
            report["graded_branches"].append({
                "index": index, "mode": mode, "reward": row["reward"],
                "passed_cases": row["passed_cases"],
                "case_result_digest": case_result_digest,
                "submitted_source_sha256": env.adapter.submitted_source_sha,
                "branch_process_nonce": nonce, "census": census})
        report["unique_branch_nonces"] = len(seen_nonces)
        report["status"] = "passed"
        return report
    finally:
        runtime.close()
        disk.unlink(missing_ok=True)


def stress(boltons_task, boltons_assets, humanize_task, humanize_assets, output,
           *, branches=20):
    output = Path(output).resolve()
    inputs = [Path(path).resolve() for path in
              (boltons_task, boltons_assets, humanize_task, humanize_assets)]
    if (type(branches) is not int or not 2 <= branches <= 50 or branches % 2
            or output.exists() and any(output.iterdir())
            or any(output.is_relative_to(path) or path.is_relative_to(output)
                   for path in inputs)):
        raise ValueError("stress_inputs_or_output_invalid")
    frame_bytes = _validate_fault_codes()
    output.mkdir(parents=True, exist_ok=True)
    report = {"kind": "cooperative_qemu_branch_failure_stress_v1",
              "status": "incomplete", "branches_per_repository": branches,
              "fault_case_frame_bytes": frame_bytes, "repositories": [],
              "source_binding": {
                  "stress_script_sha256": _file_sha(__file__),
                  "cooperative_host_sha256": _file_sha(
                      Path(__file__).with_name("host_runtime.py")),
                  "graded_episode_helper_sha256": _file_sha(
                      Path(__file__).with_name("benchmark.py")),
                  "cooperative_guest_sha256": _file_sha(
                      Path(__file__).with_name("guest_service.py")),
                  "cooperative_case_runner_sha256": _file_sha(
                      Path(__file__).with_name("guest_case_runner.py")),
                  "humanize_host_sha256": _file_sha(
                      Path(__file__).parents[1] / "cooperative_realworld_humanize/host_runtime.py"),
                  "humanize_action_helper_sha256": _file_sha(
                      Path(__file__).parents[1] / "cooperative_realworld_humanize/benchmark.py"),
                  "resident_case_runner_sha256": _file_sha(
                      Path(__file__).parents[1] / "resident_guest_candidate/case_runner.py"),
                  "resident_supervisor_sha256": _file_sha(
                      Path(__file__).parents[1] / "resident_guest_candidate/guest_supervisor.py"),
                  "microvm_runtime_sha256": _file_sha(
                      Path(__file__).parents[2] / "future_prediction_bench/microvm_runtime.py"),
              }}
    try:
        for label, task, assets in (("boltons", boltons_task, boltons_assets),
                                    ("humanize", humanize_task, humanize_assets)):
            report["in_progress_repository"] = label
            report["repositories"].append(
                _run_repository(label, task, assets, output, branches=branches))
        report.pop("in_progress_repository")
        report["status"] = "passed"
    except BaseException as exc:
        report["failure_type"] = type(exc).__name__
        raise
    finally:
        encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if any(marker in encoded for marker in ("/Users/", "/private/tmp/",
                                              "expected_stdout", "stdout_b64")):
            raise RuntimeError("stress_report_contains_private_detail")
        (output / "report.json").write_text(encoded, encoding="utf-8")
    return report


def public_report(report):
    if report.get("status") != "passed":
        raise ValueError("public_stress_report_requires_success")
    if {item.get("repository") for item in report.get("repositories", [])} != set(PINNED):
        raise ValueError("public_stress_report_missing_repository")
    for item in report["repositories"]:
        rows = item.get("graded_branches")
        fault = item.get("fault_branch", {})
        if (not isinstance(rows, list) or len(rows) != report["branches_per_repository"]
                or item.get("unique_branch_nonces") != len(rows) + 1
                or fault.get("fault_case_count") != 14
                or fault.get("detached_grandchild_handshake") is not True
                or fault.get("symlink_outside_write_denied") is not True
                or fault.get("next_case_workspace_copy_clean") is not True
                or fault.get("stdout_close_timeout_return_code") != 124
                or fault.get("cleanup_after_faults") != {
                    "named_candidate_descendants": 0,
                    "root_side_branch_pool_debris": 0}
                or any(row.get("census") != {
                    "named_candidate_descendants": 0,
                    "root_side_branch_pool_debris": 0} for row in rows)
                or any((row.get("reward"), row.get("passed_cases")) !=
                       ((1.0, 14) if row.get("mode") == "repair" else
                        (0.0, PINNED[item["repository"]][2])) for row in rows)
                or any(len({row["case_result_digest"] for row in rows
                            if row["mode"] == mode}) != 1
                       for mode in ("repair", "baseline"))):
            raise ValueError("public_stress_report_claim_not_proven")
    return {
        "kind": report["kind"], "status": report["status"],
        "branches_per_repository": report["branches_per_repository"],
        "fault_case_frame_bytes": report["fault_case_frame_bytes"],
        "source_binding": report["source_binding"],
        "candidate_faults": [
            "detached_grandchild_with_handshake",
            "symlink_to_outside_write_denied",
            "fresh_case_workspace_after_symlink",
            "stdout_closed_candidate_timeout",
        ],
        "repositories": [{
            "repository": item["repository"], "task_id": item["task_id"],
            "source_sdist_sha256": item["source_sdist_sha256"],
            "verifier_sha256": item["verifier_sha256"],
            "rootfs_qcow2_sha256": item["rootfs_qcow2_sha256"],
            "guest_service_sha256": item["guest_service_sha256"],
            "guest_case_runner_sha256": item["guest_case_runner_sha256"],
            "outer_workspace_sha256": item["outer_workspace_sha256"],
            "fault_branch": {key: item["fault_branch"][key] for key in (
                "fault_case_count", "detached_grandchild_handshake",
                "symlink_outside_write_denied", "next_case_workspace_copy_clean",
                "stdout_close_timeout_return_code", "cleanup_after_faults",
                "case_result_digest")},
            "graded_branch_count": len(item["graded_branches"]),
            "unique_branch_nonces": item["unique_branch_nonces"],
            "repair_reward_and_passes": [1.0, 14],
            "baseline_reward_and_passes": [0.0, PINNED[item["repository"]][2]],
            "all_case_results_stable_by_mode": True,
            "all_outer_tree_and_helper_checks_passed": True,
            "all_process_and_pool_censuses_empty": True,
        } for item in report["repositories"]],
        "scope": (
            "Trusted fault injection plus graded RealWorldEnv branches in one guest "
            "per repository; shared guest kernel and trusted root; no full-VM "
            "security parity or speed claim"),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boltons-task", required=True)
    parser.add_argument("--boltons-assets", required=True)
    parser.add_argument("--humanize-task", required=True)
    parser.add_argument("--humanize-assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--branches", type=int, default=20)
    args = parser.parse_args(argv)
    report = stress(args.boltons_task, args.boltons_assets,
                    args.humanize_task, args.humanize_assets, args.output,
                    branches=args.branches)
    public = json.dumps(public_report(report), indent=2, sort_keys=True,
                        allow_nan=False) + "\n"
    (Path(args.output) / "public-report.json").write_text(public, encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "branches_per_repository": report["branches_per_repository"],
                      "repositories": [item["repository"] for item in report["repositories"]]}))


if __name__ == "__main__":
    main()
