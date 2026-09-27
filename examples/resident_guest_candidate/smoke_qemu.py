"""Bounded QEMU smoke for the pinned public Boltons repair fixture.

The runtime modules are task-parametric; this driver deliberately checks the
known 14-case Boltons repair/baseline parity on one fresh VM. No policy model
or weight update runs. The JSON report contains no private per-case outputs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shlex
import shutil
import secrets
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

from candidate_api import connect_booted_vm, load_fixture, make_env, verify_assets
from examples.realworld_boltons26.microvm_benchmark import _boot, _runtime
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _outer_sha(runtime, source_path):
    command = "sha256sum " + shlex.quote("/mnt/root/workspace/" + source_path)
    result = runtime.run_shell(command, timeout=10)
    if result.get("return_code") != 0:
        raise RuntimeError("outer_workspace_source_unreadable")
    match = re.fullmatch(r"([0-9a-f]{64})\s+\S+\n?", result.get("stdout", ""))
    if match is None:
        raise RuntimeError("outer_workspace_source_digest_invalid")
    return match.group(1)


def _outer_workspace_sha(runtime, *, root="/workspace"):
    """Mirror the pinned host workspace digest inside the guest, read-only."""
    source = (
        "import hashlib,stat\n"
        "from pathlib import Path\n"
        "root=Path(" + repr(str(root)) + ")\n"
        "digest=hashlib.sha256()\n"
        "for path in sorted(root.rglob('*')):\n"
        "    if path.is_symlink() or not (path.is_file() or path.is_dir()):\n"
        "        raise RuntimeError('unsupported_workspace_node')\n"
        "    relative=path.relative_to(root).as_posix().encode('utf-8')\n"
        "    digest.update(len(relative).to_bytes(4,'big'))\n"
        "    digest.update(relative)\n"
        "    digest.update(b'F' if path.is_file() else b'D')\n"
        "    digest.update(stat.S_IMODE(path.stat().st_mode).to_bytes(2,'big'))\n"
        "    if path.is_file():\n"
        "        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode('ascii'))\n"
        "print('FPB_WORKSPACE_SHA='+digest.hexdigest())\n"
    )
    encoded = base64.b64encode(source.encode("utf-8")).decode("ascii")
    expression = "exec(__import__('base64').b64decode('" + encoded + "'))"
    command = ("LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
               "chroot /mnt/root /usr/local/bin/python3.12 -I -B -c "
               + shlex.quote(expression))
    result = runtime.run_shell(command, timeout=15)
    if result.get("return_code") != 0:
        raise RuntimeError("outer_workspace_digest_failed")
    match = re.fullmatch(r"FPB_WORKSPACE_SHA=([0-9a-f]{64})\n?",
                         result.get("stdout", ""))
    if match is None:
        raise RuntimeError("outer_workspace_digest_invalid")
    return match.group(1)


def _repair_action(task_dir, private):
    path = Path(task_dir) / "actions.solution.replace_text.jsonl"
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 20000:
        raise ValueError("pinned_repair_script_missing_or_invalid")
    actions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line.strip()]
    edits = [item for item in actions if item.get("action") == "replace_text"]
    if len(edits) != 1 or edits[0].get("path") != private.source_path:
        raise ValueError("pinned_repair_action_changed")
    action = edits[0]
    source = private.seed_file_path.read_bytes()
    old = action["old_text"].encode("utf-8")
    new = action["new_text"].encode("utf-8")
    if (not old or len(old) > 256 or len(new) > 256
            or source.count(old) != 1
            or action["expected_file_sha256"] != private.seed_file_sha256):
        raise ValueError("pinned_repair_precondition_changed")
    return action, _sha(source.replace(old, new, 1))


def _write_report(path, report):
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if any(marker in encoded for marker in ("/Users/", "/private/tmp/", "expected_stdout",
                                          "private_case_results", "stdout_sha256")):
        raise RuntimeError("report_contains_private_detail")
    Path(path).write_text(encoded, encoding="utf-8")


def _case_isolation_probe(runtime, client, private, seen_nonces, *, transport,
                          expected_workspace_sha):
    """A submitted branch cannot retain a file written by an earlier case."""
    marker = ".fpb-isolation-" + secrets.token_hex(8)
    guest_path = "/workspace/" + marker
    write = ("from pathlib import Path; Path(" + repr(guest_path)
             + ").write_text('case-only'); print('X'*65,end='')")
    check = ("from pathlib import Path; print('clean' if not Path("
             + repr(guest_path) + ").exists() else 'leaked')")
    codes = [write] + [check] * (len(private.cases) - 1)
    if len(codes) != 14:
        raise ValueError("isolation_probe_requires_pinned_14_cases")
    created = client.reset("resident-isolation-" + secrets.token_hex(8), "baseline")
    nonce = created["namespaces"]["process_nonce"]
    if nonce in seen_nonces:
        raise RuntimeError("isolation_probe_reused_branch_nonce")
    seen_nonces.add(nonce)
    try:
        client.submit()
        if transport == "batch":
            results = client.run_case_batch(codes,
                                            expected_branch_sha256=private.seed_file_sha256)
        else:
            results = [client.run_case(code,
                                       expected_branch_sha256=private.seed_file_sha256)
                       for code in codes]
        first = results[0]
        if (first["return_code"] != 0 or first["truncated"]
                or first["output_over_batch_cap"] != (transport == "batch")
                or first["stdout_bytes"] != (b"" if transport == "batch" else b"X" * 65)):
            raise RuntimeError("candidate_stdout_cap_signal_incorrect")
        if any(result["return_code"] != 0 or result["truncated"]
               or result["output_over_batch_cap"]
               or result["stdout_bytes"] != b"clean\n"
               for result in results[1:]):
            raise RuntimeError("nested_case_filesystem_isolation_failed")
        cleanup = client.close(completed=True)
    except BaseException:
        if client.state in {"active", "pending"}:
            try:
                client.close()
            except BaseException:
                pass
        raise
    outer_marker = runtime.run_shell("test ! -e " + shlex.quote(
        "/mnt/root/workspace/" + marker), timeout=5)
    if outer_marker.get("return_code") != 0:
        raise RuntimeError("nested_case_marker_leaked_to_outer_workspace")
    if _outer_sha(runtime, private.source_path) != private.seed_file_sha256:
        raise RuntimeError("isolation_probe_changed_outer_source")
    if _outer_workspace_sha(runtime) != expected_workspace_sha:
        raise RuntimeError("isolation_probe_changed_outer_workspace")
    return {"status": "passed", "case_count": len(results),
            "cross_case_file_isolation": True, "outer_workspace_pristine": True,
            "candidate_stdout_65_byte_transport_checked": True,
            "guest_cleanup_ns": cleanup["guest_cleanup_ns"]}


def _candidate_timeout_probe(runtime, client, private, seen_nonces, *, transport,
                             expected_workspace_sha, task_dir, repair_action):
    """Grade an ephemeral host-only verifier whose first case times out."""
    with tempfile.TemporaryDirectory(prefix="fpb-resident-timeout-") as temporary:
        root = Path(temporary)
        shutil.copytree(task_dir / "seed", root / "seed")
        shutil.copy2(task_dir / "task.json", root / "task.json")
        verifier = json.loads((task_dir / "verifier/verify.json").read_text(
            encoding="utf-8"))
        verifier["cases"][0]["argv"][3] = (
            "import os,time; os.close(1); time.sleep(11)")
        verifier_path = root / "verifier/verify.json"
        verifier_path.parent.mkdir()
        verifier_path.write_text(
            json.dumps(verifier, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        contract = json.loads(private.paths["contract"].read_text(encoding="utf-8"))
        contract["verifier_sha256"] = _sha(verifier_path.read_bytes())
        contract_path = root / "contract.json"
        contract_path.write_text(
            json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        timeout_private = load_fixture(
            task_path=root / "task.json", verifier_path=verifier_path,
            contract_path=contract_path,
            assets_manifest_path=private.paths["assets_manifest"])
        task = timeout_private.experimental_task()
        issued = datetime.fromisoformat(task["issued_at"])
        env = make_env(client, timeout_private, episode_mode="repair",
                       case_transport=transport,
                       clock=lambda: issued + timedelta(minutes=1))
        started = time.perf_counter()
        env.reset("scripted-timeout-reward-probe")
        nonce = env.adapter.branch_process_nonce
        if nonce in seen_nonces:
            raise RuntimeError("timeout_probe_reused_branch_nonce")
        seen_nonces.add(nonce)
        read = env.step({"action": "read_file", "path": private.source_path})
        if read["observation"].get("sha256") != private.seed_file_sha256:
            raise RuntimeError("timeout_probe_branch_not_pristine")
        edited = env.step(repair_action)
        if edited["observation"].get("sha256") is None:
            raise RuntimeError("timeout_probe_repair_failed")
        submitted = env.step({"action": "submit"})
        if submitted["info"]["status"] != "pending":
            raise RuntimeError("timeout_probe_submit_not_pending")
        grade = env.verify()
        if (grade["status"] != "graded" or grade["reward"] != 0.0
                or grade["evidence"]["case_count"] != 14
                or grade["evidence"]["passed_count"] != 13
                or client.state != "idle"):
            raise RuntimeError("candidate_timeout_was_not_graded_failure")
        audit = env.adapter.private_case_audit()
        if (audit[0]["return_code"] != 124 or audit[0]["passed"]
                or any(not case["passed"] for case in audit[1:])):
            raise RuntimeError("candidate_timeout_case_audit_mismatch")
        probe_seconds = time.perf_counter() - started
        cleanup_ns = env.adapter.metrics["guest_cleanup_ns"]
    if _outer_workspace_sha(runtime) != expected_workspace_sha:
        raise RuntimeError("timeout_probe_changed_outer_workspace")
    return {"status": "passed", "case_count": 14,
            "candidate_timeout_return_code": 124,
            "graded_reward": 0.0, "passed_cases": 13,
            "later_cases_completed": 13,
            "outer_workspace_pristine": True,
            "probe_wall_seconds": probe_seconds,
            "guest_cleanup_ns": cleanup_ns}


def _candidate_resource_probe(runtime, client, private, seen_nonces, *, transport,
                              expected_workspace_sha, task_dir, repair_action):
    """Prove guest RLIMITs apply before exec and a violation remains scoreable."""
    code = (
        "import resource as r,sys,os,errno\n"
        "caps=((r.RLIMIT_CPU,8),(r.RLIMIT_AS,134217728),"
        "(r.RLIMIT_NPROC,2),(r.RLIMIT_FSIZE,8388608),"
        "(r.RLIMIT_NOFILE,64),(r.RLIMIT_CORE,0))\n"
        "if not all(0<=r.getrlimit(k)[0]==r.getrlimit(k)[1]<=n for k,n in caps): "
        "sys.exit(63)\n"
        "p='/workspace/.fpb-resource-probe'\n"
        "try:\n with open(p,'wb') as f:\n"
        "  for _ in range(144): f.write(b'x'*65536)\n"
        "except OSError as e: sys.exit(74 if e.errno==errno.EFBIG "
        "and os.path.getsize(p)==8388608 else 62)\n"
        "sys.exit(61)\n"
    )
    if len(code.encode("utf-8")) > 512:
        raise RuntimeError("resource_probe_exceeds_case_bound")
    with tempfile.TemporaryDirectory(prefix="fpb-resident-resource-") as temporary:
        root = Path(temporary)
        shutil.copytree(task_dir / "seed", root / "seed")
        shutil.copy2(task_dir / "task.json", root / "task.json")
        verifier = json.loads((task_dir / "verifier/verify.json").read_text(
            encoding="utf-8"))
        verifier["cases"][0]["argv"][3] = code
        verifier_path = root / "verifier/verify.json"
        verifier_path.parent.mkdir()
        verifier_path.write_text(
            json.dumps(verifier, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        contract = json.loads(private.paths["contract"].read_text(encoding="utf-8"))
        contract["verifier_sha256"] = _sha(verifier_path.read_bytes())
        contract_path = root / "contract.json"
        contract_path.write_text(
            json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        resource_private = load_fixture(
            task_path=root / "task.json", verifier_path=verifier_path,
            contract_path=contract_path,
            assets_manifest_path=private.paths["assets_manifest"])
        task = resource_private.experimental_task()
        issued = datetime.fromisoformat(task["issued_at"])
        env = make_env(client, resource_private, episode_mode="repair",
                       case_transport=transport,
                       clock=lambda: issued + timedelta(minutes=1))
        started = time.perf_counter()
        env.reset("scripted-resource-reward-probe")
        nonce = env.adapter.branch_process_nonce
        if nonce in seen_nonces:
            raise RuntimeError("resource_probe_reused_branch_nonce")
        seen_nonces.add(nonce)
        read = env.step({"action": "read_file", "path": private.source_path})
        if read["observation"].get("sha256") != private.seed_file_sha256:
            raise RuntimeError("resource_probe_branch_not_pristine")
        edited = env.step(repair_action)
        if edited["observation"].get("sha256") is None:
            raise RuntimeError("resource_probe_repair_failed")
        submitted = env.step({"action": "submit"})
        if submitted["info"]["status"] != "pending":
            raise RuntimeError("resource_probe_submit_not_pending")
        grade = env.verify()
        if (grade["status"] != "graded" or grade["reward"] != 0.0
                or grade["evidence"]["case_count"] != 14
                or grade["evidence"]["passed_count"] != 13
                or client.state != "idle"):
            raise RuntimeError("candidate_resource_violation_not_graded_failure")
        audit = env.adapter.private_case_audit()
        # Exit 74 confirms EFBIG at exactly 8 MiB. A default SIGXFSZ may
        # instead kill the candidate; other exits cannot prove this guard.
        if (audit[0]["return_code"] not in {74, -25} or audit[0]["passed"]
                or any(not case["passed"] for case in audit[1:])):
            raise RuntimeError("candidate_resource_case_audit_mismatch")
        probe_seconds = time.perf_counter() - started
        cleanup_ns = env.adapter.metrics["guest_cleanup_ns"]
    if _outer_workspace_sha(runtime) != expected_workspace_sha:
        raise RuntimeError("resource_probe_changed_outer_workspace")
    return {"status": "passed", "case_count": 14,
            "candidate_resource_return_code": audit[0]["return_code"],
            "graded_reward": 0.0, "passed_cases": 13,
            "later_cases_completed": 13,
            "outer_workspace_pristine": True,
            "probe_wall_seconds": probe_seconds,
            "guest_cleanup_ns": cleanup_ns}


def run(*, task_dir, assets_dir, contract_path, output_dir, case_transport="batch"):
    task_dir, assets_dir, output_dir = (Path(p).resolve() for p in
                                      (task_dir, assets_dir, output_dir))
    if case_transport not in {"sequential", "batch"}:
        raise ValueError("invalid_case_transport")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("output_must_be_new_or_empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report = {
        "kind": "resident_candidate_pinned_boltons_two_episode_qemu_smoke_v1",
        "status": "incomplete", "security_boundary": "trusted_shared_guest_kernel_overlay",
        "case_transport": case_transport, "cases_per_episode": 14,
        "policy_training_performed": False, "episodes": [],
    }
    runtime = None
    disk = output_dir / "child.qcow2"
    stage = "preflight"
    try:
        private = load_fixture(
            task_path=task_dir / "task.json",
            verifier_path=task_dir / "verifier/verify.json",
            contract_path=contract_path,
            assets_manifest_path=assets_dir / "manifest.json")
        if len(private.cases) != 14:
            raise ValueError("pinned_boltons_case_count_changed")
        verify_assets(private, assets_dir)
        expected_workspace_sha = json.loads(
            (assets_dir / "manifest.json").read_text(encoding="utf-8")
        )["seed_workspace_sha256"]
        action, repaired_sha = _repair_action(task_dir, private)
        report["binding"] = private.binding
        stage = "clone_boot"
        started = time.perf_counter()
        report["disk_clone_mode"] = _clone_or_copy_qcow2(assets_dir / "rootfs.qcow2", disk)
        runtime = _runtime(assets_dir, disk)
        _boot(runtime)
        if _outer_sha(runtime, private.source_path) != private.seed_file_sha256:
            raise RuntimeError("outer_source_not_pristine_before_install")
        if _outer_workspace_sha(runtime) != expected_workspace_sha:
            raise RuntimeError("outer_workspace_not_pristine_before_install")
        report["clone_boot_seconds"] = time.perf_counter() - started
        stage = "install_connect"
        started = time.perf_counter()
        client = connect_booted_vm(runtime, private)
        report["install_connect_seconds"] = time.perf_counter() - started
        report["guest_helper_sha256"] = client.guest_helper_sha256
        report["boot_id"] = client.boot_id
        report["source_path"] = client.source_path
        report["seed_file_sha256"] = client.seed_file_sha256
        task = private.experimental_task()
        issued = datetime.fromisoformat(task["issued_at"])
        clock = lambda: issued + timedelta(minutes=1)
        seen_episode_ids, seen_nonces = set(), set()
        for mode, expected_reward, expected_passes in (
                ("repair", 1.0, 14), ("baseline", 0.0, 7)):
            stage = "episode_" + mode
            episode = {"mode": mode, "status": "incomplete"}
            report["episodes"].append(episode)
            if _outer_sha(runtime, private.source_path) != private.seed_file_sha256:
                raise RuntimeError("outer_source_changed_before_episode")
            if _outer_workspace_sha(runtime) != expected_workspace_sha:
                raise RuntimeError("outer_workspace_changed_before_episode")
            env = make_env(client, private, episode_mode=mode,
                           case_transport=case_transport, clock=clock)
            started = time.perf_counter()
            env.reset("scripted-resident-candidate-smoke")
            adapter = env.adapter
            if (adapter.guest_episode_id in seen_episode_ids
                    or adapter.branch_process_nonce in seen_nonces):
                raise RuntimeError("guest_branch_identity_reused")
            seen_episode_ids.add(adapter.guest_episode_id)
            seen_nonces.add(adapter.branch_process_nonce)
            read = env.step({"action": "read_file", "path": private.source_path})
            if read["observation"].get("sha256") != private.seed_file_sha256:
                raise RuntimeError("resident_branch_source_not_pristine")
            if mode == "repair":
                edited = env.step(action)
                if edited["observation"].get("sha256") != repaired_sha:
                    raise RuntimeError("resident_repair_digest_mismatch")
            submitted = env.step({"action": "submit"})
            if (submitted["info"]["status"] != "pending" or submitted["reward"] is not None
                    or adapter.submitted_source_sha !=
                    (repaired_sha if mode == "repair" else private.seed_file_sha256)):
                raise RuntimeError("submit_not_pending_or_wrong_source")
            if _outer_sha(runtime, private.source_path) != private.seed_file_sha256:
                raise RuntimeError("branch_edit_leaked_to_outer_workspace")
            if _outer_workspace_sha(runtime) != expected_workspace_sha:
                raise RuntimeError("branch_edit_leaked_to_outer_workspace_tree")
            verify_started = time.perf_counter()
            grade = env.verify()
            verify_seconds = time.perf_counter() - verify_started
            episode_seconds = time.perf_counter() - started
            if (grade["status"] != "graded" or grade["reward"] != expected_reward
                    or grade["evidence"]["case_count"] != 14
                    or grade["evidence"]["passed_count"] != expected_passes
                    or client.state != "idle"):
                raise RuntimeError("resident_reward_case_or_cleanup_mismatch")
            if _outer_sha(runtime, private.source_path) != private.seed_file_sha256:
                raise RuntimeError("outer_source_changed_after_episode")
            if _outer_workspace_sha(runtime) != expected_workspace_sha:
                raise RuntimeError("outer_workspace_changed_after_episode")
            state = adapter.get_state()
            episode.update({
                "status": "graded", "reward": grade["reward"],
                "passed_cases": grade["evidence"]["passed_count"],
                "submitted_source_sha256": adapter.submitted_source_sha,
                "outer_source_sha256": private.seed_file_sha256,
                "episode_wall_seconds": episode_seconds,
                "verify_seconds": verify_seconds,
                "guest_reset_ns": state["metrics"]["guest_reset_ns"],
                "guest_cleanup_ns": state["metrics"]["guest_cleanup_ns"],
                "guest_case_total_ns": sum(state["metrics"]["guest_case_ns"]),
                "case_requests": sum(phase["op"] in {"case", "case_batch"}
                                     for phase in state["host_requests"]),
            })
        stage = "case_isolation_probe"
        report["isolation_probe"] = _case_isolation_probe(
            runtime, client, private, seen_nonces, transport=case_transport,
            expected_workspace_sha=expected_workspace_sha)
        stage = "candidate_timeout_probe"
        report["timeout_probe"] = _candidate_timeout_probe(
            runtime, client, private, seen_nonces, transport=case_transport,
            expected_workspace_sha=expected_workspace_sha, task_dir=task_dir,
            repair_action=action)
        stage = "candidate_resource_probe"
        report["resource_probe"] = _candidate_resource_probe(
            runtime, client, private, seen_nonces, transport=case_transport,
            expected_workspace_sha=expected_workspace_sha, task_dir=task_dir,
            repair_action=action)
        report["outer_workspace_sha256"] = expected_workspace_sha
        report["status"] = "passed"
    except BaseException as exc:
        report["status"] = "failed"
        report["failure_stage"] = stage
        report["failure_type"] = type(exc).__name__
        raise
    finally:
        if runtime is not None:
            runtime.close()
        disk.unlink(missing_ok=True)
        _write_report(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--case-transport", choices=("sequential", "batch"), default="batch")
    args = parser.parse_args(argv)
    result = run(task_dir=args.task_dir, assets_dir=args.assets_dir,
                 contract_path=args.contract, output_dir=args.output,
                 case_transport=args.case_transport)
    print(json.dumps({"status": result["status"],
                      "rewards": [episode["reward"] for episode in result["episodes"]]}))


if __name__ == "__main__":
    main()
