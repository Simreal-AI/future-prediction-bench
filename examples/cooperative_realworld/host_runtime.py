"""Host-only RealWorldEnv adapter for a cooperative guest checkpoint.

Expected verifier outputs and reward never cross the VM serial bridge. This
reuses the resident example's bounded wire, edit, and case grading code while
binding the additional process/OverlayFS checkpoint service separately.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import shlex
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from future_prediction_bench.realworld import AdapterInfrastructureError, RealWorldEnv

_RESIDENT = Path(__file__).resolve().parents[1] / "resident_guest_candidate"
if str(_RESIDENT) not in sys.path:
    sys.path.insert(0, str(_RESIDENT))

from candidate_api import GUEST_HELPERS, _required, _upload, verify_assets  # noqa: E402
from host_client import ResidentHostAdapter, ResidentInterrupted  # noqa: E402
from host_realworld_adapter import HostPrivateCases, ResidentRealWorldAdapter  # noqa: E402
from wire import (MAX_REQUEST, MAX_BATCH_REQUEST, MAX_RESPONSE, ProtocolError,  # noqa: E402
                  frame, unframe, validate_request)


IDENTITY = "trusted_cooperative_guest_process_overlay_experimental_v1"
MAX_COOP_RESPONSE = 32768
HERE = Path(__file__).resolve().parent
GUEST_SERVICE = HERE / "guest_service.py"
GUEST_CASE_RUNNER = HERE / "guest_case_runner.py"
CONTRACT = HERE / "boltons_contract.json"
RESIDENT_CONTRACT = _RESIDENT / "boltons_contract_local.json"


def _sha(data):
    return hashlib.sha256(data).hexdigest()


class CooperativePrivateCases(HostPrivateCases):
    def __init__(self, task_path, verifier_path, assets_manifest_path,
                 *, contract_path=CONTRACT, resident_contract_path=RESIDENT_CONTRACT,
                 task_window=None):
        super().__init__(task_path, verifier_path, resident_contract_path,
                         assets_manifest_path)
        cooperative_path = Path(contract_path)
        if cooperative_path.is_symlink() or not cooperative_path.is_file():
            raise ValueError("cooperative_contract_missing_or_symlink")
        data = cooperative_path.read_bytes()
        if len(data) > 4096:
            raise ValueError("cooperative_contract_oversize")
        contract = json.loads(data, object_pairs_hook=self._unique_pairs)
        if (not isinstance(contract, dict)
                or set(contract) != {
                    "kind", "task_id", "source_path", "case_count",
                    "requires_live_background_process_state",
                    "requires_shared_case_filesystem_state",
                    "requires_quiescent_submitted_state",
                    "resident_case_contract_sha256"}
                or (contract["kind"], self.task_id, self.source_path) not in {
                    ("quiescent_process_fork_frozen_overlay_boltons_v1",
                     "boltons-26-singularize-ss-v2", "boltons/strutils.py"),
                    ("quiescent_process_fork_frozen_overlay_humanize_v1",
                     "humanize-4150-naturalsize-rounding-v2",
                     "src/humanize/filesize.py"),
                }
                or contract["task_id"] != self.task_id
                or contract["source_path"] != self.source_path
                or type(contract["case_count"]) is not int
                or contract["case_count"] != len(self.cases)
                or len(self.cases) != 14
                or contract["requires_live_background_process_state"] is not False
                or contract["requires_shared_case_filesystem_state"] is not False
                or contract["requires_quiescent_submitted_state"] is not True
                or contract["resident_case_contract_sha256"] !=
                   self.file_digests["contract"]):
            raise ValueError("cooperative_checkpoint_contract_not_pinned")
        if (GUEST_SERVICE.is_symlink() or not GUEST_SERVICE.is_file()
                or GUEST_CASE_RUNNER.is_symlink() or not GUEST_CASE_RUNNER.is_file()):
            raise ValueError("cooperative_guest_helper_missing")
        self.cooperative_contract_path = cooperative_path
        self.cooperative_contract_sha256 = _sha(data)
        self.cooperative_scope = (
            "trusted_shared_guest_kernel_quiescent_"
            + ("humanize" if self.source_path == "src/humanize/filesize.py"
               else "boltons") + "_three_tools")
        self.guest_service_sha256 = _sha(GUEST_SERVICE.read_bytes())
        self.guest_case_runner_sha256 = _sha(GUEST_CASE_RUNNER.read_bytes())
        self.binding.update({
            "runtime_kind": IDENTITY,
            "checkpoint_kind": contract["kind"],
            "cooperative_contract_sha256": self.cooperative_contract_sha256,
            "cooperative_guest_service_sha256": self.guest_service_sha256,
            "cooperative_guest_case_runner_sha256": self.guest_case_runner_sha256,
        })
        if task_window is None:
            now = datetime.now(timezone.utc)
            task_window = {
                "issued_at": (now - timedelta(minutes=1)).isoformat(),
                "action_deadline": (now + timedelta(minutes=30)).isoformat(),
                "outcome_not_before": (now - timedelta(minutes=1)).isoformat(),
                "verify_after": (now - timedelta(minutes=1)).isoformat(),
            }
        time_keys = {"issued_at", "action_deadline", "outcome_not_before",
                     "verify_after"}
        if not isinstance(task_window, dict) or set(task_window) != time_keys:
            raise ValueError("cooperative_time_window_invalid")
        self.task_window = copy.deepcopy(task_window)

    @staticmethod
    def _unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_contract_key")
            result[key] = value
        return result

    def check_unchanged(self):
        super().check_unchanged()
        if (_sha(self.cooperative_contract_path.read_bytes()) !=
                self.cooperative_contract_sha256
                or _sha(GUEST_SERVICE.read_bytes()) != self.guest_service_sha256
                or _sha(GUEST_CASE_RUNNER.read_bytes()) !=
                   self.guest_case_runner_sha256):
            raise RuntimeError("cooperative_contract_or_service_changed")

    def experimental_task(self):
        task = super().experimental_task()
        task.update(copy.deepcopy(self.task_window))
        task["adapter_id"] = "cooperative_microvm_process_overlay_experimental"
        task["metadata"].pop("resident_scope", None)
        task["metadata"]["cooperative_scope"] = self.cooperative_scope
        task["metadata"]["artifact_binding"] = copy.deepcopy(self.binding)
        return task


class CooperativeTransport:
    def __init__(self, runtime, *, timeout=20.0):
        self.runtime = runtime
        self.timeout = timeout
        self.last_response = None

    def exchange(self, request):
        validate_request(request)
        limit = MAX_BATCH_REQUEST if request["op"] == "case_batch" else MAX_REQUEST
        encoded = base64.b64encode(frame(request, limit=limit)).decode("ascii")
        command = ("LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
                   "chroot /mnt/root /usr/local/bin/python3.12 -I -B "
                   "/fpb_coop_guest.py --call " + shlex.quote(encoded))
        if len(command.encode("ascii")) > 3800:
            raise ProtocolError("cooperative_serial_command_too_large")
        result = self.runtime.run_shell(
            command, timeout=max(self.timeout, 290.0 if request["op"] == "case_batch"
                                 else 25.0 if request["op"] == "case" else 0.0))
        if result.get("return_code") != 0:
            raise ResidentInterrupted("cooperative_serial_call_failed")
        match = re.fullmatch(r"FPB_COOP_V0=([A-Za-z0-9+/=]+)\n?",
                             result.get("stdout") or "")
        if match is None or len(match.group(1)) > 45000:
            raise ResidentInterrupted("cooperative_serial_response_invalid")
        data = base64.b64decode(match.group(1), validate=True)
        self.last_response = unframe(data, limit=MAX_COOP_RESPONSE)
        return self.last_response


class CooperativeClient(ResidentHostAdapter):
    def connect(self, *, expected_service_sha256, expected_case_runner_sha256,
                expected_source_sha256):
        binding = super().connect()
        hello = self.transport.last_response["value"]
        checkpoint = hello.get("cooperative_checkpoint")
        if (not isinstance(checkpoint, dict)
                or set(checkpoint) != {
                    "kind", "template_ready", "checkpoint_ns",
                    "template_mount_namespace", "frozen_source_sha256",
                    "guest_service_sha256", "guest_case_runner_sha256"}
                or checkpoint["kind"] !=
                   "quiescent_process_fork_frozen_overlay_v1"
                or checkpoint["template_ready"] is not True
                or type(checkpoint["checkpoint_ns"]) is not int
                or checkpoint["checkpoint_ns"] <= 0
                or not re.fullmatch(r"mnt:\[[0-9]+\]",
                                    checkpoint["template_mount_namespace"])
                or checkpoint["frozen_source_sha256"] != expected_source_sha256
                or checkpoint["guest_service_sha256"] != expected_service_sha256
                or checkpoint["guest_case_runner_sha256"] !=
                   expected_case_runner_sha256):
            self.state = "interrupted"
            raise ResidentInterrupted("cooperative_checkpoint_binding_invalid")
        self.checkpoint = copy.deepcopy(checkpoint)
        return dict(binding, cooperative_checkpoint=copy.deepcopy(checkpoint))


class CooperativeRealWorldAdapter(ResidentRealWorldAdapter):
    """Expose the full-VM 16 KiB read semantics for this pinned source."""

    def _guest_action(self, name, arguments):
        if name != "read_file":
            return super()._guest_action(name, arguments)
        self._policy_request(name, arguments)
        value = self.client.action(name, arguments)
        if not isinstance(value, dict):
            raise ResidentInterrupted("malformed_cooperative_read")
        if value.get("accepted") is False:
            if (set(value) != {"accepted", "reason"}
                    or not isinstance(value["reason"], str)
                    or not 0 < len(value["reason"]) <= 80):
                raise ResidentInterrupted("malformed_cooperative_read_rejection")
        elif (set(value) != {"accepted", "sha256", "bytes", "text", "truncated"}
              or value["accepted"] is not True
              or not isinstance(value["sha256"], str)
              or re.fullmatch(r"[0-9a-f]{64}", value["sha256"]) is None
              or type(value["bytes"]) is not int
              or not 0 <= value["bytes"] <= 65536
              or not isinstance(value["text"], str)
              or len(value["text"].encode("utf-8")) > 16000
              or type(value["truncated"]) is not bool):
            raise ResidentInterrupted("malformed_cooperative_read_result")
        return value

    def reset(self, task, *, now):
        observation = super().reset(task, now=now)
        observation["runtime_kind"] = IDENTITY
        observation["visible_check"] = list(getattr(
            self.private_cases, "visible_check",
            ("python3", "-B", "-c", "import boltons.strutils")))
        return observation

    def step(self, action, *, now):
        transition = super().step(action, now=now)
        if action["action"] == "submit":
            transition["observation"]["snapshot_kind"] = (
                "quiescent_process_fork_frozen_overlay_v1")
        return transition

    def get_state(self):
        value = super().get_state()
        value["runtime_kind"] = IDENTITY
        value["checkpoint"] = copy.deepcopy(self.client.checkpoint)
        return value


def connect_booted_vm(runtime, private: CooperativePrivateCases):
    if not isinstance(private, CooperativePrivateCases):
        raise TypeError("cooperative_private_cases_required")
    private.check_unchanged()
    guest_source = "/mnt/root/workspace/" + private.source_path
    result = _required(runtime, "sha256sum " + shlex.quote(guest_source))
    if result.split()[0] != private.seed_file_sha256:
        raise RuntimeError("cooperative_outer_seed_mismatch")
    _required(runtime, "mkdir -p /mnt/root/dev && "
              "(test -c /mnt/root/dev/null || mknod -m 666 /mnt/root/dev/null c 1 3)")
    _required(runtime, "modprobe overlay")
    for name, target in GUEST_HELPERS.items():
        if _upload(runtime, private.helper_paths[name], target) != private.helper_digests[name]:
            raise RuntimeError("cooperative_shared_helper_install_mismatch")
    if _upload(runtime, GUEST_SERVICE, "/mnt/root/fpb_coop_guest.py") != private.guest_service_sha256:
        raise RuntimeError("cooperative_service_install_mismatch")
    if _upload(runtime, GUEST_CASE_RUNNER,
               "/mnt/root/fpb_coop_case_runner.py") != private.guest_case_runner_sha256:
        raise RuntimeError("cooperative_case_runner_install_mismatch")
    command = ("LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/aarch64-linux-gnu "
               "chroot /mnt/root /usr/local/bin/python3.12 -I -B "
               "/fpb_coop_guest.py --serve --source-path "
               + shlex.quote(private.source_path) + " --seed-sha256 "
               + shlex.quote(private.seed_file_sha256)
               + " --case-count " + str(len(private.cases))
               + " </dev/null >/mnt/root/fpb_coop.log 2>&1 & true")
    _required(runtime, command)
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        if runtime.run_shell("test -S /mnt/root/fpb-cooperative.sock", timeout=5).get("return_code") == 0:
            break
        time.sleep(0.05)
    else:
        raise RuntimeError("cooperative_guest_daemon_not_ready")
    client = CooperativeClient(CooperativeTransport(runtime))
    client.connect(expected_service_sha256=private.guest_service_sha256,
                   expected_case_runner_sha256=private.guest_case_runner_sha256,
                   expected_source_sha256=private.seed_file_sha256)
    if (client.guest_helper_sha256 != {name: private.helper_digests[name]
                                      for name in GUEST_HELPERS}
            or client.source_path != private.source_path
            or client.seed_file_sha256 != private.seed_file_sha256
            or client.case_count != len(private.cases)):
        raise RuntimeError("cooperative_running_guest_binding_mismatch")
    return client


def make_env(client, private, *, mode, clock=None):
    if not isinstance(client, CooperativeClient) or client.state != "idle":
        raise ValueError("connected_idle_cooperative_client_required")
    adapter = CooperativeRealWorldAdapter(
        client, private, infrastructure_error_class=AdapterInfrastructureError,
        episode_mode=mode, case_transport="batch")
    task = private.experimental_task()
    return RealWorldEnv(task, adapter, clock=clock)
