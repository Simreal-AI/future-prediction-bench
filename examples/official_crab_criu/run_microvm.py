"""Run the real upstream process-chain probe in disposable x86 QEMU/TCG.

Only the guest receives proc/sys/dev mounts and module requests. The host
provides no network device or directory mount to QEMU. Failed evidence is
retained; a successful disposable disk is removed after QEMU has exited.
"""
import argparse
import json
from pathlib import Path
import shlex
import time

from examples.official_crab.x86_runtime import boot, runtime
from examples.realworld_boltons26.microvm_benchmark import _required, _sha
from future_prediction_bench.microvm_runtime import _clone_or_copy_qcow2


def check(assets, output, *, memory_mib=8, mode_order="forward", diagnose_dirty=False,
          probe_program="chain"):
    if type(memory_mib) is not int or not 1 <= memory_mib <= 64:
        raise ValueError("memory_mib must be in [1,64]")
    if mode_order not in ("forward", "rotate", "rotate2") or type(diagnose_dirty) is not bool:
        raise ValueError("reviewed_mode_order_and_boolean_diagnostic_flag_required")
    if probe_program not in ("chain", "paired_epoch"):
        raise ValueError("reviewed_probe_program_required")
    if probe_program == "paired_epoch" and (mode_order != "forward" or diagnose_dirty):
        raise ValueError("paired_epoch_has_separate_fixed_trials")
    assets, output = Path(assets).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(assets) or assets.is_relative_to(output):
        raise ValueError("new_disjoint_output_required")
    manifest = json.loads((assets / "manifest.json").read_text())
    if manifest.get("schema_version") != "official-crab-criu-guest-assets-v1":
        raise ValueError("marked_criu_guest_assets_required")
    base = assets / "rootfs.qcow2"
    if base.is_symlink() or _sha(base) != manifest["rootfs_qcow2_sha256"]:
        raise ValueError("base_disk_pin_mismatch")
    output.mkdir(parents=True)
    disk = output / "disposable.qcow2"
    _clone_or_copy_qcow2(base, disk)
    result = {"schema_version": "official-crab-criu-microvm-driver-v1",
        "backend": "x86_64_tcg", "asset_manifest_sha256": _sha(assets / "manifest.json"),
        "model_inference": False, "optimizer": False, "graded_software_episode": False,
        "mode_order": mode_order, "diagnose_dirty": diagnose_dirty, "probe_program": probe_program,
        "setup_commands": [], "passed": False}
    started = time.perf_counter()
    try:
        with runtime(assets, disk) as vm:
            boot(vm)
            commands = [
                "mkdir -p /mnt/root/proc /mnt/root/sys /mnt/root/dev /mnt/root/run /mnt/root/tmp /sys/fs/cgroup /dev/pts",
                "mountpoint -q /sys/fs/cgroup || mount -t cgroup2 none /sys/fs/cgroup",
                "mountpoint -q /dev/pts || mount -t devpts devpts /dev/pts",
                "mount --bind /proc /mnt/root/proc",
                "mount --bind /sys /mnt/root/sys",
                "mount --bind /sys/fs/cgroup /mnt/root/sys/fs/cgroup",
                "mount --bind /dev /mnt/root/dev",
                "mount --bind /dev/pts /mnt/root/dev/pts",
                "mount -t tmpfs tmpfs /mnt/root/run",
                "ip link set lo up",
            ]
            for command in commands:
                response = vm.run_shell(command, timeout=30)
                result["setup_commands"].append(dict(command=command, **response))
                if response["return_code"]:
                    raise RuntimeError("guest_mount_setup_failed")
            # Modules are requested inside the disposable guest only. Capture
            # unsupported requests; actual CRIU preflight decides capability.
            for module in ("unix_diag", "inet_diag", "tcp_diag", "udp_diag",
                           "netlink_diag", "af_packet_diag", "tun", "nf_tables",
                           "x_tables", "nft_compat", "xt_mark", "ip_tables", "ip6_tables"):
                command = "modprobe " + module
                result["setup_commands"].append(dict(command=command, **vm.run_shell(command)))
            result["setup_seconds"] = time.perf_counter() - started
            # A chroot alone leaves the namespace rooted in the initramfs.
            # CRIU rejects that mismatch. Move the real ext4 mount onto /
            # in a private namespace and chroot from its saved cwd, as in
            # BusyBox's root handoff, without deleting shared initramfs files.
            program = "check_chain.py" if probe_program == "chain" else "paired_epoch_probe.py"
            arguments = " --memory-mib " + str(memory_mib)
            if probe_program == "chain":
                arguments += " --mode-order " + mode_order + (" --diagnose-dirty" if diagnose_dirty else "")
            probe_script = (
                "/bin/busybox mkdir -p /bin /sbin /usr/bin /usr/sbin; "
                "/bin/busybox --install -s; "
                "test \"$(/bin/busybox readlink /proc/self/root)\" = /; "
                "exec env PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
                "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/lib "
                "/usr/local/bin/python3.12 -B /opt/fpb/probe/" + program + " "
                "--crab-source /opt/fpb/crab --output /tmp/criu-chain-run1 "
                + arguments)
            # chroot('/') would retain the old cached root. '.' addresses
            # the ext4 mount saved before MS_MOVE; rootfs cannot be pivoted.
            namespace_script = ("mount --make-rprivate /; cd /mnt/root; mount --move . /; "
                "exec /bin/busybox chroot . /bin/sh -ec " + shlex.quote(probe_script))
            guest_command = (
                "unshare -m /bin/sh -ec " + shlex.quote(namespace_script) +
                " >/mnt/root/tmp/criu-chain-result.json 2>/mnt/root/tmp/criu-chain-stderr.log")
            result["guest_root_setup"] = "private_mount_namespace_ext4_mount_move_chroot_cwd"
            result["guest_execution"] = vm.run_shell(guest_command, timeout=300)
            result["guest_stderr"] = _required(vm, "cat /mnt/root/tmp/criu-chain-stderr.log")
            raw = _required(vm, "cat /mnt/root/tmp/criu-chain-result.json")
            (output / "guest-result.json").write_text(raw + "\n")
            result["guest_result"] = json.loads(raw)
            success_field = "passed" if probe_program == "chain" else "expected_diagnostic_passed"
            result["passed_meaning"] = ("all_actual_process_recoveries_passed" if probe_program == "chain"
                else "controlled_diagnostic_expected_outcomes_not_all_actual_recoveries")
            result["passed"] = (result["guest_execution"]["return_code"] == 0
                                and result["guest_result"].get(success_field) is True)
            # Keep each selected command's real CRIU tail independently.
            # One global tail can otherwise silently omit entire modes.
            criu_logs = []
            mode_rows = [(mode["mode"], mode) for mode in result["guest_result"].get("modes", [])]
            if probe_program == "paired_epoch":
                mode_rows = [(trial["trial"], trial["mode_result"])
                             for trial in result["guest_result"].get("trials", [])]
            for label, mode in mode_rows:
                if label not in ("every_turn_full", "selective_full", "selective_incremental",
                        "original_zero_write_control", "original_interleaved_write",
                        "complete_process_parent_candidate"):
                    raise ValueError("recorded_probe_label_outside_reviewed_trials")
                commands = [command for row in mode.get("turns", [])
                            for command in row.get("checkpoint_commands", [])]
                if mode.get("restore_command"):
                    commands.append(mode["restore_command"])
                for index, command in enumerate(commands):
                    if "--work-path" not in command:
                        continue
                    directory = command[command.index("--work-path") + 1]
                    if not isinstance(directory, str) or not directory.startswith("/tmp/criu-chain-run1/"):
                        raise ValueError("recorded_guest_log_directory_outside_owned_probe")
                    name = "restore.log" if "restore" in command else "dump.log"
                    guest_path = "/mnt/root" + directory + "/" + name
                    response = vm.run_shell("tail -c 16000 " + shlex.quote(guest_path), timeout=30)
                    filename = label + "-" + str(index) + "-" + name
                    (output / filename).write_text(response["stdout"] + "\n")
                    criu_logs.append({"mode": mode["mode"], "trial_label": label, "command_index": index,
                        "guest_path": directory + "/" + name,
                        "output_file": filename, "return_code": response["return_code"],
                        "tail_sha256": _sha(output / filename)})
            result["per_command_criu_log_tails"] = criu_logs
            result["guest_diagnostics"] = []
            diagnostics = [
                "dmesg | tail -c 16000",
                "PATH=/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
                "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib:/lib "
                "chroot /mnt/root /usr/sbin/criu check --network-lock nftables",
            ]
            for command in diagnostics:
                result["guest_diagnostics"].append(dict(command=command, **vm.run_shell(command, timeout=30)))
            # Preserve CRIU failure diagnostics, which live outside stdout.
            log_command = ("find /mnt/root/tmp/criu-chain-run1 -name '*.log' -type f "
                           "-exec sh -c 'echo LOG:$1; tail -c 16000 \"$1\"' sh {} \\; "
                           "2>/dev/null | tail -c 180000")
            logs = vm.run_shell(log_command, timeout=30)
            (output / "guest-runtime-logs.txt").write_text(logs["stdout"] + "\n")
    except Exception as exc:
        result["error"] = str(exc)
    result["driver_wall_seconds"] = time.perf_counter() - started
    result["driver_sha256"] = _sha(Path(__file__))
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["passed"]:
        disk.unlink()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--memory-mib", type=int, default=8)
    parser.add_argument("--mode-order", choices=("forward", "rotate", "rotate2"), default="forward")
    parser.add_argument("--diagnose-dirty", action="store_true")
    parser.add_argument("--probe-program", choices=("chain", "paired_epoch"), default="chain")
    args = parser.parse_args()
    report = check(args.assets, args.output, memory_mib=args.memory_mib,
                   mode_order=args.mode_order, diagnose_dirty=args.diagnose_dirty,
                   probe_program=args.probe_program)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 2)
