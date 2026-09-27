"""Extend a marked CRIU guest with the exact measured ZFS/kernel cohort.

The offline ARM/Linux utility only extracts archives and creates an owned
filesystem image. It never installs packages or runs x86 guest programs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import time

from examples.official_crab_criu.fetch_zfs_inputs import new_output_path, sha, validate_inputs


def prepare(base_assets, inputs, output, *, build_image):
    base_assets, inputs = Path(base_assets).resolve(), Path(inputs).resolve()
    output = new_output_path(output)
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (base_assets, inputs)):
        raise ValueError("new_disjoint_nonsymlink_output_required")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", build_image):
        raise ValueError("immutable_cached_build_image_required")
    manifest_path = base_assets / "manifest.json"
    if manifest_path.is_symlink():
        raise ValueError("regular_base_manifest_required")
    manifest = json.loads(manifest_path.read_bytes())
    if manifest.get("schema_version") != "official-crab-criu-guest-assets-v1":
        raise ValueError("marked_original_CRIU_base_assets_required")
    base_disk = base_assets / "rootfs.qcow2"
    if base_disk.is_symlink() or sha(base_disk) != manifest["rootfs_qcow2_sha256"]:
        raise ValueError("base_qcow_pin_mismatch")
    for name, expected in manifest["guest_probe_sha256"].items():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
            raise ValueError("bounded_probe_name_required")
    validate_inputs(inputs)
    output.mkdir(parents=True)
    source = Path(__file__).parent
    builder = source / "zfs_linux_builder.py"
    plugin_builder = source / "linux_rootfs_builder.py"
    input_checker = source / "fetch_zfs_inputs.py"
    input_plan = source / "zfs_package_inputs.json"
    sources = (Path(__file__), builder, plugin_builder, input_checker, input_plan)
    report = {"schema_version": "official-crab-zfs-offline-host-build-v1", "passed": False,
              "network_used": False, "guest_executed": False, "commands": [],
              "source_sha256": {p.name: sha(p) for p in sources},
              "base_manifest_sha256": sha(manifest_path), "inputs_manifest_sha256": sha(inputs / "inputs.json")}

    def run(argv, timeout=300):
        begin = time.perf_counter()
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        report["commands"].append({"argv": argv, "returncode": result.returncode,
                                    "stdout": result.stdout, "stderr": result.stderr,
                                    "wall_seconds": time.perf_counter() - begin})
        if result.returncode:
            raise RuntimeError("actual_offline_command_failed:" + json.dumps(report["commands"][-1]))
        return result.stdout

    raw = output / "base-rootfs.raw"
    container = None
    try:
        image = json.loads(run(["docker", "image", "inspect", build_image]))
        if len(image) != 1 or image[0].get("Id") != build_image or image[0].get("Architecture") != "arm64":
            raise ValueError("cached_native_ARM_utility_image_required")
        run(["qemu-img", "convert", "-f", "qcow2", "-O", "raw", str(base_disk), str(raw)])
        disk_info = json.loads(run(["qemu-img", "info", "--output=json", str(raw)]))
        if disk_info["virtual-size"] < 3 * 1024 ** 3:
            raise ValueError("prepare_original_CRIU_base_with_disk_mib3072_first")
        container = run(["docker", "create", "--pull", "never", "--platform", "linux/arm64",
            "--network", "none", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--read-only", "--pids-limit", "64", "--memory", "2g", "--cpus", "2",
            "--tmpfs", "/linux-stage:rw,nosuid,nodev,size=1g", "--tmpfs", "/tmp:rw,nosuid,nodev,size=32m",
            "--mount", f"type=bind,src={base_assets},dst=/base-assets,readonly",
            "--mount", f"type=bind,src={inputs},dst=/inputs,readonly",
            "--mount", f"type=bind,src={source},dst=/builder,readonly",
            "--mount", f"type=bind,src={output},dst=/output", "--entrypoint", "python3.12", build_image,
            "-I", "-B", "/builder/zfs_linux_builder.py", "--base", "/output/base-rootfs.raw",
            "--assets", "/base-assets", "--inputs", "/inputs", "--output", "/output",
            "--plugin-builder", "/builder/linux_rootfs_builder.py"]).strip()
        if not re.fullmatch(r"[0-9a-f]{64}", container):
            raise ValueError("actual_created_container_ID_required")
        run(["docker", "start", "-a", container], timeout=480)
        state = json.loads(run(["docker", "inspect", container]))[0]
        report["finished_container_state"] = state["State"]
        if state["State"]["Running"] or state["State"]["ExitCode"] != 0:
            raise RuntimeError("actual_completed_offline_assembly_required")
        built = json.loads((output / "manifest.json").read_bytes())
        if built["base_manifest_sha256"] != sha(manifest_path):
            raise ValueError("base_manifest_changed_during_build")
        run(["qemu-img", "convert", "-f", "raw", "-O", "qcow2",
             str(output / "rootfs.raw"), str(output / "rootfs.qcow2")])
        built.update(rootfs_qcow2_sha256=sha(output / "rootfs.qcow2"),
                     rootfs_qcow2_bytes=(output / "rootfs.qcow2").stat().st_size,
                     qemu_img_conversion_executed=True, build_image_sha256=build_image,
                     base_guest_crab=manifest["guest_crab"], base_guest_probe_sha256=manifest["guest_probe_sha256"])
        (output / "manifest.json").write_text(json.dumps(built, indent=2) + "\n")
        report["passed"] = True
    except Exception as exc:
        report["error"] = str(exc)
    finally:
        if container:
            try:
                run(["docker", "rm", container])
            except Exception as exc:
                report["container_cleanup_error"] = str(exc)
                report["passed"] = False
        report["source_sha256_after"] = {p.name: sha(p) for p in sources}
        if report["source_sha256_after"] != report["source_sha256"]:
            report["passed"] = False
            report["source_changed_during_build"] = True
        (output / "host-build-result.json").write_text(json.dumps(report, indent=2) + "\n")
    if report["passed"]:
        raw.unlink()
        (output / "rootfs.raw").unlink()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-assets", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--build-image", required=True)
    args = parser.parse_args()
    result = prepare(args.base_assets, args.inputs, args.output, build_image=args.build_image)
    print(json.dumps({"passed": result["passed"], "error": result.get("error")}, indent=2))
    raise SystemExit(0 if result["passed"] else 2)
