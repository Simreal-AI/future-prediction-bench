"""Real x86 QEMU/TCG assets for the bounded Crab code probe only.

This uses MicroVMRuntime's explicit x86_64_tcg backend. TCG measurements
must not be compared to the project's ARM/HVF ones. KVM is not measured here.
"""
import json
from pathlib import Path

from future_prediction_bench.microvm_runtime import MicroVMRuntime
from examples.realworld_boltons26.microvm_benchmark import _required, _sha


def runtime(assets, disk):
    assets = Path(assets)
    manifest = json.loads((assets / "manifest.json").read_text())
    if manifest.get("architecture") != "linux/amd64":
        raise ValueError("x86_assets_required")
    modloop = assets / "modloop-virt-padded.raw"
    if _sha(modloop) != manifest["modloop_disk_sha256"]:
        raise ValueError("module_disk_pin_mismatch")
    return MicroVMRuntime(assets / "vmlinuz-virt", assets / "initramfs-virt", disk,
        kernel_sha256=manifest["alpine_sha256"]["vmlinuz-virt"],
        initramfs_sha256=manifest["alpine_sha256"]["initramfs-virt"],
        readonly_disk_paths=(modloop,), backend="x86_64_tcg", command_timeout=90)


def boot(vm):
    vm.start()
    vm.wait_for_serial("Launching initramfs emergency recovery shell", timeout=90)
    # PCI enumeration differs from the ARM virt board. Discover by magic,
    # requiring exactly one SquashFS and one ext4 among these fixed devices.
    squash, ext4 = [], []
    for device in ("/dev/vda", "/dev/vdb"):
        if "68 73 71 73" in _required(vm, "hexdump -C -n 4 " + device):
            squash.append(device)
        if "53 ef" in _required(vm, "dd if=" + device + " bs=1 skip=1080 count=2 2>/dev/null | hexdump -C"):
            ext4.append(device)
    if len(squash) != 1 or len(ext4) != 1 or squash == ext4:
        raise RuntimeError("x86_pinned_disk_layout_not_found")
    for command in ("mkdir -p /media/modloop /mnt/root",
                    "mount -t squashfs -o ro " + squash[0] + " /media/modloop",
                    "mount --bind /media/modloop/modules /lib/modules",
                    "modprobe ext4", "mount -t ext4 " + ext4[0] + " /mnt/root"):
        _required(vm, command)
    result = _required(vm, "LD_LIBRARY_PATH=/usr/local/lib:/usr/lib/x86_64-linux-gnu "
                       "chroot /mnt/root /usr/local/bin/python3.12 --version")
    if "Python 3.12" not in result:
        raise RuntimeError("x86_guest_python_missing")
