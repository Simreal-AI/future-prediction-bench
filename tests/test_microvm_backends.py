"""QEMU backend configuration and template binding contracts.

Socket-process doubles check propagation and launch topology. These tests
do not establish guest boot, kernel capability, KVM availability or speed.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from future_prediction_bench.microvm_runtime import MicroVMRuntime, MicroVMRuntimeError, _sha256_file
from future_prediction_bench.microvm_template import MicroVMTemplate, _canonical_bytes
from test_microvm_runtime import FakeQEMUProcess


class MicroVMBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.kernel, self.initramfs, self.disk, self.readonly = [self.root / name
            for name in ("kernel", "initramfs", "disk.qcow2", "modules.raw")]
        for path in (self.kernel, self.initramfs, self.disk, self.readonly):
            path.write_bytes(path.name.encode())
        self.processes = []

    def factory(self, args, **kwargs):
        process = FakeQEMUProcess(args, **kwargs)
        if "-S" in args:
            process.snapshots.add("warm")
        self.processes.append(process)
        return process

    def runtime(self, **overrides):
        return MicroVMRuntime(self.kernel, self.initramfs, self.disk,
            kernel_sha256=_sha256_file(self.kernel), initramfs_sha256=_sha256_file(self.initramfs),
            readonly_disk_paths=(self.readonly,), popen_factory=self.factory, **overrides)

    @staticmethod
    def fake_artifacts(vm):
        vm._readonly_disk_sha256s = tuple(_sha256_file(path) for path in vm.readonly_disk_paths)

    @staticmethod
    def fake_connect(vm, path, deadline):
        return vm._process.host_sockets.pop(0)

    def test_defaults_preserve_arm_hvf_configuration(self):
        vm = self.runtime()
        self.assertEqual(vm.backend, "aarch64_hvf")
        self.assertEqual(vm.qemu_binary, "qemu-system-aarch64")
        self.assertEqual(vm.kernel_append, "console=ttyAMA0")

    def test_invalid_backend_is_rejected_before_artifact_access(self):
        for invalid in (None, True, [], "x86", "x86_64_hvf", "aarch64_tcg"):
            with self.subTest(backend=invalid), self.assertRaisesRegex(ValueError, "backend"):
                self.runtime(backend=invalid)

    def test_x86_defaults_derive_binary_and_console(self):
        for backend in ("x86_64_tcg", "x86_64_kvm"):
            with self.subTest(backend=backend):
                vm = self.runtime(backend=backend)
                self.assertEqual(vm.qemu_binary, "qemu-system-x86_64")
                self.assertEqual(vm.kernel_append, "console=ttyS0")

    def test_explicit_console_and_binary_remain_supported(self):
        vm = self.runtime(backend="x86_64_tcg", kernel_append="console=ttyS0 quiet",
                          qemu_binary="/operator/bin/pinned-qemu-system-x86_64")
        self.assertEqual(vm.kernel_append, "console=ttyS0 quiet")
        self.assertEqual(vm.qemu_binary, "/operator/bin/pinned-qemu-system-x86_64")

    def test_launch_topology_uses_backend_specific_block_and_serial_devices(self):
        for backend, machine, cpu, block, serial in (
            ("aarch64_hvf", "virt,accel=hvf", "host", "virtio-blk-device", "virtio-serial-device"),
            ("x86_64_tcg", "q35,accel=tcg", "max", "virtio-blk-pci", "virtio-serial-pci"),
            ("x86_64_kvm", "q35,accel=kvm", "host", "virtio-blk-pci", "virtio-serial-pci"),
        ):
            captured = []
            def capture_and_stop(args, **kwargs):
                captured.append(args)
                raise OSError("launch intentionally stopped before QEMU")
            vm = self.runtime(backend=backend, enable_action_port=True)
            vm._popen_factory = capture_and_stop
            with self.subTest(backend=backend), patch.object(vm, "_check_artifacts"), \
                    self.assertRaisesRegex(MicroVMRuntimeError, "vm_start_failed"):
                vm.start(paused=True)
            args = captured[0]
            self.assertEqual(args[args.index("-machine") + 1], machine)
            self.assertEqual(args[args.index("-cpu") + 1], cpu)
            self.assertIn(block + ",drive=work", args)
            self.assertIn(block + ",drive=read0", args)
            self.assertIn(serial + ",id=fpbvs", args)
            self.assertIn("virtserialport,chardev=fpbctl,name=fpb.control", args)
            self.assertEqual(args[args.index("-nic") + 1], "none")
            self.assertIn("-S", args)
            self.assertIsNone(vm._socket_dir)

    def test_linux_uses_short_private_tmp_control_socket_directory(self):
        vm = self.runtime(backend="x86_64_tcg")
        with patch("future_prediction_bench.microvm_runtime.sys.platform", "linux"), \
                patch.object(MicroVMRuntime, "_check_artifacts", self.fake_artifacts), \
                patch.object(MicroVMRuntime, "_connect", self.fake_connect):
            vm.start()
            try:
                self.assertEqual(vm._socket_dir.parent, Path("/tmp"))
                self.assertLess(len(str(vm._socket_dir / "serial.sock")), 100)
                self.assertEqual(vm._socket_dir.stat().st_mode & 0o777, 0o700)
            finally:
                vm.close()

    def test_x86_fork_carries_backend_and_returns_independent_children(self):
        vm = self.runtime(backend="x86_64_tcg")
        with patch.object(MicroVMRuntime, "_check_artifacts", self.fake_artifacts), \
                patch.object(MicroVMRuntime, "_connect", self.fake_connect):
            vm.start()
            try:
                snapshot = vm.save_snapshot("warm")
                self.assertEqual(snapshot["backend"], "x86_64_tcg")
                children = vm.fork_snapshot("warm", [self.root / "child.qcow2"])
                try:
                    self.assertEqual(children[0].backend, vm.backend)
                    self.assertEqual(children[0].get_state()["backend"], vm.backend)
                    self.assertIn("q35,accel=tcg", self.processes[-1].args)
                    self.assertIn("virtio-blk-pci,drive=work", self.processes[-1].args)
                    self.assertNotEqual(children[0].disk_path.stat().st_ino, vm.disk_path.stat().st_ino)
                finally:
                    for child in children:
                        child.close()
            finally:
                vm.close()

    def export_template(self, backend="x86_64_tcg"):
        vm = self.runtime(backend=backend)
        with patch.object(MicroVMRuntime, "_check_artifacts", self.fake_artifacts), \
                patch.object(MicroVMRuntime, "_connect", self.fake_connect):
            vm.start()
            try:
                return MicroVMTemplate.export(vm, self.root / "template.qcow2")
            finally:
                vm.close()

    def rewrite_manifest(self, template, change):
        manifest = json.loads(template.manifest_path.read_text())
        manifest.pop("template_id")
        change(manifest)
        template_id = hashlib.sha256(_canonical_bytes(manifest)).hexdigest()
        manifest["template_id"] = template_id
        template.manifest_path.chmod(0o600)
        template.manifest_path.write_text(json.dumps(manifest))
        return template_id

    def test_v2_template_pins_backend_and_spawn_preserves_x86_topology(self):
        template = self.export_template()
        manifest = json.loads(template.manifest_path.read_text())
        self.assertEqual(manifest["schema"], "qemu_full_vm_template_v2")
        self.assertEqual(manifest["backend"], "x86_64_tcg")
        with patch.object(MicroVMRuntime, "_check_artifacts", self.fake_artifacts), \
                patch.object(MicroVMRuntime, "_connect", self.fake_connect):
            children = template.spawn([self.root / "template-child.qcow2"], popen_factory=self.factory)
            try:
                self.assertEqual(children[0].backend, "x86_64_tcg")
                self.assertIn("q35,accel=tcg", self.processes[-1].args)
            finally:
                for child in children:
                    child.close()
        manifest["backend"] = "x86_64_kvm"
        template.manifest_path.chmod(0o600)
        template.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(MicroVMRuntimeError, "template_id_mismatch"):
            MicroVMTemplate.open(template.manifest_path, expected_template_id=template.template_id)

    def test_v2_missing_or_unknown_backend_is_rejected_with_valid_record_digest(self):
        template = self.export_template()
        for backend in (None, "unsupported", []):
            with self.subTest(backend=backend):
                expected = self.rewrite_manifest(template, lambda manifest: manifest.update(backend=backend))
                with self.assertRaisesRegex(MicroVMRuntimeError, "template_backend_invalid"):
                    MicroVMTemplate.open(template.manifest_path, expected_template_id=expected)
        expected = self.rewrite_manifest(template, lambda manifest: manifest.pop("backend"))
        with self.assertRaisesRegex(MicroVMRuntimeError, "template_backend_invalid"):
            MicroVMTemplate.open(template.manifest_path, expected_template_id=expected)

    def test_legacy_v1_arm_template_reopens_and_spawns_with_inferred_backend(self):
        template = self.export_template(backend="aarch64_hvf")
        def downgrade(manifest):
            manifest["schema"] = "qemu_hvf_full_vm_template_v1"
            manifest.pop("backend")
        expected = self.rewrite_manifest(template, downgrade)
        legacy = MicroVMTemplate.open(template.manifest_path, expected_template_id=expected)
        with patch.object(MicroVMRuntime, "_check_artifacts", self.fake_artifacts), \
                patch.object(MicroVMRuntime, "_connect", self.fake_connect):
            children = legacy.spawn([self.root / "legacy-child.qcow2"], popen_factory=self.factory)
            try:
                self.assertEqual(children[0].backend, "aarch64_hvf")
                self.assertIn("virt,accel=hvf", self.processes[-1].args)
            finally:
                for child in children:
                    child.close()

    def test_legacy_schema_cannot_sneak_in_a_new_backend_field(self):
        template = self.export_template()
        expected = self.rewrite_manifest(template,
            lambda manifest: manifest.update(schema="qemu_hvf_full_vm_template_v1"))
        with self.assertRaisesRegex(MicroVMRuntimeError, "legacy_backend_field"):
            MicroVMTemplate.open(template.manifest_path, expected_template_id=expected)


if __name__ == "__main__":
    unittest.main()
