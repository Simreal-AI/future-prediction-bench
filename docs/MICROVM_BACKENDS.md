# QEMU backend selection and template provenance

`MicroVMRuntime` accepts an explicit `backend`:

| Backend | Machine and accelerator | Execution evidence |
| --- | --- | --- |
| `aarch64_hvf` (default) | ARM `virt`, macOS HVF | Existing real ARM guest and graded environment experiments; [four-episode compatibility regression](measurements/official_arm_backend_graded_regression_2026-09-27.json). |
| `x86_64_tcg` | x86 `q35`, QEMU software emulation | Real pinned Crab guest inspection and full RAM/disk restoration; [one-pair direct-backend regression](measurements/official_crab_x86_direct_backend_smoke_2026-09-27.json). |
| `x86_64_kvm` | x86 `q35`, Linux KVM | Launch configuration and propagation unit tests only. No actual KVM execution or speed measurement. |

The backend derives the default executable, CPU model, serial console, and
block/serial device topology. All three retain `-nic none`; QEMU receives no
host directory mount. Linux uses a short private `/tmp` directory for Unix
control sockets. Forked children preserve the parent backend.

New template exports use `qemu_full_vm_template_v2`, binding the backend into
the template identity together with the pinned artifacts and configuration.
Old `qemu_hvf_full_vm_template_v1` manifests remain readable and infer only
`aarch64_hvf` after their original identity is checked. A legacy manifest
cannot add a backend field without adopting the new schema and identity.

The existing `MicroVMCodingAdapter`, prepared coding environment, and semantic
recovery journal still implement the ARM/HVF task contract. They reject x86
runtimes and recovery replacements before launching or relabeling artifacts.
Generic x86 runtime support therefore does not imply x86 graded-task adapter
support.

The [official Crab example](../examples/official_crab/README.md) now uses the
runtime's actual x86 backend directly. Its earlier three-pair report remains
an independent measurement of the earlier launch wrapper; the new one-pair
run is a compatibility check, not a stability benchmark. Neither TCG result
can establish native KVM performance or be pooled with ARM/HVF timings.

For native-speed process checkpoint measurements, use a disposable x86 Linux
host with compatible kernel capabilities and access to `/dev/kvm`, pinned x86
assets, and the correct backend. The separate
[CRIU process-chain experiment](../examples/official_crab_criu/README.md)
checks actual CRIU/kernel capabilities before attempting process capture.
No GPU is needed for that experiment. Model rollout and optimizer throughput
require their own matched model/GPU comparison.
