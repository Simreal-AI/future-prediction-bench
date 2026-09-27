# Resident guest candidate (experimental public example)

This directory contains a task-parametric, experimental resident guest adapter for the `RealWorldEnv` interface. It runs inside one already booted ARM64 QEMU VM. A trusted guest supervisor creates an episode child in fresh mount and PID namespaces with a private overlay; each verifier case gets another nested namespace and overlay. The verifier's expected outputs and reward calculation remain on the host. The policy has only `read_file`, exact-SHA `replace_text`, and `submit`.

The operator supplies a pinned task, verifier, explicit stateless-case contract, and VM assets. `candidate_api.py --check-assets` checks their hashes before a run. `connect_booted_vm` checks the outer source and installed helper hashes and binds the guest's declared source path, seed digest, and exact case count. The guest supervisor requires `--source-path`, `--seed-sha256`, and `--case-count` at launch. These checks **do not attest the boot chain or a hostile guest**. The caller must boot the clone of the checked assets; guest root and the serial-shell operator are trusted. Installing the helper files changes the child VM's root filesystem, so "pristine" means the task workspace tree matches its pinned seed digest, not that the entire child qcow2 is byte-for-byte unchanged. The narrower task contract excludes live background process state and case-to-case filesystem dependence. This is not a full-VM checkpoint/restore implementation and is not equivalent to DeltaBox or Crab.

The guest enforces the frozen case count on sequential calls, batch calls, and completed close. The host verifies the same count at hello and before attaching the adapter. If the batch request exceeds its 2 KiB frame or any expected stdout exceeds the 64-byte batch cap, the adapter uses sequential case transport. Candidate stdout above the batch cap is a definite failed case when the expected output fits; it is not classified as an infrastructure failure. Case and batch IPC deadlines outlive the 10-second per-case execution cap, so an ordinary candidate timeout remains a scored case rather than becoming infrastructure-pending. An unreadable submit source, changed trusted file, bad guest frame, failed case process, or failed cleanup prevents a reward. Cleanup must complete before a score is published.

Before each unprivileged verifier process starts, the nested case runner applies inherited hard Linux resource limits: 8 CPU seconds, 128 MiB virtual address space per process, 2 processes for the candidate UID, 8 MiB per file, 64 open file descriptors, and zero core dump bytes. An existing stricter operator limit remains in force. The outer case guard also allows at most 10 wall seconds and 12,000 stdout bytes; the episode overlay has a 64 MiB tmpfs upper. A candidate that hits a limit and exits or receives a signal yields a failed case, while failure to launch the runner or clean its namespace withholds reward. `RLIMIT_AS` and `RLIMIT_NPROC` are process and UID limits, not a cgroup memory budget. This example assumes one case at a time and trusted guest root; it does not contain a seccomp profile or a multi-tenant isolation claim.

## Reproduce the pinned Boltons smoke

From the project root, run the offline tests:

```sh
python3 -m pytest -q tests/test_resident_guest_candidate.py
```

The pinned QEMU smoke requires local Linux/ARM64 assets, the Boltons v2 fixture, QEMU/HVF, and permission to create local Unix control sockets. Build the v1 guest and then the v2 task/assets using the [microVM setup](../../docs/MICROVM_ENV.md) and [v2 rebinding](../../docs/SMALL_EDIT_MICROVM.md) instructions. Generated inputs remain under ignored `runs/`; this release contains neither guest binaries nor the Boltons source archive. The runtime uses one VM with 512 MiB RAM, one vCPU, no NIC, and an ext4 qcow2 work disk. Run each mode into a new output directory:

```sh
PYTHONPATH=.:examples/resident_guest_candidate python3 examples/resident_guest_candidate/smoke_qemu.py \
  --task-dir runs/boltons-v2-task \
  --assets-dir runs/boltons-v2-assets \
  --contract examples/resident_guest_candidate/boltons_contract_local.json \
  --output runs/resident-guest-batch-repro \
  --case-transport batch
```

Use `--case-transport sequential` and a different `--output` for the other transport. The script clones the pinned qcow2, boots one VM, installs hashed helpers, runs one scripted repair and one unchanged baseline, checks 14 host-only cases per episode, and then runs a 14-case isolation probe. The probe writes a marker and prints 65 bytes in one case; it requires all later cases and the outer workspace to lack the marker and checks the batch output-cap signal against the sequential return. A temporary host-only verifier then runs a repaired episode with one candidate that closes stdout and sleeps past its 10-second cap; it must finish as **graded reward 0, 13/14 passes**. A second temporary verifier checks the six resource limits inside the guest before writing beyond the 8 MiB file cap; it requires `EFBIG` at exactly 8 MiB and must also finish as **graded reward 0, 13/14 passes** while later cases and the outer workspace remain intact. The temporary verifiers are deleted, and no expected output enters the guest. The child qcow2 is deleted on exit. The report contains aggregate counts, timings, and public binding digests; it contains no expected outputs or per-case stdout hashes.

## Current evidence and scope

Two independent final-code smoke runs on 2026-09-25 passed. Each used a newly cloned disk and one VM, with two graded warm episodes in that VM. Both modes scored repair **14/14, reward 1.0** and unchanged baseline **7/14, reward 0.0**. The cross-case write/read, timeout, and resource-limit probes passed in both modes, and the entire outer task workspace tree matched the pinned seed digest after every probe. Exact reports and SHA-256:

| Mode | Report SHA-256 | Clone + boot | Install + connect | Repair episode | Baseline episode |
| --- | --- | ---: | ---: | ---: | ---: |
| Batch | `b6135a8e99caff3c4569c2e47190bbdf373828e5546b2d331db4ffbc140d2cb3` | 5.720 s | 0.908 s | 0.914 s | 0.737 s |
| Sequential | `2165c57ff5906c2c96527a015ba1843f4fa9b92ef7a7967e26422c9c30044ea6` | 5.679 s | 0.845 s | 1.251 s | 1.239 s |

[Batch report](../../docs/measurements/resident_guest_batch_v0.9.0.json) and [sequential report](../../docs/measurements/resident_guest_sequential_v0.9.0.json) contain the sanitized observations. Batch makes one host transport request for 14 cases; sequential makes 14. These are **single-run smoke measurements**, not repeated randomized A/B throughput estimates. Clone/boot and helper installation are separate one-time costs and are excluded from the episode figures. The first episode branch reset measured 10.85 ms (batch run) and 8.70 ms (sequential run); the second reset measured 2.45 ms and 1.81 ms. Earlier same-code smoke runs varied materially with host load, so these numbers are not stable latency distributions. All are shared-kernel namespace/overlay branch resets, not VM memory/disk checkpoint or restore. No model inference, optimizer step, or RL weight update was measured.

The real-VM proof currently covers this pinned, already-public Boltons fixture and trusted guest root. It does not establish arbitrary repository task support, hostile guest isolation, or general reward parity with the full-VM adapter. A larger alternating-pair benchmark and fault-injection run are needed before comparing throughput beyond this smoke.
