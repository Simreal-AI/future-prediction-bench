# Cached validation for QEMU full-state restore

`MicroVMRuntime.load_snapshot` still calls QEMU `loadvm` on every restore. It
restores the CPU, RAM, devices, and writable qcow2 disk; this change only
removes two redundant checks on the common in-process path. A read-only raw
guest disk is SHA-256 checked when the VM starts. Subsequent restores compare
its device, inode, size, mode, nanosecond modification time, and change time.
The code also omits `info snapshots` for a tag saved by the same live runtime.
An inherited tag in a cloned child still needs the QEMU listing, and QEMU's
`loadvm` remains authoritative. `full_validation=True` rechecks the disk hash
and snapshot listing for audits.

This cache assumes a trusted host/operator. Hostile privileged mutation that
forges filesystem identity metadata is outside its boundary. A changed file
identity fails closed; even a same-size rewrite followed by restoration of its
modification time changes the inode's change time in the tested host setup.

The [public reproducer](../examples/realworld_boltons26/benchmark_qemu_restore_validation.py)
uses one pinned Boltons v2 task and ARM64 QEMU/HVF asset set. It clones a
writable qcow2 and boots a 128 MiB VM with no NIC. Before each timed restore,
it changes an ext4 file, a tmpfs file, and a live shell process's in-memory
variable, then synchronizes the guest filesystem. Strict and cached calls are
alternated AB/BA in 40 measured pairs after two warm-up pairs. Every `loadvm`
must restore all three states exactly. Mutations and post-restore checks are
outside the measured call; cold boot, model actions, grading, and optimizer
updates are outside it too.

```sh
python3 -m pytest -q tests/test_microvm_runtime.py
PYTHONPATH=. python3 examples/realworld_boltons26/benchmark_qemu_restore_validation.py \
  --task-dir runs/boltons-v2-task --assets-dir runs/boltons-v2-assets \
  --output-dir runs/qemu-restore-validation --pairs 40
```

The [path-free raw report](measurements/qemu_restore_validation_v0.9.0.json)
(SHA-256 `a2796e0d77dfb44937218e2dd4f08b15a244102366e741e132791e1a8f39727f`)
records all 80 correct restores. On one Apple M3/8 GB host, whole-call p50 was
**124.11 ms strict / 87.20 ms cached** and p95 was **450.56 / 223.81 ms**.
The validation portion alone had p50 **13.28 / 0.17 ms**. The cache won
27/40 matched pairs, with a 21.26 ms median paired difference. QEMU's own
`loadvm` p50 still measured 107.24 / 87.13 ms and dominated both calls.

An earlier 20-pair run without explicit guest `sync` had strict/cached
whole-call p50 **67.66 / 55.47 ms**, but p95 **95.74 / 177.69 ms**: its
cached tail was worse. The two runs consistently show reduced validation
work and correct state restoration, but do not establish a stable full-VM
restore or graded-episode speedup. These operations are different from
[DeltaBox v2](https://arxiv.org/html/2605.22781v2), which couples a modified
guest filesystem with process checkpoint/template machinery. The separate
cooperative guest checkpoint example tests a narrower process-level contract.
