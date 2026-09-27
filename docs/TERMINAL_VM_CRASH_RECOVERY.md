# Terminal verifier restart after a VM process crash

The [Crab paper, Section 6](https://arxiv.org/html/2604.28138) separates a sandbox crash from the agent's logical progress: a host coordinator must not expose an unfinished tool result as a completed turn. Its [manifest publication rule in Section 5.3](https://arxiv.org/html/2604.28138) likewise exposes only complete recovery points. This project applies a narrower rule to host-private grading after `submit`: a VM may be replaced only after the journal has a committed `terminal-submit.json` bound to the submitted full-VM snapshot, and no reward is returned until a validated `terminal-result.json` is durable. This terminal-recovery path is an independent design transfer, not a reproduction of upstream Crab recovery. The [official repository](https://github.com/open-agent-infra/crab) was verified on 2026-09-27; a separate [real-guest monitor and policy probe](OFFICIAL_CRAB_MICROVM.md) executes selected upstream code without establishing the full recovery backend.

[`CodingVMRecoveryJournal.resume_verification_after_vm_crash()`](../future_prediction_bench/semantic_vm_recovery.py) keeps the original host adapter and verifier, validates the action manifest, terminal attempt and submit marker, and checks the replacement VM's disk path, pinned kernel/initramfs identities, read-only disks, memory, vCPU count, and transport mode. It starts the replacement paused, verifies its read-only disk hashes, restores the `submitted` QEMU CPU/RAM/device/qcow2 state, then invokes the existing verifier retry. That verifier restores the submitted state before each hidden case. The helper never reissues `submit`. If a replacement fails to start or load, the reward gate stays closed and another matching replacement can be attempted. A present durable result is returned through the existing result validator without restarting a VM.

Focused [fault-injection tests](../tests/test_semantic_vm_recovery.py) cover a VM dying during a hidden case, case-local file state disappearing after restore, a transport that has already closed the failed VM handle, a replacement `load_snapshot` failure followed by successful retry, a mismatched replacement disk, and a crash before the submit marker was published. They assert one submit, no reward before the result record, and no extra policy turn after terminal submission. Run them with:

```bash
python3 -m unittest discover -s tests -p 'test_semantic_vm_recovery.py' -v
```

The unit tests use a fake VM. The host coordinator, adapter, private verifier, and submitted-snapshot metadata must survive; a crashed host process or power loss is outside the contract. The method currently rejects the opt-in stateless verifier because its guest helper rebinding has not been verified. It also does not provide Crab's eBPF classifier, ZFS/CRIU backend, LLM request replay, or transparent recovery of an in-flight policy action. No throughput or RL training speedup is claimed from this correctness change.

The [real-QEMU regression script](../examples/realworld_boltons26/check_terminal_vm_crash_recovery.py) compares a normal control with a QEMU process killed after the first hidden case, rejects a replacement with a mismatched kernel digest before boot, restores a matching VM from `submitted`, and compares all 14 case results, reward, action observations, and final source hash. Reproduce it on the pinned Boltons v2 assets with a fresh output directory:

```bash
python3 -m examples.realworld_boltons26.check_terminal_vm_crash_recovery \
  --task-dir runs/boltons-v2-task-static-20260925 \
  --assets-dir runs/microvm-assets-v2task-static-20260925 \
  --output runs/terminal-vm-crash-reproduction
```

## One pinned real-QEMU validation

On 2026-09-25, the control and VM-crash/replacement arms each scored **14/14 hidden cases and reward 1.0**. Their complete per-case result records, action observations, and final source SHA-256 were exactly equal. The crash arm had a durable terminal submit marker but no terminal result when QEMU was killed after the first hidden case. A replacement with the wrong kernel SHA-256 was rejected before boot. The matching replacement restored `submitted`, removed the interrupted case's guest `/tmp` marker, and completed grading with exactly one `submit` call. The run used `semantic_vm_recovery.py` SHA-256 `a4f2ee422d7b908718ce46e1317b22dd9f9a52d4ac4b56ee494566ed360510ae` and `microvm_runtime.py` SHA-256 `28e1ba54bf8973277a24f63f5f1c41974063f59252ef5d5b34d54fec1790b5be`.

The [path-free report](measurements/terminal_vm_crash_recovery_2026-09-25.json) has SHA-256 `403312f95634c34dbf0d06e75b3834546c3eb40b7325e458ff13c2363344ed54`. It omits internal terminal IDs and hidden expected outputs. This is one correctness check on a solved fixture, not a matched throughput A/B or evidence of RL training acceleration.
