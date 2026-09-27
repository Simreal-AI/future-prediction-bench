# Non-root candidate Python in full-VM grading

The full-VM coding adapter now starts candidate-visible Python and every
host-private Python case as guest UID/GID 65534. The trusted serial wrapper
and QEMU monitor remain outside that candidate process. Its frozen task
artifact binding includes `candidate_python_identity=guest_uid_gid_65534_v2`,
so an old task binding cannot silently be replayed under the new execution
identity. Historical timing reports using guest-root Python describe their
earlier contract; they are not republished as measurements of this revision.

The wrapper also treats stdout EOF as a stream event, not proof that the
candidate process exited. If the child closed stdout and remains alive, the
same overall case deadline applies. At timeout the wrapper kills its process
group, reaps it, and returns candidate code 124. A local real-subprocess
regression covers both a normal exit with stdout closed and a hang with
stdout closed.

## Real guest check

Run the three scripted arms on pinned local Boltons v2 assets:

```bash
python3 -m examples.realworld_boltons26.probe_guest_identity \
  --task-dir runs/boltons-v2-task-static-20260925 \
  --assets-dir runs/microvm-assets-v2task-static-20260925 \
  --output runs/guest-identity-probe \
  --public-output docs/measurements/guest_uid_reward_integrity_2026-09-25.json
```

The [sanitized report](measurements/guest_uid_reward_integrity_2026-09-25.json)
records three **complete, offline QEMU/HVF graded episodes**, each with 14
full-state restores and host-only expected outputs. The untouched baseline
received reward 0 with 7/14 cases, the scripted public repair reward 1 with
14/14, and a candidate module that attempted to kill guest PID 1 on import
received a **resolved reward 0 with 0/14 cases**, rather than a pending
infrastructure result. All 14 attack cases returned code 1, and the VM
restored for every case.

This is one adversarial probe on one solved public repository, not a general
containment proof. Guest UID 65534 still shares the guest kernel; task source
and verifier inputs remain operator-trusted. Other QEMU or guest failures
must remain pending rather than being mislabeled as policy failures. The
separate virtio action port has its own QEMU-side disconnect requirement
before any submitted snapshot.
