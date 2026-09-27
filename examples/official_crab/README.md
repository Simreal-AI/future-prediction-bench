# Official Crab inspection and policy probe

This example executes pinned upstream Crab code against actual Linux RAM in
an x86-64 QEMU/TCG guest. It checks one stopped worker and one file, and uses
full QEMU snapshots for every capture. It does not run Crab's CRIU/ZFS/eBPF
backend, an LLM agent, or a software-task grader. See the
[measurement and scope](../../docs/OFFICIAL_CRAB_MICROVM.md).

## Prepare pinned inputs

Obtain a clean checkout; upstream source stays outside the release:

```bash
git clone https://github.com/open-agent-infra/crab.git runs/official-crab
git -C runs/official-crab checkout 9607d61a41dc44358cf078c4b438bfd971c8ee9d
```

Prepare the pinned Boltons v2 task using its
[runbook](../realworld_boltons26/README.md). The task seed is reused only as
guest asset input; this probe does not grade the Boltons repair.

Download the exact operator-managed inputs listed in
[`prepare_x86.py`](prepare_x86.py) into a new input directory: Alpine v3.24
x86-64 netboot files (`vmlinuz-virt`, `initramfs-virt`, `modloop-virt`), the
pinned standalone CPython archive named `python-musl.tar.gz`, and the pinned
musl package named `musl.apk`. The script defines source URLs and SHA-256
checks. It never downloads or pulls an image. With local QEMU utilities and
a cached **ARM64 Linux image containing `mke2fs`**, run from the project root:

```bash
python3 -m examples.official_crab.prepare_x86 \
  --source runs/crab-x86-inputs \
  --task-dir runs/boltons-v2-task-static-20260925 \
  --output runs/crab-x86-assets \
  --build-image sha256:<cached-build-image-id>
```

Docker is an offline image-building utility here. The measured guest runs
through `qemu-system-x86_64` with TCG, not Docker or hardware virtualization.
All input hashes are checked, and output directories must be new and
disjoint from inputs.

## Run actual capture and restore checks

```bash
python3 -m examples.official_crab.check_microvm \
  --checkout runs/official-crab \
  --task-dir runs/boltons-v2-task-static-20260925 \
  --assets runs/crab-x86-assets \
  --output runs/crab-x86-packed-reproduction \
  --x86-tcg --repetitions 3
```

The default packs action, inspection, state response, and commit-boundary
baseline work into one guest RPC. Both arms use that transport. Add
`--legacy-rpc` with a different output directory to test the slower separate
RPC path. The known-write soft-dirty calibration must pass before any
selective action; a failure writes `capability_failure.json` and aborts.

`measurement.json` records upstream source hashes, actual decisions,
capture counts, wall times, RAM/disk restoration, and killed-worker recovery.
The disposable VM disk is removed after a successful run. Timing excludes
initial setup and the subsequent correctness challenges; the published
report reports these boundaries explicitly. Source/artifact checkouts,
guest disks, and runtime logs under `runs/` are not release contents.
