# Original Waypoint image-store CPU fixture

This example imports the complete, unmodified author package at
[`Alex-XJK/waypoint`](https://github.com/Alex-XJK/waypoint), commit
`dcb6a7b0f4114f4732b9d7f1d1824dc5e4ff4a24`. It exercises the exported
`NewManagerWithSession` and `FlushCheckpointImages` APIs on owned byte fixtures.
The fixtures are **not CRIU images**. No process checkpoint, agent rollout,
model inference, or training runs.

The component copies checkpoint files from a real Linux tmpfs to a separate
filesystem, calls the author's actual fsync implementation, and publishes the
new canonical path through an atomic symlink rename. An actual shared flock
prevents the author's exclusive handoff while readers use the old images.

The external harness checks complete fixture bytes through two concurrent
readers, case-distinct filenames, old/new path validity, removal of the tmpfs
copy after publication, an idempotent second flush, and recovery after a real
permission error. The author implementation can leave a partial disk directory
on copy failure; the harness verifies that retry replaces it while preserving
the original canonical target and source bytes. It removes its own temporary
directories on completion.

The genuine execution passed both cases on 2026-09-27. See the
[sanitized evidence](../../docs/measurements/official_waypoint_imagestore_cpu_2026-09-27.json)
for the executed source/helper pins, full fixture hashes, observed failure and
retry, and the retained initial socket-path-guard failure.

## Prepare without executing

From the project root:

```sh
git clone --depth 1 https://github.com/Alex-XJK/waypoint.git runs/official-waypoint-source
git -C runs/official-waypoint-source fetch --depth 1 origin dcb6a7b0f4114f4732b9d7f1d1824dc5e4ff4a24
git -C runs/official-waypoint-source checkout --detach dcb6a7b0f4114f4732b9d7f1d1824dc5e4ff4a24
python3 examples/official_waypoint/check_imagestore.py \
  --source runs/official-waypoint-source \
  --cache runs/waypoint-public-inputs \
  --output runs/waypoint-images-prepared --prepare-only
```

Preparation checks the clean author source, licenses, all 29 Go files, the
official Go 1.27.1 Linux ARM64 distribution checksum, and fixed public module
inputs. It downloads only to the specified cache and does not launch Go or
Docker, extract the toolchain, or install anything globally.

Use a new, nonexistent output directory for each invocation. Source, output
and cache paths must be separate with no nesting. Writable paths and cached
targets reject symlinks. Existing fixed cache inputs are verified and reused;
downloads use HTTPS-only redirects with curl configuration disabled, an
exclusive owned temporary file, and publication that cannot overwrite a
cached target. Failed downloads remove the temporary file.

## Execute the genuine package

Docker must already contain an operator-selected Linux ARM64 utility image
with `/bin/sh` and a writable `/tmp`. Pass its complete cached image digest;
the script never pulls an image. Python 3.12+, Git, curl and Docker are required.

```sh
python3 examples/official_waypoint/check_imagestore.py \
  --source runs/official-waypoint-source \
  --cache runs/waypoint-public-inputs \
  --output runs/waypoint-images-result \
  --docker-image sha256:<cached-linux-arm64-image-id>
```

The disposable container runs as UID 65534 with no capabilities, no network,
and one CPU. Source, external module, toolchain and local module proxy mounts
are read-only. Go imports the author's package normally; its Linux files and
dependencies are compiled together. There are no extracted function copies,
module doubles, source patches or trainer substitutes. The package's namespace
and CRIU code is compiled, but those runtime operations are not invoked.

The original `go.sum` verifies the module content hashes; `go mod verify` runs
after compilation. `GOTOOLCHAIN=local` prevents automatic toolchain downloads.
The result records the executed helper hashes, source checks before and after,
toolchain, container image, runtime build information and fixture evidence.

## Measurement boundary

The deliberate 40 ms shared-lease hold is a correctness barrier. Its wall time
is a functional test duration, **not a checkpoint latency or speedup**. The
build-and-fixture duration includes cold compilation and container startup.
The fixture does not test recovery after reboot or power loss.

This example validates the image-store component only. Waypoint's current
CRIU path uses full dumps; applying the same handoff to an incremental CRIU
parent chain requires additional dependency-lifetime and actual restore tests.
An image still on tmpfs is available for live restore but is not yet durable.
The original flusher skips symlinks and directories, so a parent link must
not be silently dropped when adapting this method. This experiment uses one
flusher per checkpoint and does not validate concurrent writers, the detached
flusher subprocess, or the original unit-test suite.

The author source is Apache-2.0 with a NOTICE file. Preserve both when
redistributing it. See [the source review](../../docs/WAYPOINT_SOURCE_REVIEW.md).
