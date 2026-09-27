# Waypoint: original image-store execution review

## Verified author provenance

Columbia DAPLab's [Waypoint project page](https://daplab.cs.columbia.edu/projects/waypoint/)
links the author-maintained [Alex-XJK/waypoint repository](https://github.com/Alex-XJK/waypoint).
The project is related to [Toward Systems Foundations for Agentic Exploration](https://arxiv.org/abs/2510.05556).
This review pins commit `dcb6a7b0f4114f4732b9d7f1d1824dc5e4ff4a24`, dated
2026-09-05. The license is Apache-2.0; the checkout also includes NOTICE.

All 29 `.go` files are checked using a SHA-256 over sorted repository-relative
paths followed by NUL, original file bytes, and NUL. The digest is
`94c8943f9e0ae8631aa5c08b2df3951974e92d9930513c9773e41fbc332f59ac`.

| Original file | SHA-256 |
| --- | --- |
| `pkg/waypoint/imagestore.go` | `7da668ecaf5c22e5d9e293547aa95601264cdf7a881ba8dc86dd306bc3ce3300` |
| `pkg/waypoint/checkpoint.go` | `03cecc6fab5aaafff9369dafccb53373beabb14f4dfa574d2a349061b8939a01` |
| `pkg/waypoint/criu.go` | `3290638133b35f85a5e1b481ea984f74413244e0e680ea330139aeb4badc6e56` |
| `go.mod` | `96c01ebb7fba1b9bc243ba028af4aa55754425c0d1a24cebbdc28530e49dee64` |
| `go.sum` | `73ad92b4a3264581a00aee3dd7dee6dee3f64e014fe11d278e130e2303a05209` |
| `LICENSE` | `cddf00b77e2d22068bde8e501d95bca1e3de615e0b4e905298e5752b04deea11` |
| `NOTICE` | `098349e723ae12a0c1babfcf87b7cefa222b75b9dea863d1c92ea3cb2e44f63a` |

## Transferable implemented mechanism

The actual [images store](https://github.com/Alex-XJK/waypoint/blob/dcb6a7b0f4114f4732b9d7f1d1824dc5e4ff4a24/pkg/waypoint/imagestore.go)
supports a canonical `criu` symlink initially pointing at tmpfs. The author
flusher copies regular files into `criu.disk`, fsyncs files and the directory,
takes an exclusive `images.lock`, atomically renames a replacement symlink,
fsyncs the checkpoint directory and removes the tmpfs copy. The actual
[restore path](https://github.com/Alex-XJK/waypoint/blob/dcb6a7b0f4114f4732b9d7f1d1824dc5e4ff4a24/pkg/waypoint/criu.go)
takes a shared lock during CRIU restore so the image location cannot disappear
mid-read. The source documents the period before flushing completes as
vulnerable to host reboot.

This can separate the frozen process window from disk persistence. A product
interface should distinguish `volatile_ready` and `durable_ready`, acquire
reader leases while restoring, and preserve image-parent dependencies. This
is an integration proposal; this fixture does not modify the product runtime.
Waypoint currently performs full CRIU dumps. Its image-store mechanism alone
does not establish safe migration of an incremental CRIU parent chain.
The original flusher copies only top-level regular files and skips symlinks
and directories, including a CRIU parent symlink. The exclusive lock covers
publication, not the preceding copy phase; keep one flusher per checkpoint
unless concurrent writers are separately verified. Neither limitation is
changed or hidden in this example.

## Faithful CPU fixture execution

The [external example](../examples/official_waypoint/README.md) normally imports
the full original `github.com/Alex-XJK/waypoint/pkg/waypoint` package and invokes
its exported `NewManagerWithSession` and `FlushCheckpointImages` methods.
No author source is edited. Fixture image files contain arbitrary bounded
bytes and are explicitly not a saved process.
The package's 12 production Go files compile normally together. All 29 Go
files in the repository, including original tests and CLI files, are pinned;
the original unit tests and CLI runtime are not executed by this fixture.

The original module requires Go 1.25.0, `github.com/creack/pty` v1.1.24 and
`golang.org/x/sys` v0.42.0. The package uses Linux-only syscall fields without
platform build tags, so an unchanged Darwin build is not the execution target.
We use the official [Go distribution](https://go.dev/dl/) and
[installation documentation](https://go.dev/doc/install) to pin Go 1.27.1
Linux ARM64, compatible with the declared Go version. The archive is
67,009,954 bytes with SHA-256
`3450b45a3f9ee8568792736a5c5e70a1f2e9b36c35a8f74958c03e51d7d92bec`.
It is isolated in task storage; nothing is installed globally.

Compilation uses real modules, the untouched author `go.sum`, an offline local
module proxy and `go mod verify`. A disposable Linux ARM64 container runs
the fixture as non-root. It checks real tmpfs source staging, a distinct disk
filesystem, exact full fixture bytes, concurrent flock-protected readers,
atomic publication, post-flush source removal, idempotency, and real permission
failure followed by retry. Filesystem sync calls are the author's implementation.

## Observed result on 2026-09-27

The genuine fixture passed both external cases. It ran as UID 65534 under
Go 1.27.1 on Linux ARM64, with actual tmpfs source storage (`0x1021994`) and
overlayfs disk storage (`0x794c7630`). `go mod verify` passed, and the clean
author source and full source hashes matched before and after execution.

Each case uses 262,185 fixture bytes across three regular files, including
case-distinct `pages-fixture.img` and `PAGES-fixture.img`. In the final replay,
two shared-lock readers completed 43 old-target exact-byte reads during
publication with no reader errors. Eight separate post-handoff checks read
the new target. This replay observed no concurrent new-target read; the
retained earlier passing trial observed one. These counts are not all concurrent.
The author's exclusive publication waited for the real shared lease. The
tmpfs copy was removed only after publication, and a second flush was a no-op.
A real unreadable source file produced `EACCES`; the old canonical target and
source bytes remained intact. A partial disk directory was observed, and retry
replaced it and preserved all fixture bytes.

The final atomic case took 51.949 ms **including a deliberate 40 ms reader-lease
hold**. This is a correctness experiment, not a checkpoint speed result.
Cold compilation, container startup and the fixture together took 13.734 s.
The detached flusher subprocess and actual CRIU restore are not exercised.
No agent rollout, GPU test, trainer or power-loss recovery is claimed.

The first attempt compiled and reached the original manager, then failed its
real Unix socket-path guard (110 characters exceeded 107). Only the external
harness's owned temporary path was shortened before the passing run; the
author source was unchanged. That failed attempt is retained in the
[public evidence](measurements/official_waypoint_imagestore_cpu_2026-09-27.json).
The final passing checker SHA is
`b0cd53439aafd7366227cf7aa1bcdb9a79d37d3a38103b7a38567b6a2f7115bc`,
and the passing harness SHA is
`be0064320b0cbf648c50637d8482442409f34cb357d18aced7e7474a953d2346`.

The final replay also binds the hardened preparation helper. Twenty-two
low-cost guards verified overlap rejection, fresh output paths, symlink
rejection, cache reuse, HTTPS-only curl arguments, exclusive descriptor
downloads, failure cleanup and no-overwrite publication. Those downloader
guards use controlled local subprocess output; they do not replace any author
runtime code. The subsequent genuine package replay passed both cases with
the final helper hash. Prior execution and preparation failures remain labeled
separately in the evidence.

## Related work with different status

[REACH](https://arxiv.org/html/2609.19636v1) proposes matching environment state,
history, remaining budget and seeds when switching model checkpoints. Its
paper links a referenced Agent-STAR implementation, but this review did not
verify an author repository implementing REACH's handoff protocol. It remains
a useful evaluation design, not a reproduced OS checkpoint implementation.

[Orchard's official README](https://github.com/microsoft/Orchard) currently lists
full-state pause/resume and branching under its roadmap. Those statements
must not be presented as completed checkpoint code.
