# CubeCoW: original GNU build, real XFS tests and populated branch isolation

The unchanged CubeCoW crate from Tencent's CubeSandbox v0.7.0 compiled for native Linux ARM64 GNU, and all seven original reflink tests executed successfully inside an ARM/HVF microVM on an owned XFS filesystem with `reflink=1`. Six tests require a successful real `FICLONE` probe before reaching their assertions. The seventh checks name validation. No unsupported-filesystem skip was accepted.

A separate fixture now executes the same unchanged public library on fully populated 8 and 64 MiB files. Six fresh native processes pass branch isolation, independent mutations, growth, deleted-origin recovery and cleanup. An independent Python byte oracle accepts all 92 complete-file witnesses and ten expected whole-file hash states.

The full [curated evidence](measurements/official_cubecow_xfs_tests_2026-09-27.json) preserves both successful guest runs, three earlier unsuccessful host runs and the failed musl compilation. This is execution of the original filesystem component; it does not establish a CubeSandbox VMM, memory-checkpoint, model-rollout, GPU, or training-throughput result.

## Original source and implementation

The source is pinned to [`d0081641c59822e4e5653b7462e914410b81910a`](https://github.com/TencentCloud/CubeSandbox/tree/d0081641c59822e4e5653b7462e914410b81910a), released as v0.7.0. The [standalone crate](https://github.com/TencentCloud/CubeSandbox/tree/d0081641c59822e4e5653b7462e914410b81910a/cubecow) was compiled normally, including its complete production library and original test registration. No author source, dependency, syscall, or snapshot implementation was replaced. The 34-file crate inventory and Git cleanliness agree before and after compilation.

The original [`reflink.rs`](https://github.com/TencentCloud/CubeSandbox/blob/d0081641c59822e4e5653b7462e914410b81910a/cubecow/src/engine/reflink.rs) issues Linux `FICLONE` directly. Its destination is created exclusively, and a failed clone removes the partial destination rather than copying the source. Snapshots and writable volumes share one name namespace. A snapshot of a snapshot retains the initial origin volume. Startup scanning rebuilds the namespace from disk and removes zero-length snapshot artifacts left by incomplete clones.

The source file SHA256 is `ac28eaffc545c4a7c49b4a299c8492360621c3e81d5b28a2782b09370fcbbbc0`. The original `Cargo.lock` SHA256 is `11cbe42975a8f62c7a577981dab958869b24e7688c81d30e1863feeac76f1431`. The upstream [license](https://github.com/TencentCloud/CubeSandbox/blob/d0081641c59822e4e5653b7462e914410b81910a/LICENSE) states Apache-2.0 with listed third-party exceptions.

## Native compilation and input provenance

The successful build used official Rust 1.93.1 for `aarch64-unknown-linux-gnu`, genuine GCC 12.2.0-14+deb12u1 and glibc 2.36-9+deb12u14, and 106 checksum-pinned Cargo registry archives. All 5,214 vendored files were rehashed after compilation. The build ran as an unprivileged user in an immutable cached ARM64 utility container with networking disabled, source and vendor mounts read-only, two CPUs, and a 2 GiB memory limit. It did not install host packages.

The executed original build commands were:

```text
cargo test --manifest-path /source/cubecow/Cargo.toml --offline --locked --lib --no-run
cargo build --manifest-path /source/cubecow/Cargo.toml --offline --locked --bin cubecow-cli
```

The build supplied the genuine GNU compiler/sysroot through owned environment configuration; it did not change `Cargo.toml`, `Cargo.lock`, or the author's toolchain file. Compilation produced these original executables:

| Artifact | SHA256 |
| --- | --- |
| Library test executable | `42feb13799c67da9e5e95f5f393020d1a9015086c45d76f89f07e8bf03fe617d` |
| `cubecow-cli` | `421bca5d1432ca8352ef6d018849659adef02a8164adee772b50639b32e6aaa9` |

Both hashes were checked inside the actual microVM against the compiled artifacts. The loaded GNU dynamic linker, libc, and libgcc hashes also match the native preflight. The assembly inventory includes libm, but the test executable did not list it as loaded and this guest run did not independently hash it.

The trust records distinguish three mechanisms:

- Official Rust component archives matched the publisher's SHA256 manifest. Rust distribution signature verification is not claimed.
- Debian's original cached archive keyring verified the signed `InRelease`; all 32 compiler/sysroot package archives matched SHA256 values in its bound package index.
- The 27-package Alpine v3.23 ARM64 XFS-tools closure passed real RSA/SHA1 compressed-control signature verification, recorded APKINDEX control checksums, and signed compressed-payload SHA256 checks. The Alpine signing key was obtained from the official HTTPS source. A minimal set of unchanged tools/libraries was copied into an isolated guest prefix.

The guest kernel, initramfs, and matching modloop came from the pinned Alpine v3.24 ARM64 netboot cohort. They were hash-pinned; no kernel signature verification is claimed. Actual module loading, loop backing-file inspection, and the mounted XFS filesystem are recorded separately from offline preparation.

The musl build reached the unchanged author's `FICLONE` call and failed with `E0308`: the pinned musl `libc::ioctl` request type is `i32`, while the original constant is `u64`. The original code was retained. The GNU ABI build succeeded. Earlier owned linker-configuration failures are also retained. The target-resolution graph's 64 nodes are not a count of successfully compiled dependencies.

## What the real tests establish

The accepted run used the native `aarch64_hvf` backend, 512 MiB guest RAM, Linux `6.18.52-0-virt`, genuine XFS and loop modules, and a new 512 MiB regular file inside the disposable guest. That file was checked as blank, formatted using the original `mkfs.xfs -m reflink=1`, and mounted with `loop,nosuid,nodev`. Both the format output and mounted-filesystem information confirmed `reflink=1`. `TMPDIR` pointed into this XFS mount before invoking the unchanged test executable.

| Original test | Assertions reached |
| --- | --- |
| `create_and_list_volume_roundtrip` | Volume size/path/listing and duplicate rejection |
| `snapshot_create_delete_and_listing` | Snapshot and snapshot-of-snapshot creation, origin flattening, snapshot survival after origin deletion, and final orphan-directory removal |
| `create_volume_from_volume_source` | Writable clone creation and its file surviving source deletion |
| `names_share_a_global_namespace` | Conflicts between volume and snapshot names |
| `resize_only_grows_volume_main_file` | Volume growth, unchanged snapshot size, and shrink rejection |
| `scan_recovers_volumes_and_snapshots_after_restart` | Disk namespace reconstruction and removal of a zero-byte incomplete-clone artifact |
| `validate_name_rejects_bad_inputs` | Invalid names rejected and a valid name accepted |

The whole library registered 19 tests. The explicit reflink selector ran seven, with zero failures, zero ignored tests, and zero skip markers. The 12 filtered tests are seven configuration tests and five S3 tests; they were not executed in this run.

These author tests verify real clone/snapshot operations and their metadata, namespace, size, and lifecycle behavior. They do not compare complete populated-file contents after independent branch mutations. The separate populated experiment below supplies that additional check; its assertions are not attributed to the author's original tests.

The test command's host-observed elapsed time was 13.342 ms; the Rust test runner printed a rounded 0.01 s. The full driver took 8.565 s, including boot, hash checks, formatting, mounting, test registration, execution, sync, and unmount. These are observation timings for one correctness run, without a baseline or a speedup claim. Offline compilation and image preparation are outside the driver timer.

Two earlier host records remain unsuccessful:

1. The first stopped before XFS formatting/tests because `sha256sum` was unavailable through the chosen guest chroot path. No test success is inferred from that run.
2. The second genuinely executed all seven tests, but the host gate incorrectly required zero filtered tests. Its historical `passed: false` is preserved. A fresh third run recorded the whole 19-test registry and accepted the correct seven-selected/12-filtered result.

## Populated filesystem branches: actual execution

The owned external fixture links the complete CubeCoW crate through a normal Cargo path dependency. It provides data and byte assertions; it does not implement a substitute clone, copy fallback or snapshot engine. Its six phases execute in distinct fresh native processes, each using the original public constructor and rebuilding the original namespace from disk. The accepted run uses the same actual ARM/HVF backend, kernel, XFS loop filesystem and original GNU library/runtime cohort described above. It also reruns all seven original tests without skips.

| Phase | Additional contract checked at both 8 and 64 MiB |
| --- | --- |
| Populate | Every origin byte is nonzero; an original snapshot, snapshot-of-snapshot and two writable forks match every byte of the origin; duplicate creation returns `AlreadyExists` |
| Mutate | Three disjoint 64 KiB regions change independently: fork A's prefix, fork B's midpoint and the live origin's suffix; both snapshots retain the original full contents |
| Resize | Fork A grows by 1 MiB; the extension is entirely zero and all previous contents remain exact; attempted shrink returns `InvalidArg` |
| Delete origin | The source name returns `NotFound`; its snapshots, backing directory and mutated branches remain correct |
| Recover orphan | A fresh original constructor recovers both canonical snapshot names; a new fork from the orphan snapshot contains the complete original bytes; deleting the final snapshot reaps the origin directory |
| Cleanup | Surviving branches still have their expected complete contents, then all branch names and the owned namespace are removed |

The deleted origin's `list_snapshots()` result is empty by the original API contract. Recovery is demonstrated by canonical snapshot lookup and a genuine new writable branch, rather than changing that contract. All phase-reported PIDs match the actually spawned child PIDs and are distinct from the master and one another. Full backing paths, logical lengths, filesystem device and surviving inode identities remain checked.

The runtime compares every file byte and also hashes every complete file using the genuine, hash-pinned GNU `sha256sum`. A separately executed Python verifier reconstructs the deterministic byte formula, three mutation patterns and zero extension without importing or running the Rust fixture. It validates the raw phase outputs, exact six-phase/two-size matrix, all 92 witnesses and all ten expected SHA-256 states. The sum of repeated complete-file comparisons is **3,485,466,624 bytes**; this is not unique data size or a throughput measurement.

Owned writers call `File::sync_all()` before byte checks. The unchanged upstream `FICLONE` operation does not fsync the destination. The experiment checks sequential branch isolation and restart namespace recovery; it does not establish concurrent-writer atomicity, crash/power-loss durability or VM/process-memory recovery.

The populated fixture's measured source SHA-256 is `9512f947ff915b4ad234bf238780b418f30311a8a471d2eb98c084116da60574`, and its normally linked native executable is `2bdf7e7c65bd03a372f79688df47ad99090ea73b9a951fa957761801f52ffeb3`. The frozen input manifest is `9c4aa6ecec9c915fae1cf0bb82f43a3d7bd90db7f7c0c1ec633133212a2e115f`. Its build retains all 106 original registry archive checksums; all 5,214 vendor files are compared directly to the original archive members. The newly declared vendor digest serialization is explicit and does not overwrite the earlier record's different serialization.

The actual runtime report SHA-256 is `41583f9813d03996598347dff9b0f0630260f1a4c782f742ed0d308e1c6ed130`; the full host record is `ad581f791e3d3d125885d626572baa092feb43708accc2c4e772db7087d87dfe`. The independent verifier source is `258cc236b9a611b58d5b6f7bd68e490c966df68843e2669806ff55e7bad7a6a6`, and its acceptance record is `82ec9a8dbb1f7055f282b029d0bafd35b4e75c7b9420d985f7c6f39a47420a9a`. The public measurement retains these full parsed records and unchanged original source/artifact pins.

The first populated guest attempt stopped before XFS formatting and tests because an external prerequisite check requested `/bin/sh`, absent from the original guest root. That failure remains visible. A fresh second attempt changed only the external prerequisite check to use the available native `/bin/ls`; the original library and populated executable bytes were unchanged. No snapshot, training or model-rollout speed claim is made from these observations.

## Reproduce the original component tests

Use an owned Linux ARM64 GNU environment with Rust 1.93.1 and a GNU linker/runtime already available. Provide an existing owned XFS filesystem with reflink enabled; this recipe does not format or mount a disk. Keep Cargo output outside the author checkout. Substitute fresh paths below.

```bash
set -euo pipefail
CUBE_REPRO_ROOT=/path/to/new-cubecow-reproduction
CUBE_XFS_TMPDIR=/path/to/existing-owned-xfs/test-temp
test ! -e "$CUBE_REPRO_ROOT"
mkdir "$CUBE_REPRO_ROOT"
mkdir -p "$CUBE_XFS_TMPDIR"
test "$(findmnt -n -o FSTYPE --target "$CUBE_XFS_TMPDIR")" = xfs
xfs_info "$CUBE_XFS_TMPDIR" > "$CUBE_REPRO_ROOT/xfs-info.txt"
rg 'reflink=1' "$CUBE_REPRO_ROOT/xfs-info.txt"
git clone https://github.com/TencentCloud/CubeSandbox.git "$CUBE_REPRO_ROOT/source"
git -C "$CUBE_REPRO_ROOT/source" checkout --detach d0081641c59822e4e5653b7462e914410b81910a
test -z "$(git -C "$CUBE_REPRO_ROOT/source" status --porcelain --untracked-files=all)"
export CARGO_TARGET_DIR="$CUBE_REPRO_ROOT/target"
cd "$CUBE_REPRO_ROOT/source/cubecow"
cargo +1.93.1 fetch --locked
cargo +1.93.1 test --offline --locked --lib --no-run
cargo +1.93.1 build --offline --locked --bin cubecow-cli
TMPDIR="$CUBE_XFS_TMPDIR" cargo +1.93.1 test --offline --locked --lib -- --list \
  > "$CUBE_REPRO_ROOT/registered-tests.txt" 2>&1
TMPDIR="$CUBE_XFS_TMPDIR" cargo +1.93.1 test --offline --locked --lib engine::reflink::tests \
  -- --test-threads=1 --nocapture > "$CUBE_REPRO_ROOT/reflink-tests.txt" 2>&1
cat "$CUBE_REPRO_ROOT/reflink-tests.txt"
test -z "$(git -C "$CUBE_REPRO_ROOT/source" status --porcelain --untracked-files=all)"
python3 - "$CUBE_REPRO_ROOT" <<'PY'
from pathlib import Path
import re, sys
root = Path(sys.argv[1])
listing = (root / "registered-tests.txt").read_text()
text = (root / "reflink-tests.txt").read_text()
names = {
    "create_and_list_volume_roundtrip", "create_volume_from_volume_source",
    "names_share_a_global_namespace", "resize_only_grows_volume_main_file",
    "scan_recovers_volumes_and_snapshots_after_restart",
    "snapshot_create_delete_and_listing", "validate_name_rejects_bad_inputs",
}
expected = {"engine::reflink::tests::" + name for name in names}
registered = re.findall(r"^(.+): test$", listing, re.M)
executed = re.findall(r"^test (engine::reflink::tests::[a-z_]+) \.\.\. ok$", text, re.M)
if len(registered) != 19 or len(set(registered)) != 19 or not expected <= set(registered):
    raise SystemExit("unexpected original test registry")
if len(executed) != 7 or set(executed) != expected:
    raise SystemExit("not all seven original reflink tests executed")
if re.search(r"\[skip\]|skipping|does not support FICLONE", text, re.I):
    raise SystemExit("unsupported filesystem: a silent author skip is not a pass")
if "7 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out" not in text:
    raise SystemExit("unexpected test result")
print("Seven original reflink tests executed without skips.")
PY
```

A caller using a preverified vendor directory can omit the network fetch and configure Cargo's vendored source in an owned Cargo home. This recipe reproduces the original author test suite; it does not execute the separate populated fixture. The byte-identical populated fixture source and lockfile, an offline builder and independent report verifier are now included in [the public reproduction example](../examples/official_cubecow/README.md). Its builder has a separate execution record; earlier guest outcomes remain bound to the original frozen build. Upstream source, compiler/runtime distributions and executable binaries remain outside the public archive. The recipe is not a claim of identical binary hashes on a different compiler/runtime environment. The measured source, compiler inputs, guest assets, binaries, full commands, and prior outcomes are bound in the adjacent evidence record.

## Executed public reproduction entrypoint

The promoted [offline builder and independent verifier](../examples/official_cubecow/README.md) now run on the exact original inputs. Both normal Cargo commands complete successfully, preserving all 34 author files, 106 archive checksums and 5,214 original vendor members. The public builder generates a byte-identical populated executable (`2bdf7e7c65bd03a372f79688df47ad99090ea73b9a951fa957761801f52ffeb3`). A fresh native ARM/HVF replay dispatches that actual public-build output and passes the seven original tests, all six new child-process phases and 92 whole-file witnesses at both sizes. The public Python verifier accepts that newly exported raw result; twelve focused controls pass with no skips. The [full curated reproduction record](measurements/official_cubecow_public_reproduction_2026-09-27.json) preserves build, source/vendor, genuine Git prerequisites, guest and verification evidence. This is a correctness reproduction; its complete boot/setup/validation elapsed time is outside the separate primitive latency comparison.

A subsequent deeper GNU sysroot payload guard found a host case-collision candidate between `xt_CONNMARK.h` and `xt_connmark.h`. Existing compiler-tree before/after equality and actual ELF/runtime/test records remain preserved; they do not prove that every relocated package header was reconstructed exactly. The next cohort rebuilds and verifies package members on a case-sensitive Linux filesystem. See the [latency review](CUBE_COW_LATENCY.md) for the retained input-integrity limit.
