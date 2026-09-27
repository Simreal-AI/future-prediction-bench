# Original CubeCoW: reproducible populated filesystem experiment

This example normally links the complete, unchanged CubeCoW library from
[CubeSandbox v0.7.0](https://github.com/TencentCloud/CubeSandbox/tree/d0081641c59822e4e5653b7462e914410b81910a/cubecow),
commit `d0081641c59822e4e5653b7462e914410b81910a`. It executes the original
public constructor and snapshot/volume APIs, including real Linux `FICLONE`.
It supplies no replacement engine or full-copy fallback.

`src/main.rs` and `Cargo.lock` are byte-identical to the populated fixture
that passed on real XFS in an ARM/HVF microVM. They have SHA-256 values
`9512f947ff915b4ad234bf238780b418f30311a8a471d2eb98c084116da60574` and
`1bc25eab0f64a3e71c3c781a0250733e5c9d12d5105e0d58cc09ae49b75b50dd`, respectively.
The [source review](../../docs/CUBE_COW_SOURCE_REVIEW.md) and
[full measurement](../../docs/measurements/official_cubecow_xfs_tests_2026-09-27.json)
retain the actual execution, independent oracle, original seven-test suite,
and earlier failed attempts. The external upstream repository, vendor files,
compiler distribution, guest disks and executable binaries are not packaged.

The public wrapper is a new reproduction entrypoint. Its input preparation
and actual-report verifier can be checked without a guest. An actual build
or a native fixture execution must be recorded separately; earlier successful
guest results are not automatically assigned to new wrapper bytes.
The public prepare-only CLI and the actual native ARM64 GNU build both passed
against the exact original source/archive/vendor cohort. The public entrypoint
ran normal offline, locked Cargo metadata and release-build commands, both with
exit status zero, and rechecked the original source, all archives/vendor bytes,
public inputs, compiler distribution and GNU sysroot afterwards. It reproduced
the same 1,452,688-byte populated-fixture ELF with SHA-256
`2bdf7e7c65bd03a372f79688df47ad99090ea73b9a951fa957761801f52ffeb3`.
A fresh ARM/HVF XFS replay then dispatched that executable from the public
builder's own output directory. All seven original reflink tests passed without
skips, followed by six distinct native fixture processes covering both 8 and
64 MiB origins. The public independent verifier accepted all 92 complete
file witnesses in the new raw result, SHA-256
`e316bf46bec9ccad23548909f19f7a626e5b651da425db310ceb215532172549`.
Original/public source, archives, vendor, compiler/sysroot records and native
guest dependencies matched their guards before and after this replay. Twelve
focused checks also accepted the fresh report and rejected altered evidence.
The public build and fresh replay are separate actual records; compilation
alone is never counted as another runtime experiment or a speed measurement.
The [public reproduction record](../../docs/measurements/official_cubecow_public_reproduction_2026-09-27.json)
retains both actual Cargo commands, the full new guest phases, independent
verification, host caps/input guards, missing-Git preflight and genuine cached
Git acquisition. Personal host paths are replaced with semantic placeholders;
original raw SHA fingerprints remain distinct from sanitized parsed records.

## Prerequisites supplied by the operator

- A genuine working Git executable and a clean official checkout at the exact
  commit above. The builder invokes real Git to check the revision and clean
  state; an environment without Git fails before preparing a build. It also
  matches all 34 original crate files against `source_pins.json`.
- The unchanged original lockfile's 106 `.crate` archives, plus a corresponding
  Cargo vendor directory. The builder verifies each archive SHA and all 5,214
  original member bytes, checks every generated checksum map and rejects extra
  or unverified vendor files. A caller can prepare these through normal
  `cargo vendor --locked` and an authenticated Cargo archive cache in a
  separate setup step. This script performs no download.
- Native Linux GNU, an owned standalone Rust/Cargo **1.93.1** distribution,
  and a working GNU compiler/CRT/linker. The measured experiment used ARM64,
  GCC 12.2 and glibc 2.36. The wrapper also accepts native x86-64 GNU, without
  claiming that architecture has reproduced the earlier guest measurement.
  The original musl build fails its unmodified ioctl ABI; do not patch it to
  treat a musl build as the recorded GNU result.
- For execution, an existing owned XFS filesystem with `reflink=1`, sufficient
  free space, procfs, `/dev/null`, the native GNU runtime, and genuine GNU
  `/usr/bin/sha256sum`. The measured XFS file was 512 MiB; populated origins
  were 8 and 64 MiB. No script here installs software, formats or mounts disks.

Version checks and input hashes are not distribution-signature verification.
The operator must authenticate supplied compiler/packages separately. Use an
unprivileged, disposable Linux GNU build environment with source, vendor,
archives, toolchain and relocated sysroot mounted read-only and networking
disabled. Cargo build scripts are real upstream code, not a sandbox supplied
by this Python wrapper.

## Prepare and build offline

Output must be fresh, its parent must already exist, and it must be disjoint
from all supplied inputs. Symlink paths/ancestors are rejected. The wrapper
generates only an owned fixture `Cargo.toml` containing the external absolute
path dependency. It preserves the promoted source and lock bytes and creates
its Cargo home, target directory, temporary files and logs below output.
Original source, vendor and declared compiler inputs are rehashed afterwards.
Genuine sysroot symlinks are recorded as link text without following them.

```bash
python3 examples/official_cubecow/prepare_build.py \
  --source /owned/official-CubeSandbox \
  --vendor /owned/cargo-vendor \
  --archives /owned/original-crate-archives \
  --output /owned/new-cubecow-build \
  --toolchain-root /owned/rust-1.93.1-gnu \
  --cargo /owned/rust-1.93.1-gnu/bin/cargo \
  --rustc /owned/rust-1.93.1-gnu/bin/rustc
```

For a relocated GNU compiler/CRT, additionally supply `--sysroot` and the
matching caller-owned `LD_LIBRARY_PATH`, `GCC_EXEC_PREFIX`, `COMPILER_PATH`
and `LIBRARY_PATH`. For example, pass
`--rustflags '-C linker=/owned/gnu-sysroot/usr/bin/aarch64-linux-gnu-gcc-12 -C link-arg=--sysroot=/owned/gnu-sysroot'`.
`--sysroot` protects and records that directory; it does not silently configure
a linker or install libraries. With no explicit `--rustflags`, the wrapper uses
the caller's `RUSTFLAGS` and records those compiler flags.

Normal Cargo commands are `metadata --offline --locked` and
`build --release --offline --locked --jobs 2`.
No explicit `--target` is supplied: native GNU linker flags must also reach
host build scripts and proc macros, as in the measured original build.
The generated package is an ordinary Cargo workspace and uses the complete
original library path dependency. It does not regenerate or relax the lock.
`build-result.json` retains command status, stdout/stderr hashes, all input
guards and the resulting ELF hash. Failed commands retain their owned output.
Compilation alone never counts as a successful reflink experiment.

For platform-neutral preparation without invoking Cargo, use the same first
four arguments with `--prepare-only` and a different fresh output. A prepared
record explicitly has `compiled: false` and `fixture_runtime_executed: false`.

## Execute on existing owned XFS

Use the binary from your own `build-result.json`; binary hashes may differ
from the measured compiler/runtime cohort. Keep its native GNU dependencies
available. The fixture always invokes `/usr/bin/sha256sum`, whose actual bytes
must be recorded before execution. For the measured run that binary was GNU
coreutils 9.1 with SHA-256
`d72f4681a531bd27e3deaed5cbe4b1df0f87a4d69b9c12d5772712321405d709`.

The following checks an existing mount; it does not create one. Substitute
canonical, caller-owned paths with no symlink ancestors. Both parent
directories must exist and root/report paths must be fresh and disjoint.

```bash
set -euo pipefail
CUBE_XFS_PARENT=/owned/existing-xfs
CUBE_REPORT_PARENT=/owned/reports
CUBE_FIXTURE=/owned/new-cubecow-build/target/release/fpb-cubecow-populated-fixture
test "$(findmnt -n -o FSTYPE --target "$CUBE_XFS_PARENT")" = xfs
xfs_info "$CUBE_XFS_PARENT" | rg 'reflink=1'
test -d "$CUBE_REPORT_PARENT"
test ! -e "$CUBE_XFS_PARENT/new-populated" && test ! -L "$CUBE_XFS_PARENT/new-populated"
test ! -e "$CUBE_REPORT_PARENT/new-populated.json" && test ! -L "$CUBE_REPORT_PARENT/new-populated.json"
"$CUBE_FIXTURE" --root "$CUBE_XFS_PARENT/new-populated" \
  --report "$CUBE_REPORT_PARENT/new-populated.json"
```

The master launches six distinct fresh native processes:

1. Fully populate nonzero 8/64 MiB origins; create a snapshot, a snapshot of
   that snapshot and two writable forks; compare all contents and SHA values.
2. Mutate separate 64 KiB regions in fork A's prefix, fork B's midpoint and
   the live origin's suffix; both snapshots retain the complete original data.
3. Grow A by 1 MiB, require a fully zero new tail and exact prior contents,
   and require `InvalidArg` for a shrink.
4. Delete the origin; require `NotFound` for that name while its two snapshots
   and original backing directory remain intact.
5. Reconstruct the namespace through a fresh original public constructor,
   recover snapshots by canonical names and create a new writable fork from
   the orphan snapshot. The original API intentionally lists no snapshots
   for a deleted origin. Deleting the final snapshot must reap its directory.
6. Verify all surviving branch bytes and then delete every branch, leaving
   an empty namespace and owned volume directory.

Duplicate creation requires the exact `AlreadyExists` error. No unsupported
filesystem skip is accepted. The fixture compares every file byte, complete
SHA, length, backing path/device/inode and child PID. The 92 repeated file
comparisons total 3,485,466,624 bytes; this is not unique data or throughput.
Writers use `File::sync_all()` before verification; upstream `FICLONE` does not
fsync the destination. There is no concurrent-writer, power-loss durability,
whole-VM/process-memory, model-rollout or training-speed claim.

## Independently verify the raw result

Provide hashes from the actual executables you dispatched, not values copied
from an untrusted report. They are explicit operator declarations: the Python
verifier checks their agreement with the report, not remote execution or
distribution authenticity. It never imports or executes the Rust fixture.
Its independent stdlib byte oracle reconstructs all ten full-file states and
checks all six phases, both sizes, 92 witnesses, lengths, roles, PID matrix,
namespace paths, surviving inodes and expected negative API outcomes.

```bash
python3 examples/official_cubecow/verify_result.py \
  --result /owned/reports/new-populated.json \
  --output /owned/reports/new-populated-verified.json \
  --expected-fixture-sha256 YOUR_DISPATCHED_ELF_SHA256 \
  --expected-hash-utility-sha256 YOUR_ACTUAL_GNU_SHA256SUM_SHA256
```

The fresh verification record binds the raw result and verifier bytes.
Oracle execution time is a verifier cost, not snapshot or training speed.
Raw result provenance, real XFS/FICLONE execution and compiler authentication
remain separate evidence. Report verification does not attest native execution.

## Focused checks without another native experiment

```bash
FPB_CUBECOW_RESULT=/owned/preserved-actual-populated-result.json \
FPB_CUBECOW_EXPECTED_FIXTURE_SHA256=YOUR_DISPATCHED_ELF_SHA256 \
FPB_CUBECOW_EXPECTED_HASH_UTILITY_SHA256=YOUR_ACTUAL_GNU_SHA256SUM_SHA256 \
python3 -m unittest discover -s examples/official_cubecow -p 'test_*.py' -v
```

The report tests use one independently computed byte oracle and alter copies
of a genuine report to test missing/failed phases, repeated PIDs, changed
binary/source declarations, incomplete file comparisons, malformed paths,
wrong hashes and invalid lifecycle outcomes. Missing explicit real evidence
is reported as a skip. Guard tests exercise real temporary files and do not
pretend to execute a kernel, XFS or the native fixture.

The wrapper/verifier/example are project code. Upstream CubeSandbox retains
its Apache-2.0 license and documented third-party exceptions; no upstream
source, third-party binary or vendor payload is redistributed here.
