# Populated CubeCoW snapshot/fork latency

This fixture measures the complete, unchanged CubeCoW library from
[CubeSandbox v0.7.0](https://github.com/TencentCloud/CubeSandbox/tree/d0081641c59822e4e5653b7462e914410b81910a/cubecow),
commit `d0081641c59822e4e5653b7462e914410b81910a`. It calls the author's public
constructor, snapshot and writable-volume APIs; it provides no replacement
engine, source patch or full-copy fallback for `FICLONE`.

The promoted Rust source, manifest and lock are byte-identical to the files
compiled for the actual native ARM/HVF XFS experiment. Their SHA-256 values are:

- `src/main.rs`: `004060e519922c9934cb1e644494ba236c576ce6e6fbf434f6ae5d1367d4b117`
- `Cargo.toml`: `02c286a8d6cfe6945b91f3b30da0bc77c3c35cea9dc0b42c7ac0cb206ee2a72c`
- `Cargo.lock`: `efceea9589b16ea3c4f1e78e07fa0ee3abce7042dcbbb0512dcc67ae418ae6b1`

The [measurement review](../../../docs/CUBE_COW_LATENCY.md) provides all twelve
groups, matching persistence boundaries and precise scope. The
[full evidence](../../../docs/measurements/official_cubecow_latency_2026-09-27.json)
preserves all 240 raw method trials, all 2,720 whole-file witnesses, independent
statistics and failed-attempt summaries. The public verifier was executed
against those actual records. The following manual build/run recipe is a new
reproduction entrypoint; it has not independently generated the recorded ELF
or repeated that guest run. No compiler, upstream library, vendor payload,
guest disk or executable binary is redistributed.

## Operator prerequisites

Use a disposable **native Linux GNU** environment. The measured cohort used
ARM64, genuine Rust/Cargo 1.93.1, GCC 12.2, glibc 2.36, an unchanged official
checkout and 106 authenticated registry archives matching the original lock.
The corresponding vendor tree's 5,214 original files were checked against
archive bytes. Read [the populated example](../README.md) for preparation and
source/archive/vendor verification. Its `prepare_build.py` prepares the
populated fixture; it is not a latency-fixture build command.

Mount the original checkout at `/source`, the original vendor directory at
`/vendor`, the standalone GNU Rust toolchain at `/toolchain`, the matching
relocated GCC/CRT sysroot at `/gnu-sysroot`, and this directory at
`/latency-input`, all read-only. Supply a fresh owned writable `/work` and a
bounded writable temporary directory. The measured build container disabled
networking, dropped all capabilities, ran as an ordinary user and limited CPU,
memory and process count. Real upstream Cargo build scripts require that
separate sandbox; this example does not sandbox a host compiler itself.

For execution, provide a pre-existing **owned XFS filesystem, `reflink=1`, at
most 512 MiB in statvfs total size**, with enough free space for the fixed
344 MiB maximum live payload plus metadata/headroom. The experiment used a
512 MiB regular backing file in a dedicated 512 MiB, one-vCPU ARM/HVF guest,
not Docker overlay storage. Supply the matching native GNU loader/libraries,
procfs, `/dev/null` and genuine GNU `/usr/bin/sha256sum`. The unchanged original
musl compile has an ioctl ABI error; this recipe does not patch it.

These are operator-supplied, authenticated inputs. No command below installs
packages, formats a disk, creates a loop device or mounts a filesystem.

## Manual native GNU build

Verify the original Git commit, clean state and all 34 crate file hashes from
the parent example's `source_pins.json` before and after compilation. Verify
the original archive/vendor and compiler distribution separately. The fixed
Cargo manifest uses the container's `/source/cubecow` path dependency.

```bash
set -euo pipefail
CUBE_LATENCY_BUILD=/work/new-latency-build
test ! -e "$CUBE_LATENCY_BUILD" && test ! -L "$CUBE_LATENCY_BUILD"
test "$(git -C /source rev-parse HEAD)" = d0081641c59822e4e5653b7462e914410b81910a
test -z "$(git -C /source status --porcelain --untracked-files=all)"
/toolchain/bin/rustc -vV
/toolchain/bin/cargo -vV
mkdir -p "$CUBE_LATENCY_BUILD/fixture/src" "$CUBE_LATENCY_BUILD/cargo-home"
cp /latency-input/Cargo.toml /latency-input/Cargo.lock "$CUBE_LATENCY_BUILD/fixture/"
cp /latency-input/src/main.rs "$CUBE_LATENCY_BUILD/fixture/src/main.rs"
cat > "$CUBE_LATENCY_BUILD/cargo-home/config.toml" <<'EOF'
[source.crates-io]
replace-with = "vendored-sources"
[source.vendored-sources]
directory = "/vendor"
EOF
export CARGO_HOME="$CUBE_LATENCY_BUILD/cargo-home"
export CARGO_TARGET_DIR="$CUBE_LATENCY_BUILD/target"
export RUSTC=/toolchain/bin/rustc
export RUSTDOC=/toolchain/bin/rustdoc
export RUSTFLAGS='-C linker=/gnu-sysroot/usr/bin/aarch64-linux-gnu-gcc-12 -C link-arg=--sysroot=/gnu-sysroot'
export LD_LIBRARY_PATH=/toolchain/lib:/gnu-sysroot/usr/lib/aarch64-linux-gnu:/gnu-sysroot/lib/aarch64-linux-gnu
export GCC_EXEC_PREFIX=/gnu-sysroot/usr/lib/gcc/
export COMPILER_PATH=/gnu-sysroot/usr/lib/gcc/aarch64-linux-gnu/12:/gnu-sysroot/usr/bin
export LIBRARY_PATH=/gnu-sysroot/usr/lib/gcc/aarch64-linux-gnu/12:/gnu-sysroot/usr/lib/aarch64-linux-gnu:/gnu-sysroot/lib/aarch64-linux-gnu
export PATH=/gnu-sysroot/usr/bin:/toolchain/bin:/usr/local/bin:/usr/bin:/bin
/toolchain/bin/cargo metadata --manifest-path "$CUBE_LATENCY_BUILD/fixture/Cargo.toml" \
  --format-version 1 --offline --locked > "$CUBE_LATENCY_BUILD/metadata.json"
/toolchain/bin/cargo build --manifest-path "$CUBE_LATENCY_BUILD/fixture/Cargo.toml" \
  --offline --locked --release --jobs 2
sha256sum "$CUBE_LATENCY_BUILD/fixture/src/main.rs" \
  "$CUBE_LATENCY_BUILD/fixture/Cargo.toml" "$CUBE_LATENCY_BUILD/fixture/Cargo.lock" \
  "$CUBE_LATENCY_BUILD/target/release/fpb-cubecow-latency-fixture" \
  /usr/bin/sha256sum > "$CUBE_LATENCY_BUILD/dispatched-input-sha256.txt"
```

Use the correct native compiler/CRT paths for a different GNU architecture and
record that as a different build/runtime cohort. No explicit Cargo `--target`
is passed: native linker flags also reach host build scripts and proc macros.
Keep the promoted lock unchanged; a successful compile is not an XFS result.

## Run on an existing owned mount

The existing XFS mount must meet the bound above; 344 MiB is planned live
payload, not the recommended total mount size. This recipe checks the existing
mount. Root/report paths must be fresh, canonical, disjoint and nonsymlinked,
with existing parents. Record actual binary and hash-utility bytes before the
run, rather than taking their values from the report.

```bash
set -euo pipefail
CUBE_LATENCY_XFS=/tmp/fpb-xfs
CUBE_LATENCY_BINARY=/work/new-latency-build/target/release/fpb-cubecow-latency-fixture
CUBE_LATENCY_REPORT=/tmp/new-cubecow-latency.json
test "$(findmnt -n -o FSTYPE --target "$CUBE_LATENCY_XFS")" = xfs
xfs_info "$CUBE_LATENCY_XFS" | rg 'reflink=1'
test ! -e "$CUBE_LATENCY_XFS/latency" && test ! -L "$CUBE_LATENCY_XFS/latency"
test ! -e "$CUBE_LATENCY_REPORT" && test ! -L "$CUBE_LATENCY_REPORT"
sha256sum "$CUBE_LATENCY_BINARY" /usr/bin/sha256sum
"$CUBE_LATENCY_BINARY" --root "$CUBE_LATENCY_XFS/latency" \
  --report "$CUBE_LATENCY_REPORT"
```

Every size/mode/operation group performs two warmup pairs and eight measured
pairs, alternating original-first and copy-first. The complete matrix is
12 groups and 240 method trials. A failed performed trial is persisted and
aborts acceptance; it cannot be dropped from a speed distribution. Sources
are populated with independently verified nonzero data. All validation,
bidirectional mutations, restoration, cleanup and report persistence remain
outside the primitive timer.

## Independently verify actual raw records

The verifier uses only Python's standard library and no Rust fixture import.
It independently regenerates the complete 8/64 MiB patterns and all mutation
digests, validates the complete chronology and witness matrix, recalculates
all distributions and keeps all eight measured pairs. Supply the executable
path/root and lowercase SHA values from your actual dispatched inputs.

```bash
python3 examples/official_cubecow/latency/verify_result.py \
  --result /owned/reports/new-cubecow-latency.json \
  --expected-root /tmp/fpb-xfs/latency \
  --expected-executable /work/new-latency-build/target/release/fpb-cubecow-latency-fixture \
  --expected-fixture-sha256 YOUR_DISPATCHED_ELF_SHA256 \
  --expected-hash-utility-sha256 YOUR_ACTUAL_GNU_SHA256SUM_SHA256 \
  --output /owned/reports/new-cubecow-latency-verified.json
```

Input hashes are explicit operator declarations. Verifying their consistency
with a raw record is not remote execution attestation or package authenticity.
Without a separate actual host record, `host_check.status` is `unverified`;
without the external historical build/source inputs, `build_check.status` is
`unverified`. Neither optional gate is silently treated as passed.

For the frozen historical experiment only, `--host-result` accepts the full
unsanitized operator-owned host record and validates native HVF resources,
owned XFS, no network/host filesystem share, original seven unskipped test
bodies, process completion, guest/raw byte hashes and immutable runtime/build
inputs. `--build-dir` plus `--source` rehash all frozen build records, the
compiled ELF/helper and 34 unchanged original crate files. These ignored
external artifacts are not redistributed. The public measurement preserves
their hashes and the result of the actual independent check.

Paired full-copy/original median ratios are distinct from ratios of method
medians. The verifier reports both, all raw pairs and observed ranges. It uses
20,000 deterministic order-stratified bootstrap replicates, including a more
conservative `0.05/12` tail choice across twelve groups. These are descriptive
bounds; eight serial pairs in one guest do not establish population confidence
coverage or universal performance. This measures warm-cache populated XFS
snapshot/fork primitives, not process RAM checkpoints, whole VMM startup,
model rollout, GPU utilization or training speed. Caller fsync completion is
not a tested host power-loss guarantee.

## Copied-evidence controls

```bash
FPB_CUBECOW_LATENCY_RESULT=/owned/preserved-actual-latency-result.json \
FPB_CUBECOW_LATENCY_FIXTURE_SHA256=YOUR_DISPATCHED_ELF_SHA256 \
FPB_CUBECOW_LATENCY_HASH_UTILITY_SHA256=YOUR_ACTUAL_GNU_SHA256SUM_SHA256 \
python3 -m unittest discover -s examples/official_cubecow/latency -p 'test_*.py' -v
```

Set `FPB_CUBECOW_LATENCY_ROOT` and `FPB_CUBECOW_LATENCY_EXECUTABLE` when your
actual dispatch paths differ from the historical guest. Controls alter copies
of a real record and require rejection of omitted/extra/failed samples,
unbalanced order, warmup inclusion, invalid clocks, false durability,
incomplete persistence, hash/isolation failures, aliasing, path escape,
resource/cleanup drift and invented statistics. They never modify the original
raw input and do not pretend to execute a kernel or XFS. Missing explicitly
supplied real evidence is a skip.

These owned fixture/verifier files are project code. Upstream CubeSandbox
retains its Apache-2.0 license and documented third-party exceptions.
