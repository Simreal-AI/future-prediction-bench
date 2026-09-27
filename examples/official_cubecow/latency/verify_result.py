"""Independent acceptance and descriptive statistics for the frozen XFS trial.

Only Python's standard library is used. This does not execute the Rust helper
or import its implementation. Runtime records remain operator-owned evidence,
not an externally signed attestation of execution.
"""
import argparse
import hashlib
import itertools
import json
import math
import random
import re
import statistics
from pathlib import Path, PurePosixPath

MIB = 1_048_576
SIZES = (8 * MIB, 64 * MIB)
MODES = ("return", "caller-durable")
OPERATIONS = ("snapshot", "fork", "checkpoint+fork")
METHODS = ("original", "full-copy")
PIN = "d0081641c59822e4e5653b7462e914410b81910a"
MANIFEST_SHA = "1649c2cfcb557a85b3741928406765a583f91072b731489aa6a252476c0c68c9"
FIXTURE_SHA = "662740d496615e69829ab4bd2ed1688188b74b74f30392c2da5ecb6fdafae355"
COREUTILS_SHA = "d72f4681a531bd27e3deaed5cbe4b1df0f87a4d69b9c12d5772712321405d709"
ORACLE_SHA = "82ec9a8dbb1f7055f282b029d0bafd35b4e75c7b9420d985f7c6f39a47420a9a"
ROOT = PurePosixPath("/tmp/fpb-xfs/latency")
ORIGINAL_TESTS = {
    "validate_name_rejects_bad_inputs", "create_and_list_volume_roundtrip",
    "snapshot_create_delete_and_listing", "create_volume_from_volume_source",
    "names_share_a_global_namespace", "resize_only_grows_volume_main_file",
    "scan_recovers_volumes_and_snapshots_after_restart",
}
DECLARED_HASHES = {
    8 * MIB: {
        "base": "d203aa98a4525db8c356a38098e44cccc998195281ba810466f23cb3577f1a7e",
        "prefix": "e9496e71926535b3e2230d51b6e8788c46ec9bc8ead6b49810eba1092fe7ccc2",
        "middle": "f9df3f081e35b420ba290978be3b02d51b0b0c73bdbaf9a885492ee7cdabacf1",
        "suffix": "cb488e44efee70943569c6d6e8145e04be930b4bb481d14858545bfea4508b13",
    },
    64 * MIB: {
        "base": "f14951d41351d8f42d1e91a579226d288ee8fb740b28846e87b5eee234786c92",
        "prefix": "b9f644ef4b30d46e652e341bd93e36422c4e44a9240c8b81ae4422eab71fef18",
        "middle": "6349192744d3e85ad6580d48f1ececc642f7d5a21f2baa8425bf5f7e53c83c6e",
        "suffix": "5214d00813aac9cbf00d8956fbab12398f1a27ab07cb5793546f93f0f7e59e9b",
    },
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, message, minimum=0, maximum=None):
    require(type(value) is int and value >= minimum, message)
    require(maximum is None or value <= maximum, message)
    return value


def no_duplicates(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def load(path):
    raw = Path(path).read_bytes()
    def invalid(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=no_duplicates, parse_constant=invalid), hashlib.sha256(raw).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_path(value):
    require(type(value) is str, "path is not a string")
    path = PurePosixPath(value)
    require(path.is_absolute() and str(path) == value and ".." not in path.parts,
            "noncanonical or relative guest path")
    return path


def independent_oracle():
    """Independently generate every byte, patch and whole-file SHA-256."""
    hashes = {}
    mutations = {}
    for size in SIZES:
        # Division, rather than the Rust helper's shifts, expresses the high
        # bit terms. The bounded offsets cannot overflow the Rust u64 formula.
        data = bytearray((((i * 73) ^ (i // 256) ^ (i // MIB) ^ (size // MIB)) % 251) + 1
                         for i in range(size))
        require(len(data) == size and 0 not in data, "independent population defect")
        baseline = hashlib.sha256(data).hexdigest()
        output = {"base": baseline}
        details = []
        for kind, start, tag in (("prefix", 0, 31), ("middle", size // 2, 67),
                                 ("suffix", size - 65_536, 113)):
            saved = data[start:start + 65_536]
            patch = bytes(((i * 17 + tag) % 251) + 1 for i in range(65_536))
            changed = sum(a != b for a, b in zip(saved, patch))
            require(changed > 0, "independent patch did not change data")
            data[start:start + 65_536] = patch
            output[kind] = hashlib.sha256(data).hexdigest()
            data[start:start + 65_536] = saved
            require(hashlib.sha256(data).hexdigest() == baseline, "independent restoration defect")
            details.append({"kind": kind, "offset": start, "bytes": 65_536,
                            "tag": tag, "bytes_changed": changed})
        require(output == DECLARED_HASHES[size], "fresh independent hashes disagree with frozen constants")
        hashes[size], mutations[str(size)] = output, details
    return hashes, mutations


def verify_build(build_dir, source_dir):
    build_dir, source_dir = Path(build_dir), Path(source_dir)
    manifest, digest = load(build_dir / "manifest.json")
    require(digest == MANIFEST_SHA, "frozen build manifest differs")
    require(manifest["upstream_commit"] == PIN and manifest["original_source_unchanged"] is True,
            "incorrect original source cohort")
    records = dict(manifest["record_sha256"])
    records[manifest["fixture_executable"]["path"]] = FIXTURE_SHA
    records[manifest["coreutils_executable"]["path"]] = COREUTILS_SHA
    for relative, expected in records.items():
        path = PurePosixPath(relative)
        require(not path.is_absolute() and ".." not in path.parts, "build record escaped scope")
        require(sha_file(build_dir / relative) == expected, "build record changed: " + relative)
    for relative, expected in manifest["original_source_files_sha256"].items():
        require(sha_file(source_dir / "cubecow" / relative) == expected,
                "original library source changed: " + relative)
    return {"manifest_sha256": digest, "build_record_files_rehashed": len(records),
            "original_library_files_rehashed": len(manifest["original_source_files_sha256"]),
            "fixture_sha256": FIXTURE_SHA, "coreutils_sha256": COREUTILS_SHA,
            "scope": "fixed build records, executable and original library files; full vendor/compiler revalidation remains in the pinned build record"}


def distribution(values):
    if not values:
        return None
    ordered = sorted(values)
    return {"count": len(values), "minimum_ns": min(values),
            "median_ns": statistics.median(values),
            "mean_ns": sum(values) / len(values),
            "p95_ns": ordered[math.ceil(len(values) * .95) - 1],
            "maximum_ns": max(values), "raw_ns": values}


def compare_distribution(actual, values):
    expected = distribution(values)
    if expected is None:
        require(actual is None, "unexpected distribution for unmeasured boundary")
        return
    require(type(actual) is dict and set(actual) == set(expected), "distribution fields differ")
    for key, value in expected.items():
        if key in ("median_ns", "mean_ns"):
            require(type(actual[key]) in (int, float) and math.isfinite(actual[key]) and
                    math.isclose(actual[key], value, rel_tol=1e-12, abs_tol=1e-9),
                    "reported distribution differs from actual samples: " + key)
        else:
            require(actual[key] == value, "reported distribution differs from actual samples: " + key)


class Witnesses:
    def __init__(self, oracle):
        self.oracle, self.common_dev = oracle, None
        self.prepared_identities = {}
        self.count, self.bytes = 0, 0

    def group(self, records, specifications, size, trial_identities):
        require(type(records) is list and len(records) == len(specifications), "whole-file witness count differs")
        for record, (role, path, kind) in zip(records, specifications):
            require(type(record) is dict and set(record) == {"role", "path", "size_bytes", "kind", "sha256",
                    "independent_expected_sha256", "dev", "inode", "allocated_bytes"}, "witness fields differ")
            require(record["role"] == role and canonical_path(record["path"]) == path and
                    record["kind"] == kind, "witness role/path/kind differs")
            require(integer(record["size_bytes"], "invalid logical size") == size, "whole-file logical size differs")
            require(record["sha256"] == self.oracle[size][kind] and
                    record["independent_expected_sha256"] == self.oracle[size][kind], "independent whole-file digest differs")
            dev, inode = integer(record["dev"], "invalid device", 1), integer(record["inode"], "invalid inode", 1)
            integer(record["allocated_bytes"], "nonpopulated or invalid allocation", 512)
            if self.common_dev is None:
                self.common_dev = dev
            require(dev == self.common_dev, "witness escaped the common XFS device")
            identity = (dev, inode)
            old = trial_identities.setdefault(path, identity)
            require(old == identity, "file identity changed within a trial")
            if path.name.startswith(("source-", "reference-")):
                old = self.prepared_identities.setdefault(path, identity)
                require(old == identity, "protected source/reference identity changed")
            self.count += 1
            self.bytes += size


def verify_samples(report, oracle, *, root=ROOT, expected_fixture_sha256=FIXTURE_SHA, expected_hash_utility_sha256=COREUTILS_SHA, expected_executable="/opt/fpb-cubecow/latency-fixture"):
    require(report["status"] == "pass" and report["cleanup_all_prepared_data_completed"] is True,
            "runtime or final cleanup did not pass")
    require(report["source_commit"] == PIN and report["fixture_executable_sha256"] == expected_fixture_sha256 and
            report["coreutils_executable_sha256"] == expected_hash_utility_sha256 and
            report["independent_pattern_digest_evidence_sha256"] == ORACLE_SHA, "runtime executable/source cohort differs")
    require(canonical_path(report["fixture_executable"]) == canonical_path(expected_executable), "unexpected executed fixture path")
    require(report["warmup_pairs"] == 2 and type(report["warmup_pairs"]) is int and
            report["measured_pairs"] == 8 and type(report["measured_pairs"]) is int and
            report["sizes_bytes"] == list(SIZES) and report["planned_max_live_payload_bytes"] == 344 * MIB,
            "fixed measurement matrix changed")
    samples, groups = report["samples"], report["groups"]
    require(type(samples) is list and len(samples) == 240 and type(groups) is list and len(groups) == 12,
            "incomplete or extra sample/group matrix")
    witnesses = Witnesses(oracle)
    checked = []
    for group_index, (size, mode, operation) in enumerate(itertools.product(SIZES, MODES, OPERATIONS)):
        group = groups[group_index]
        require(group["size_bytes"] == size and group["mode"] == mode and group["operation"] == operation and
                group["all_pairs_accepted"] is True and group["warmup_pairs"] == 2 and group["measured_pairs"] == 8 and
                group["original_first_measured_pairs"] == 4 and group["copy_first_measured_pairs"] == 4,
                "group identity, pair acceptance or order differs")
        measured = {method: [] for method in METHODS}
        paired = []
        for pair in range(10):
            order = METHODS if pair % 2 == 0 else METHODS[::-1]
            current_pair = {}
            for slot, method in enumerate(order):
                sample = samples[group_index * 20 + pair * 2 + slot]
                require(sample["status"] == "pass" and sample["size_bytes"] == size and
                        sample["mode"] == mode and sample["operation"] == operation and
                        integer(sample["pair"], "invalid pair") == pair and
                        type(sample["warmup"]) is bool and sample["warmup"] == (pair < 2) and
                        sample["order"] == list(order) and integer(sample["order_slot"], "invalid order slot") == slot and
                        sample["method"] == method and "error" not in sample, "sample sequence, status or fixed order differs")
                returned = integer(sample["operation_return_ns"], "invalid actual operation time", 1, 900_000_000_000)
                require(sample["cleanup_completed"] is True and
                        sample["generated_layout_has_no_unaccounted_metadata_files"] is True, "sample cleanup or layout failed")
                namespace = root / ("official" if method == "original" else "copy")
                source_name = f"source-{size // MIB}m"
                source = namespace / "volumes" / source_name / source_name
                reference = source.parent / f"reference-{size // MIB}m"
                label = f"{size // MIB}m-{mode}-{operation.replace('+', '_')}-p{pair}-{method}"
                snapshot = source.parent / ("snap-" + label)
                branch = namespace / "volumes" / ("fork-" + label) / ("fork-" + label)
                created = []
                if operation != "fork":
                    created.append(("snapshot", snapshot, "base"))
                if operation != "snapshot":
                    created.append(("branch", branch, "base"))
                final = created[-1][1]
                identities = {}
                witnesses.group(sample["before"], [("source", source, "base"),
                                ("prepared-reference", reference, "base")], size, identities)
                witnesses.group(sample["at_return_validation"], created, size, identities)
                destination = [("mutated-destination", final, "prefix"),
                               ("source-after-destination-write", source, "base"),
                               ("reference-after-destination-write", reference, "base")]
                if operation == "checkpoint+fork":
                    destination += [("snapshot-after-branch-write", snapshot, "base"),
                                    ("mutated-snapshot", snapshot, "middle"),
                                    ("branch-after-snapshot-write", branch, "prefix")]
                destination += [("restored-source", source, "base"), ("restored-reference", reference, "base")]
                witnesses.group(sample["destination_write_isolation"], destination, size, identities)
                upstream = reference if operation == "fork" else source
                witnesses.group(sample["upstream_write_isolation"], [("mutated-upstream", upstream, "suffix"),
                                ("destination-after-upstream-write", final, "prefix")], size, identities)
                require(len(set(identities.values())) == len(identities), "created/source/reference files alias inodes")
                space = sample["filesystem_space_before"]
                total = integer(space["total_bytes"], "invalid filesystem total", 1, 512 * MIB)
                used = integer(space["used_bytes"], "invalid filesystem usage", 0, total)
                worst = size * (2 if operation == "checkpoint+fork" else 1)
                require(space["worst_new_destination_bytes"] == worst and space["reserved_headroom_bytes"] == 8 * MIB and
                        used + worst + 8 * MIB <= total, "actual free-space/headroom gate fails")
                after = sample["filesystem_space_after_cleanup"]
                require(integer(after["total_bytes"], "post-cleanup filesystem total") == total and
                        integer(after["used_bytes"], "post-cleanup filesystem usage", 0, total) <= total and
                        after["volume_count"] == 2 and after["snapshot_count"] == 2, "original namespace counters did not return to prepared state")
                if mode == "caller-durable":
                    elapsed = integer(sample["caller_durable_total_ns"], "invalid durable time", returned, 900_000_000_000)
                    require(integer(sample["caller_durable_extra_ns"], "invalid durable extra") == elapsed - returned,
                            "durable elapsed arithmetic differs")
                    scope = sample["caller_durable_scope"]
                    directories = set()
                    for _, path, _ in created:
                        parent = path.parent
                        while True:
                            directories.add(parent)
                            if parent == root.parent:
                                break
                            parent = parent.parent
                    directories = sorted(directories, key=lambda p: (-len(p.parts), str(p)))
                    require(scope["data_files_synced"] == [str(path) for _, path, _ in created] and
                            scope["ancestor_directories_synced"] == [str(path) for path in directories] and
                            scope["scope_stop"] == str(root.parent) and scope["extra_separate_index_metadata_files"] == 0,
                            "caller-durable persistence scope differs")
                else:
                    require(sample["caller_durable_total_ns"] is None and sample["caller_durable_extra_ns"] is None and
                            "caller_durable_scope" not in sample, "return trial contains unmeasured persistence data")
                    elapsed = returned
                if pair >= 2:
                    measured[method].append(sample)
                    current_pair[method] = elapsed
            if pair >= 2:
                paired.append({"pair": pair, "order": list(order),
                               "original_ns": current_pair["original"], "full_copy_ns": current_pair["full-copy"],
                               "full_copy_over_original": current_pair["full-copy"] / current_pair["original"]})
        for method in METHODS:
            original = group["statistics"][method]
            require(set(original) == {"operation_return", "caller_durable_total"}, "unexpected group distribution fields")
            compare_distribution(original["operation_return"], [s["operation_return_ns"] for s in measured[method]])
            compare_distribution(original["caller_durable_total"], [s["caller_durable_total_ns"] for s in measured[method]]
                                 if mode == "caller-durable" else [])
        checked.append({"size_bytes": size, "mode": mode, "operation": operation, "paired_measurements": paired,
                        "latency_distributions": {method: distribution([p["original_ns" if method == "original" else "full_copy_ns"]
                                                                       for p in paired]) for method in METHODS}})
    require(len(witnesses.prepared_identities) == 8 and len(set(witnesses.prepared_identities.values())) == 8,
            "protected source/reference identities are incomplete or alias")
    require(witnesses.count == 2720, "fixed complete whole-file witness matrix differs")
    return {"status": "pass", "groups": checked, "verified_method_trials": 240,
            "verified_measured_method_trials": 192, "verified_warmup_method_trials": 48,
            "verified_whole_file_witness_count": witnesses.count,
            "verified_whole_file_witness_bytes": witnesses.bytes,
            "guest_filesystem_device": witnesses.common_dev, "protected_distinct_inodes": 8}


def quantile(values, q):
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def add_statistics(report, raw_sha, bootstrap_replicates=20_000):
    for group in report["groups"]:
        pairs = group["paired_measurements"]
        ratios = [row["full_copy_over_original"] for row in pairs]
        strata = [[row["full_copy_over_original"] for row in pairs if row["order"][0] == method] for method in METHODS]
        require(all(len(stratum) == 4 for stratum in strata), "bootstrap order strata are unbalanced")
        seed = int(hashlib.sha256((raw_sha + json.dumps([group[k] for k in ("size_bytes", "mode", "operation")])).encode()).hexdigest(), 16)
        rng = random.Random(seed)
        draws = [statistics.median([rng.choice(stratum) for stratum in strata for _ in range(4)])
                 for _ in range(bootstrap_replicates)]
        alpha = .05 / 12  # More conservative tail choice across the 12 groups.
        group["paired_full_copy_over_original_ratio"] = {
            "count": 8, "median": statistics.median(ratios), "minimum": min(ratios), "maximum": max(ratios),
            "raw_ratios": ratios,
            "order_stratified_bootstrap_95_percentile_interval": [quantile(draws, .025), quantile(draws, .975)],
            "order_stratified_bootstrap_familywise_adjusted_interval": [quantile(draws, alpha / 2), quantile(draws, 1 - alpha / 2)],
            "familywise_tail_alpha_per_group": alpha, "bootstrap_replicates": bootstrap_replicates,
            "seed_hex": hex(seed),
            "interpretation": "Descriptive paired bootstrap with four samples resampled in each execution-order stratum. Familywise tails use 0.05/12, but n=8, one guest and serial observations do not establish population coverage or universal speedup. Preserve observed min/max and all raw pairs."}
        group["ratio_of_method_medians"] = (group["latency_distributions"]["full-copy"]["median_ns"] /
                                              group["latency_distributions"]["original"]["median_ns"])
        group["median_latency_ms"] = {method: group["latency_distributions"][method]["median_ns"] / 1_000_000 for method in METHODS}


def verify_host(host, report, raw_sha):
    require(host["passed"] is True and host["original_7_tests_passed_without_skip"] is True and
            host["latency_all_240_trials_passed"] is True and host["XFS_formatted"] is True and
            host["XFS_mounted"] is True and host["XFS_unmounted"] is True and host["actual_procfs"] is True,
            "host execution/lifecycle gate failed")
    require(host["backend"] == "aarch64_hvf" and host["architecture"] == "aarch64" and
            host["kernel"] == "6.18.52-0-virt" and host["memory_mib"] == 512 and host["vcpu_count"] == 1 and
            host["whole_VMM_executed"] is False and host["model_or_optimizer_executed"] is False,
            "native guest resource or claim scope differs")
    require(host["latency_result"] == report and host["latency_result_sha256"] == raw_sha and
            host["latency_report_guest_sha256"] == raw_sha, "host and actual guest raw bytes differ")
    require(host["latency_cohort_manifest_sha256"] == MANIFEST_SHA and
            host["latency_cohort_manifest_sha256_after"] == MANIFEST_SHA and
            host["source_sha256"] == host["source_sha256_after"] and
            host["vm_runtime_source_sha256"] == host["vm_runtime_source_sha256_after"], "execution inputs changed")
    argv = host["actual_QEMU_argv"]
    require(type(argv) is list and all(type(v) is str for v in argv), "actual QEMU argv absent")
    for flag, value in (("-machine", "virt,accel=hvf"), ("-cpu", "host"), ("-m", "512"), ("-smp", "1"), ("-nic", "none")):
        require(argv.count(flag) == 1 and argv[argv.index(flag) + 1] == value, "QEMU backend/resource/network isolation differs")
    require(not any("cache=unsafe" in word or "9p" in word or "virtiofs" in word for word in argv),
            "unsafe cache or host directory mount in actual QEMU argv")
    require(re.search(r"/mnt/root/tmp/fpb-xfs xfs ", host["actual_mounts"]) and
            "reflink=1" in host["actual_XFS_info"] and
            "/mnt/root/opt/fpb-owned/owned-xfs-vdev.img" in host["actual_loop_backing_file"], "genuine owned XFS evidence absent")
    require(host["latency_execution"]["return_code"] == 0 and
            host["latency_execution"]["supervised_background_guest_process"] is True, "actual benchmark process failed")
    summary = json.loads(host["latency_execution"]["stdout"])
    require(summary["status"] == "pass" and summary["groups"] == 12 and summary["all_method_trials"] == 240 and
            summary["measured_method_trials"] == 192, "actual benchmark stdout summary differs")
    require(host["latency_poll_states"] and host["latency_poll_states"][-1] == "done" and
            set(host["latency_poll_states"]) <= {"running", "done"}, "supervised process lifecycle differs")
    original = host["original_tests"]
    # The actual serial transport combines stdout/stderr in its stdout field.
    text = original["stdout"] + original.get("stderr", "")
    names = re.findall(r"^test engine::reflink::tests::([a-z_]+) \.\.\. ok$", original["stdout"], re.M)
    require(original["return_code"] == 0 and len(names) == 7 and set(names) == ORIGINAL_TESTS and
            not re.search(r"\[skip\]|skipping|does not support FICLONE", text, re.I) and
            "7 passed; 0 failed; 0 ignored; 0 measured; 12 filtered out" in text, "original seven test bodies missing, skipped or failed")
    return {"status": "pass", "actual_backend": "aarch64_hvf", "vcpus": 1, "memory_mib": 512,
            "genuine_owned_XFS_reflink": True, "original_test_bodies_passed_without_skip": 7,
            "host_power_loss_tested": False, "external_signed_attestation": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--host-result", type=Path,
                        help="Optional full historical host record: independently verifies the measured native HVF execution gates")
    parser.add_argument("--build-dir", type=Path,
                        help="Optional external historical build record directory (not packaged)")
    parser.add_argument("--source", type=Path,
                        help="Exact unchanged official checkout, required together with --build-dir")
    parser.add_argument("--expected-root", required=True,
                        help="Canonical root supplied in your actual --root command")
    parser.add_argument("--expected-executable", required=True,
                        help="Canonical native executable actually dispatched")
    parser.add_argument("--expected-fixture-sha256", required=True,
                        help="SHA-256 of executable you dispatched; operator declaration, not remote attestation")
    parser.add_argument("--expected-hash-utility-sha256", required=True,
                        help="SHA-256 of actual GNU /usr/bin/sha256sum you dispatched")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    root = canonical_path(args.expected_root)
    canonical_path(args.expected_executable)
    for value in (args.expected_fixture_sha256, args.expected_hash_utility_sha256):
        require(re.fullmatch(r"[0-9a-f]{64}", value) is not None, "require an explicit lowercase SHA-256 declaration")
    require((args.build_dir is None) == (args.source is None), "supply --build-dir and --source together")
    output = args.output.absolute()
    require(output.parent.resolve() == output.parent and not output.exists() and not output.is_symlink()
            and output.parent.is_dir(), "require a fresh output file below canonical nonsymlink owned ancestors")
    result, raw_sha = load(args.result)
    build = ({"status": "unverified", "reason": "external historical build/source record inputs were not supplied"}
             if args.build_dir is None else {"status": "pass", **verify_build(args.build_dir, args.source)})
    host_sha = None
    if args.host_result is None:
        host_check = {"status": "unverified", "reason": "raw-report validation does not attest native execution"}
    else:
        require(args.expected_fixture_sha256 == FIXTURE_SHA and
                args.expected_hash_utility_sha256 == COREUTILS_SHA and root == ROOT and
                args.expected_executable == "/opt/fpb-cubecow/latency-fixture",
                "historical host gate is restricted to its exact compiled cohort")
        host, host_sha = load(args.host_result)
        host_check = verify_host(host, result, raw_sha)
    oracle, mutation_details = independent_oracle()
    report = verify_samples(result, oracle, root=root, expected_fixture_sha256=args.expected_fixture_sha256,
                            expected_hash_utility_sha256=args.expected_hash_utility_sha256,
                            expected_executable=args.expected_executable)
    add_statistics(report, raw_sha)
    report.update({"raw_guest_result_sha256": raw_sha, "host_result_sha256": host_sha,
                   "build_check": build, "host_check": host_check,
                   "operator_declared_executable_sha256": args.expected_fixture_sha256,
                   "operator_declared_hash_utility_sha256": args.expected_hash_utility_sha256,
                   "operator_declared_guest_root": str(root),
                   "operator_declared_executable_path": args.expected_executable,
                   "independent_expected_sha256": {str(size): values for size, values in oracle.items()},
                   "independent_mutations": mutation_details,
                   "verifier_source_sha256": sha_file(__file__),
                   "external_signed_execution_attestation": False,
                   "scope": "populated warm-cache XFS filesystem snapshot/fork primitive report; native execution requires its separate actual host record; no complete agent/model/token/GPU/training throughput claim",
                   "timing_validation_boundary": "whole-file data, bidirectional isolation, restoration, namespace inspection, cleanup and report persistence are outside the primitive timers"})
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "pass", "groups": 12, "samples": 240,
                      "whole_file_witnesses": report["verified_whole_file_witness_count"],
                      "host_execution_gate": host_check["status"], "build_gate": build["status"],
                      "output": str(output), "output_sha256": sha_file(output)}, indent=2))


if __name__ == "__main__":
    main()
