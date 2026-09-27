"""Independently verify the exact public populated CubeCoW report contract.

Expected bytes are reconstructed by stdlib Python, never by the Rust fixture.
Binary hashes are explicit operator pins, not remote execution attestations.
This verifier reads a report; it does not execute code, mount or format a disk.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import time

PIN = "d0081641c59822e4e5653b7462e914410b81910a"
MIB = 1048576
REGION = 65536
SIZES = (8 * MIB, 64 * MIB)
PHASES = ("populate", "mutate", "resize", "delete-origin", "recover-orphan", "cleanup")
BOUNDARY = "writers used File::sync_all before verification; upstream FICLONE does not fsync destination"
MAX_RESULT_BYTES = 8 * MIB


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_int(value, message, minimum=0):
    require(type(value) is int and value >= minimum, message)
    return value


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def parse_json(text):
    def invalid_constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    return json.loads(text, object_pairs_hook=no_duplicate_keys, parse_constant=invalid_constant)


def independent_oracle():
    output = {}
    for total in SIZES:
        mib_count = total // MIB
        # Division expresses the two high-bit terms independently of Rust's
        # shift implementation. These bounded offsets never overflow u64.
        data = bytearray(((((index * 73) ^ (index // 256) ^ (index // MIB) ^ mib_count) % 251) + 1) for index in range(total))
        require(len(data) == total and 0 not in data, "oracle population length/nonzero defect")
        initial = hashlib.sha256(data).hexdigest()
        hashes = {"original": initial}
        mutation_details = {}
        for kind, start, tag in (("a", 0, 31), ("b", total // 2, 67), ("source", total - REGION, 113)):
            saved = data[start:start + REGION]
            replacement = bytes(((relative * 17 + tag) % 251) + 1 for relative in range(REGION))
            changed = sum(left != right for left, right in zip(saved, replacement))
            require(len(saved) == REGION and len(replacement) == REGION and changed > 0, "oracle patch defect")
            data[start:start + REGION] = replacement
            patched = hashlib.sha256(data)
            hashes[kind] = patched.hexdigest()
            if kind == "a":
                expanded = patched.copy()
                expanded.update(bytes(MIB))
                hashes["expanded_a"] = expanded.hexdigest()
            data[start:start + REGION] = saved
            require(hashlib.sha256(data).hexdigest() == initial, "oracle patch restoration defect")
            mutation_details[kind] = {"offset": start, "bytes": REGION, "tag": tag, "bytes_changed": changed}
        require(len(set(hashes.values())) == 5, "independent oracle hashes unexpectedly collide")
        output[str(total)] = {"original_bytes": total, "sha256": hashes, "mutations": mutation_details, "expanded_a_bytes": total + MIB}
    return output


def safe_absolute_guest_path(value, label):
    require(type(value) is str and value.startswith("/") and "\0" not in value, "invalid " + label)
    path = PurePosixPath(value)
    require(".." not in path.parts and str(path) == value, "noncanonical " + label)
    return path


class WitnessVerifier:
    def __init__(self, oracle):
        self.oracle = oracle
        self.guest_root = None
        self.common_dev = None
        self.identities = {}
        self.checked = []

    def one(self, witness, size, role, kind):
        require(type(witness) is dict, "witness must be an object")
        expected_keys = {"role", "path", "logical_bytes", "allocated_bytes", "dev", "inode", "full_bytes_compared", "sha256", "expected_sha256", "fsync_boundary"}
        require(set(witness) == expected_keys, "unexpected/missing witness fields")
        require(witness["role"] == role, "wrong witness role")
        expected_size = size + (MIB if kind == "expanded_a" else 0)
        require(strict_int(witness["logical_bytes"], "bad logical length") == expected_size, "wrong logical length")
        require(strict_int(witness["full_bytes_compared"], "bad byte comparison count") == expected_size, "incomplete byte comparison")
        allocated = strict_int(witness["allocated_bytes"], "bad allocated bytes", minimum=size)
        require(allocated % 512 == 0, "allocated bytes not sector-aligned")
        digest = self.oracle[str(size)]["sha256"][kind]
        require(witness["sha256"] == digest and witness["expected_sha256"] == digest, "independent whole-file hash mismatch")
        require(witness["fsync_boundary"] == BOUNDARY, "persistence boundary text changed")
        path = safe_absolute_guest_path(witness["path"], "witness path")
        mib_count = size // MIB
        filename = f"{role}-{mib_count}m"
        origin = f"source-{mib_count}m" if role in ("snap", "snap2") else filename
        if self.guest_root is None:
            require(role == "source" and size == SIZES[0] and len(path.parts) >= 5, "cannot establish actual guest root")
            self.guest_root = path.parents[2]
            require(str(self.guest_root) != "/", "unsafe guest fixture root")
        require(path == self.guest_root / "volumes" / origin / filename, "wrong actual API backing path")
        dev = strict_int(witness["dev"], "bad device ID", minimum=1)
        inode = strict_int(witness["inode"], "bad inode", minimum=1)
        if self.common_dev is None:
            self.common_dev = dev
        require(dev == self.common_dev, "witnesses span different filesystems")
        identity = (dev, inode)
        if str(path) in self.identities:
            require(self.identities[str(path)] == identity, "surviving backing file inode changed")
        else:
            self.identities[str(path)] = identity
        self.checked.append({"size_bytes": size, "role": role, "kind": kind, "path": str(path), "logical_bytes": expected_size, "sha256": digest, "dev": dev, "inode": inode})
        return identity

    def group(self, witnesses, size, roles):
        require(type(witnesses) is list and len(witnesses) == len(roles), "wrong witness list length")
        identities = [self.one(witness, size, role, kind) for witness, (role, kind) in zip(witnesses, roles)]
        require(len(set(identities)) == len(identities), "live files alias the same inode")
        return identities


def roles(mutated, resized, deleted=False):
    return [entry for entry in [
        ("source", "source" if mutated else "original"),
        ("snap", "original"), ("snap2", "original"),
        ("a", "expanded_a" if resized else ("a" if mutated else "original")),
        ("b", "b" if mutated else "original")
    ] if not (deleted and entry[0] == "source")]


def verify_result(result, oracle, *, fixture_sha256, hash_utility_sha256):
    require(type(result) is dict, "result must be an object")
    require(result.get("status") == "pass" and result.get("failed_phase") is None, "fixture did not pass")
    require(result.get("official_source_commit") == PIN, "wrong source pin")
    require(result.get("fixture_executable_sha256") == fixture_sha256, "runtime fixture executable differs from compiled bytes")
    safe_absolute_guest_path(result.get("fixture_executable"), "fixture executable")
    require(result.get("sha256sum_executable") == "/usr/bin/sha256sum" and result.get("sha256sum_executable_sha256") == hash_utility_sha256, "wrong genuine hashing executable")
    require(result.get("timing_or_speedup_claimed") is False, "unexpected speed claim")
    require(result.get("runtime_scope") == "actual unchanged cubecow public library filesystem reflink; no VM/process-memory checkpoint, model rollout, optimizer or training throughput result", "scope changed")
    master = strict_int(result.get("master_pid"), "invalid master PID", 1)
    phases = result.get("separate_native_process_phases")
    require(type(phases) is list and len(phases) == 6, "need exactly six native phases")
    pids = set()
    witness = WitnessVerifier(oracle)
    for record, phase in zip(phases, PHASES):
        require(type(record) is dict and record.get("phase") == phase, "phase order/matrix mismatch")
        require(type(record.get("returncode")) is int and record["returncode"] == 0, "phase failed")
        spawned = strict_int(record.get("spawned_pid"), "invalid spawned PID", 1)
        require(spawned not in pids and spawned != master, "phases did not use distinct child processes")
        pids.add(spawned)
        require(record.get("stderr") == "", "unexpected child diagnostic output")
        require(type(record.get("stdout")) is str, "missing raw phase stdout")
        observed = parse_json(record["stdout"])
        require(observed == record.get("result"), "raw phase stdout differs from embedded result")
        require(observed.get("phase") == phase and strict_int(observed.get("pid"), "invalid reported PID", 1) == spawned, "child identity mismatch")
        require(observed.get("official_source_commit") == PIN and observed.get("actual_public_constructor") is True and observed.get("silently_skipped") is False, "phase source/constructor/skip evidence mismatch")
        cases = observed.get("cases")
        require(type(cases) is list and len(cases) == 2, "need exactly8+64MiB cases")
        for case, size in zip(cases, SIZES):
            require(type(case) is dict and strict_int(case.get("size_bytes"), "invalid case size") == size, "wrong case matrix")
            expected_origin = f"source-{size // MIB}m"
            if phase == "populate":
                witness.group(case.get("witnesses"), size, roles(False, False))
                require(case.get("source_fully_nonzero") is True and case.get("duplicate_name_rejected") is True, "population/duplicate witness missing")
                require(case.get("snapshot_of_snapshot_origin") == expected_origin, "snapshot origin flattening mismatch")
            elif phase == "mutate":
                witness.group(case.get("before"), size, roles(False, False))
                require(case.get("mutation_regions") == {"a": [0, REGION], "b": [size // 2, REGION], "source": [size - REGION, REGION]}, "wrong actual mutation regions")
                witness.group(case.get("witnesses"), size, roles(True, False))
            elif phase == "resize":
                witness.group(case.get("before"), size, roles(True, False))
                require(case.get("resize_result") == [size, size + MIB] and case.get("shrink_rejected") is True, "resize/shrink contract mismatch")
                witness.group(case.get("witnesses"), size, roles(True, True))
            elif phase == "delete-origin":
                witness.group(case.get("before"), size, roles(True, True))
                require(case.get("deleted_origin_not_found") is True and case.get("orphan_directory_preserved") is True, "origin deletion contract mismatch")
                witness.group(case.get("witnesses"), size, roles(True, True, True))
            elif phase == "recover-orphan":
                before = witness.group(case.get("before"), size, roles(True, True, True))
                require(case.get("deleted_origin_list_is_empty") is True, "official deleted-origin list contract mismatch")
                require(case.get("canonical_recovered_snapshot_names") == [f"snap-{size // MIB}m", f"snap2-{size // MIB}m"], "canonical snapshot identities mismatch")
                recovered = witness.one(case.get("new_branch_from_orphan"), size, "c", "original")
                require(recovered not in before, "recovered branch aliases a live backing file")
                witness.one(case.get("remaining_snapshot"), size, "snap2", "original")
                require(case.get("last_snapshot_reaped_origin_directory") is True, "last snapshot orphan reap missing")
                witness.group(case.get("survivors"), size, [("a", "expanded_a"), ("b", "b"), ("c", "original")])
            else:
                witness.group(case.get("before"), size, [("a", "expanded_a"), ("b", "b"), ("c", "original")])
                require(case.get("deleted_all_branches") is True, "branch deletion missing")
    require(len(witness.checked) == 92, "unexpected complete witness count")
    return {"status": "pass", "verified_native_phase_count": 6, "verified_distinct_child_pids": sorted(pids), "master_pid": master, "verified_sizes_bytes": list(SIZES), "verified_whole_file_witness_count": len(witness.checked), "verified_full_bytes_compared_total": sum(record["logical_bytes"] for record in witness.checked), "guest_fixture_root": str(witness.guest_root), "guest_filesystem_device": witness.common_dev, "unique_recorded_backing_paths": len(witness.identities), "witnesses": witness.checked, "oracle_implementation": "independent Python bytearray/division/XOR/patch+restore/hashlib; no Rust helper import/execution", "GPU_or_training_speed_claim": False}



def hash_pin(value, field):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), field + " requires a SHA256 declaration")


def canonical(path, *, must_exist=True):
    lexical = Path(path).absolute()
    require(not any(p.is_symlink() for p in (lexical, *lexical.parents)), "nonsymlink path and ancestors required")
    return lexical.resolve(strict=must_exist)


def verify_file(result_path, *, fixture_sha256, hash_utility_sha256, oracle=None):
    hash_pin(fixture_sha256, "fixture executable")
    hash_pin(hash_utility_sha256, "hash utility executable")
    path = canonical(result_path)
    require(path.is_file() and 0 < path.stat().st_size <= MAX_RESULT_BYTES, "bounded regular raw result required")
    raw = path.read_bytes()
    actual = parse_json(raw.decode("utf-8"))
    expected = independent_oracle() if oracle is None else oracle
    report = verify_result(actual, expected, fixture_sha256=fixture_sha256, hash_utility_sha256=hash_utility_sha256)
    report["raw_runtime_result_sha256"] = hashlib.sha256(raw).hexdigest()
    report["independent_expected_digests"] = expected
    report["operator_binary_sha256_declarations"] = {"fixture": fixture_sha256, "GNU_sha256sum": hash_utility_sha256}
    report["binary_declarations_are_execution_attestation"] = False
    report["verification_scope"] = "report contracts and independently expected bytes/hashes; actual native execution must be evidenced separately"
    report["verifier_source_sha256"] = sha_file(Path(__file__))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-fixture-sha256", required=True)
    parser.add_argument("--expected-hash-utility-sha256", required=True)
    args = parser.parse_args()
    result = canonical(args.result)
    output = canonical(args.output, must_exist=False)
    require(not output.exists() and output.parent.is_dir(), "fresh output under existing owned parent required")
    require(output != result and not output.is_relative_to(result) and not result.is_relative_to(output), "input/output overlap")
    started = time.perf_counter()
    report = verify_file(result, fixture_sha256=args.expected_fixture_sha256, hash_utility_sha256=args.expected_hash_utility_sha256)
    report["CPython_version"] = sys.version
    report["verification_wall_seconds"] = time.perf_counter() - started
    report["verification_time_is_not_snapshot_or_training_speed"] = True
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(output, flags, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output_sha256": sha_file(output), "raw_runtime_result_sha256": report["raw_runtime_result_sha256"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("verifier error: " + str(error), file=sys.stderr)
        raise SystemExit(1)
