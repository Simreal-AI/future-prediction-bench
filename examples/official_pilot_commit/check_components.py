#!/usr/bin/env python3
"""Exercise unchanged author Pilot-Commit components with real CPU batches."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

COMMIT = "6def20ea211fc936ed092a5a08624807c54df381"
PYTHON_COUNT = 450
PYTHON_SHA256 = "9ea69afcf931fb1a1c4a19573fc3150379632e7018ebf99885a3934ad93cd264"
DEPENDENCY_PINS = {
    "pyvers": {"version":"0.1.0", "file_count":9,"sha256":"53acaacdd0fee27e102753fe7d41f58452421339e59b71d0cfd275a0b1a65e74"},
    "tensordict": {"version":"0.9.1", "file_count":47,"sha256":"43cbe1005a2efe7f63d21863cdd8e4e4a711a6bdb9b00d4ec4bfe47311b77938"},
    "ray": {"version":"2.58.0", "file_count":2767,"sha256":"308f3875c8ef7e5f5b885c158183ed66d6e76ed49a5b5cdc1e737777e6657165"},
}


def sha_file(path: Path) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("regular_file_required")
        stream = os.fdopen(fd,"rb")
        fd = None
        with stream:
            while chunk := stream.read(1 << 20):
                digest.update(chunk)
    finally:
        if fd is not None:
            os.close(fd)
    return digest.hexdigest()


def directory_path(path: Path, *, writable=False) -> Path:
    absolute = Path(os.path.abspath(path))
    nearest = None
    for part in (absolute,*absolute.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("directory_ancestry_symlink_rejected")
        if not stat.S_ISDIR(info.st_mode):
            raise ValueError("directory_path_required")
        if nearest is None:
            nearest = info
    if writable and (nearest is None or nearest.st_uid != os.getuid()):
        raise ValueError("owned_writable_parent_required")
    return absolute


def checked_inputs(source: Path, output: Path, dependencies: list[Path]):
    source = directory_path(source).resolve(strict=True)
    dependency_paths = [directory_path(path).resolve(strict=True) for path in dependencies]
    output = directory_path(output,writable=True)
    all_paths = [source,output,*dependency_paths]
    for left,right in itertools.combinations(all_paths,2):
        if left.is_relative_to(right) or right.is_relative_to(left):
            raise ValueError("source_output_dependency_overlap_rejected")
    if os.path.lexists(output):
        raise ValueError("new_nonexistent_output_directory_required")
    source_state(source)
    pins = dependency_state(dependency_paths)
    return source,output,dependency_paths,pins


def dependency_state(dependencies: list[Path]) -> dict:
    observed = {}
    for root in dependencies:
        for distribution in importlib.metadata.distributions(path=[str(root)]):
            name = distribution.metadata["Name"].lower()
            if name not in DEPENDENCY_PINS:
                continue
            if name in observed:
                raise ValueError("duplicate_pinned_dependency_distribution")
            digest = hashlib.sha256()
            count = 0
            for relative in sorted(distribution.files,key=str):
                if relative.suffix == ".pyc":
                    continue
                path = distribution.locate_file(relative)
                if not path.resolve().is_relative_to(root):
                    raise ValueError("dependency_file_outside_input_directory")
                directory_path(path.parent)
                digest.update(str(relative).encode())
                digest.update(b"\0")
                digest.update(bytes.fromhex(sha_file(path)))
                count += 1
            actual = {"version":distribution.version,"file_count":count,"sha256":digest.hexdigest()}
            if actual != DEPENDENCY_PINS[name]:
                raise ValueError(f"dependency_pin_mismatch: {name} {actual}")
            observed[name] = actual
    if set(observed) != set(DEPENDENCY_PINS):
        raise ValueError("all_pinned_dependencies_required")
    return observed


def source_state(source: Path) -> dict:
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    clean = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=all"], text=True)
    digest = hashlib.sha256()
    files = sorted(source.glob("**/*.py"))
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode())
        digest.update(b"\0")
        directory_path(path.parent)
        digest.update(bytes.fromhex(sha_file(path)))
    observed = {"commit": revision, "python_file_count": len(files), "python_sha256": digest.hexdigest(), "git_clean": not clean}
    if observed != {"commit": COMMIT, "python_file_count": PYTHON_COUNT, "python_sha256": PYTHON_SHA256, "git_clean": True}:
        raise ValueError(f"official_source_pin_or_clean_state_mismatch: {observed}")
    return observed


def run(source: Path, output: Path, dependencies: list[Path], official_tests: bool, dependency_pins: dict) -> dict:
    started = time.perf_counter()
    before = source_state(source)
    if any(name == "verl" or name.startswith("verl.") or name.startswith("recipe.pc") or name in {"ray","tensordict","pyvers"} for name in sys.modules):
        raise ValueError("fresh_process_without_preloaded_author_components_required")
    sys.path[:0] = [str(source),*[str(path) for path in dependencies]]
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(output/"uncached-bytecode")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONPYCACHEPREFIX"] = sys.pycache_prefix
    # Standard package imports execute the original verl initializer. No
    # extracted functions, synthetic modules, model or infrastructure doubles.
    import numpy as np
    import torch
    from recipe.pc.utils import select_prompts, extract_original_prompts
    from recipe.pc.replay_buffer import ReplayBuffer
    from verl.protocol import DataProto
    from tensordict import TensorDict
    assert Path(inspect.getfile(select_prompts)).resolve() == source/"recipe/pc/utils.py"
    assert Path(inspect.getfile(ReplayBuffer)).resolve() == source/"recipe/pc/replay_buffer.py"
    assert Path(inspect.getfile(DataProto)).resolve() == source/"verl/protocol.py"
    for name in DEPENDENCY_PINS:
        path = Path(sys.modules[name].__file__).resolve()
        if not any(path.is_relative_to(root) for root in dependencies):
            raise ValueError("pinned_dependency_imported_outside_declared_inputs: "+name)

    report = {
        "schema_version": 1,
        "source": before,
        "dependency_pins": dependency_pins,
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "unchanged_official_selection_and_replay_components_with_real_CPU_batches",
        "license": "Apache-2.0 plus NOTICE and original third-party notices",
        "platform": platform.platform(),
        "python": sys.version,
        "inference_executed": False,
        "optimizer_update_executed": False,
        "gpu_speedup_measured": False,
        "full_trainer_executed": False,
        "fixtures": {},
        "upstream_boundary_observations": {},
    }
    fixtures = report["fixtures"]
    observations = report["upstream_boundary_observations"]
    prompt_indices = np.repeat(np.arange(100, 107), 8)
    rewards = np.concatenate([np.array([1.0] * k + [0.0] * (8-k)) for k in (0, 1, 2, 4, 6, 7, 8)])
    selection = select_prompts(prompt_indices, rewards, diversity_threshold_upper=.25,
                               diversity_threshold_lower=.125, exclude_threshold_upper=1.0)
    normalized = {key: [int(x) for x in value] for key, value in selection.items()}
    expected = {"keep": [101, 102, 103, 104], "too_correct": [105, 106], "too_incorrect": [100], "exclude_too_easy": [106]}
    assert normalized == expected
    fixtures["selection_with_inclusive_threshold_boundaries"] = {"rates": [0, .125, .25, .5, .75, .875, 1], "actual": normalized, "expected": expected, "passed": True}
    empty = select_prompts(np.array([], dtype=np.int64), np.array([], dtype=np.float64))
    assert all(value == [] for value in empty.values())
    fixtures["empty_selection"] = {"actual": empty, "passed": True}

    source_rows = {}
    def make_batch(uid, binary, revision="policy-7"):
        n = len(binary)
        row_ids = np.arange(uid*100, uid*100+n, dtype=np.int64)
        response = torch.tensor([[uid, int(row)] for row in row_ids], dtype=torch.int64)
        original = np.tile(np.array([uid, 201, 202], dtype=np.int64), (n, 1))
        batch = DataProto.from_dict(
            tensors={"responses": response},
            non_tensors={"index": np.full(n, uid, dtype=np.int64), "uid": np.array([f"prompt-{uid}"]*n, dtype=object),
                         "acc": np.asarray(binary, dtype=np.float64), "row_id": row_ids,
                         "policy_revision": np.array([revision]*n, dtype=object),
                         "input_ids_original": original,
                         "attention_mask_original": np.ones_like(original),
                         "position_ids_original": np.tile(np.arange(3, dtype=np.int64), (n, 1)),
                         "raw_prompt_ids": original.copy()},
            meta_info={"fixture": "CPU_owned_rows"},
        )
        assert isinstance(batch.batch, TensorDict) and all(tensor.device.type == "cpu" for tensor in batch.batch.values())
        for i, row_id in enumerate(row_ids):
            source_rows[int(row_id)] = {"uid": uid, "reward": float(binary[i]), "revision": revision, "tokens": response[i].tolist()}
        return batch

    def aligned_rows(batch):
        result = []
        for i, row_id in enumerate(batch.non_tensor_batch["row_id"]):
            expected_row = source_rows[int(row_id)]
            actual = {"uid": int(batch.non_tensor_batch["index"][i]), "reward": float(batch.non_tensor_batch["acc"][i]),
                      "revision": str(batch.non_tensor_batch["policy_revision"][i]), "tokens": batch.batch["responses"][i].tolist()}
            assert actual == expected_row, (actual, expected_row)
            assert str(batch.non_tensor_batch["uid"][i]) == f"prompt-{actual['uid']}"
            result.append({"row_id": int(row_id), **actual})
        return result

    buffer = ReplayBuffer(max_size=10, max_off_steps=2)
    for uid, binary, step in [(10, [0,1,0,1], 1), (20, [0,0,1,1], 2), (30, [0,0,0,1], 3), (40, [1,1,1,1], 4)]:
        buffer.add([make_batch(uid, binary)], step=step)
    sampled, when_added = buffer.sample(size=2, n_responses=2, prompt_sampling_strategy="max_variance", response_sampling_strategy="max_variance")
    actual_ids = sampled.non_tensor_batch["index"].tolist()
    assert actual_ids == [20,20,10,10] and when_added == [2,1]
    assert len(sampled) == 4 and len(buffer) == 2 and set(buffer.buffer) == {30,40}
    assert all(sampled.non_tensor_batch["from_buffer"])
    for uid in (20,10):
        assert sorted(sampled.non_tensor_batch["acc"][sampled.non_tensor_batch["index"] == uid].tolist()) == [0.,1.]
    fixtures["variance_ranked_real_batch_and_step_provenance"] = {"actual_prompt_ids": actual_ids, "when_added": when_added, "rows": aligned_rows(sampled), "remaining_ids": sorted(buffer.buffer), "passed": True}

    stale = ReplayBuffer(max_size=10, max_off_steps=2)
    for uid, step in [(50,1), (60,2), (70,3)]:
        stale.add([make_batch(uid, [0,1])], step=step)
    evicted_at_4 = stale.flush(current_step=4)
    ids_at_4 = sorted(stale.buffer)
    evicted_at_5 = stale.flush(current_step=5)
    ids_at_5 = sorted(stale.buffer)
    enforced = stale.flush(current_step=5, enforce=True)
    assert (evicted_at_4, ids_at_4, evicted_at_5, ids_at_5, enforced, len(stale)) == (1,[60,70],1,[70],1,0)
    fixtures["strict_step_staleness_boundary"] = {"max_off_steps": 2, "at_step_4": {"evicted": evicted_at_4, "remaining_ids": ids_at_4}, "at_step_5": {"evicted": evicted_at_5, "remaining_ids": ids_at_5}, "enforced_evicted": enforced, "passed": True}

    popped = ReplayBuffer(max_size=10, max_off_steps=2)
    popped.add([make_batch(80,[0,1]), make_batch(90,[1,0])], step=5)
    pop_batch, pop_steps = popped.pop([90,80])
    assert pop_batch.non_tensor_batch["index"].tolist() == [90,90,80,80] and pop_steps == [5,5] and popped.is_empty()
    fixtures["explicit_pop_order_and_alignment"] = {"rows": aligned_rows(pop_batch), "when_added": pop_steps, "passed": True}
    empty_pair = popped.sample(size=1,n_responses=2)
    assert empty_pair == ([],[])
    fixtures["empty_replay_result"] = {"actual": [[],[]], "passed": True}

    prompt_batch = DataProto.concat([make_batch(91,[0,1]),make_batch(92,[1,0])])
    reduced = extract_original_prompts(prompt_batch, ["uid", "index", "policy_revision", "raw_prompt_ids"])
    assert len(reduced) == 2
    assert reduced.non_tensor_batch["index"].tolist() == [91,92]
    assert reduced.batch["input_ids"].tolist() == [[91,201,202],[92,201,202]]
    assert reduced.non_tensor_batch["policy_revision"].tolist() == ["policy-7", "policy-7"]
    assert all(key not in prompt_batch.non_tensor_batch for key in ("input_ids_original","attention_mask_original","position_ids_original"))
    fixtures["original_prompt_extraction_and_UID_reduction"] = {"prompt_ids": reduced.non_tensor_batch["index"].tolist(), "tokens": reduced.batch["input_ids"].tolist(), "policy_revisions": reduced.non_tensor_batch["policy_revision"].tolist(), "original_fields_consumed": True, "passed": True}

    # Exhaustive independent combinatorial oracle; integer conversion is used
    # solely to inspect mathematical indices returned by the helper, never to
    # repair or mask the real DataProto sampling path below.
    exhaustive = []
    helper = ReplayBuffer(max_size=10,max_off_steps=2)
    for n in range(1,9):
        for ones in range(n+1):
            values = np.array([1.]*ones+[0.]*(n-ones))
            for m in range(1,n+1):
                idx = helper._maxvar_downsample_binary(values,m,seed=123)
                assert len(idx) == m and len(np.unique(idx)) == m
                assert np.array_equal(idx,np.floor(idx)) and all(0 <= i < n for i in idx)
                chosen = values[idx.astype(np.int64)]
                numerator = int(chosen.sum())*(m-int(chosen.sum()))
                oracle = max(sum(values[list(combination)])*(m-sum(values[list(combination)])) for combination in itertools.combinations(range(n),m))
                assert numerator == oracle
                exhaustive.append({"available":n,"ones":ones,"selected":m,"actual_variance_numerator":numerator,"oracle_variance_numerator":int(oracle),"index_dtype":str(idx.dtype)})
    assert len(exhaustive) == 240
    fixtures["binary_maximum_variance_exhaustive_oracle"] = {"trial_count": len(exhaustive), "trials": exhaustive, "passed": True}

    # Preserve real upstream limitations as observations, not as successful
    # trainer tests or justification to silently change original code.
    invalid_selection = select_prompts(np.array([1,1]),np.array([np.nan,0.]))
    assert invalid_selection["keep"] == [1]
    observations["non_finite_rewards_are_not_rejected"] = {"actual_keep": [int(x) for x in invalid_selection["keep"]], "requires_client_guard": True}
    capacity = ReplayBuffer(max_size=1,max_off_steps=2)
    capacity.add([make_batch(93,[0,1]),make_batch(94,[0,1])],step=1)
    assert len(capacity) == 2
    observations["max_size_not_enforced_by_add"] = {"configured_max_size":1,"actual_prompt_count":len(capacity),"requires_client_guard":True}
    bypass = ReplayBuffer(max_size=10,max_off_steps=2)
    bypass.add([make_batch(95,[0,1,0,1]),make_batch(96,[0,1,0,1])],step=1)
    bypass_batch,bypass_steps = bypass.sample(size=2,n_responses=2,prompt_sampling_strategy="max_variance",response_sampling_strategy="max_variance")
    assert len(bypass_batch) == 8 and bypass_steps == [1,1]
    observations["sample_all_branch_returns_all_responses"] = {"requested_prompt_count":2,"requested_responses_per_prompt":2,"expected_rows_from_request":4,"actual_rows":len(bypass_batch),"rows":aligned_rows(bypass_batch),"requires_client_guard":True}
    collapsed = ReplayBuffer(max_size=10,max_off_steps=2)
    collapsed.add([make_batch(97,[0,0,0,0])],step=1)
    collapsed.add([make_batch(98,[1,1,1,1])],step=2)
    try:
        collapsed.sample(size=1,n_responses=2,prompt_sampling_strategy="max_variance",response_sampling_strategy="max_variance")
    except (IndexError,TypeError) as exc:
        observations["collapsed_binary_pool_yields_float_index_failure"] = {"exception_type":type(exc).__name__,"message":str(exc),"requires_client_guard":True}
    else:
        raise AssertionError("expected_real_float_index_failure_not_observed")
    missing = ReplayBuffer(max_size=10,max_off_steps=2)
    try:
        missing.pop([999])
    except AssertionError as exc:
        observations["missing_pop_is_rejected"] = {"exception_type":type(exc).__name__,"message":str(exc)}
    else:
        raise AssertionError("missing_pop_was_not_rejected")
    revision_age = ReplayBuffer(max_size=10,max_off_steps=2)
    revision_age.add([make_batch(99,[0,1],revision="policy-6")],step=2)
    assert revision_age.flush(current_step=3) == 0
    older_batch,older_steps = revision_age.pop([99])
    assert older_batch.non_tensor_batch["policy_revision"].tolist() == ["policy-6","policy-6"] and older_steps == [2]
    observations["step_age_does_not_filter_policy_revisions"] = {"consumer_revision":"policy-7","retained_revision":"policy-6","current_step":3,"added_step":2,"rows":aligned_rows(older_batch),"requires_consumer_policy_provenance_contract":True,"upstream_off_policy_algorithm_not_executed":True}

    modules = {}
    for name,module in sorted(sys.modules.items()):
        file = getattr(module,"__file__",None)
        if file:
            path = Path(file).resolve()
            if path.is_relative_to(source) and path.suffix == ".py":
                modules[name] = {"relative_path":path.relative_to(source).as_posix(),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
    assert "verl" in modules and "verl.protocol" in modules and "recipe.pc.utils" in modules and "recipe.pc.replay_buffer" in modules
    report["loaded_original_modules"] = modules
    report["loaded_original_module_count"] = len(modules)
    report["observed_dependency_versions"] = {name:importlib.metadata.version(name) for name in ["torch","numpy","tensordict","pyvers","ray","transformers","pytest"]}
    if official_tests:
        command = [sys.executable,"-m","pytest",str(source/"tests/test_protocol_on_cpu.py"),"-q","-p","no:cacheprovider",f"--junitxml={output/'official-tests.xml'}"]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([str(source),*[str(path) for path in dependencies]])
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        test_start = time.perf_counter()
        result = subprocess.run(command,cwd=source,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env,timeout=120)
        log = output/"official-tests.log"
        log.write_text(result.stdout,encoding="utf-8")
        report["official_tests"] = {"command":command,"returncode":result.returncode,"wall_ms":(time.perf_counter()-test_start)*1000,"log_sha256":hashlib.sha256(log.read_bytes()).hexdigest()}
        if result.returncode:
            raise RuntimeError(f"official_protocol_tests_failed: {result.stdout[-4000:]}")
        suite = ET.parse(output/"official-tests.xml").getroot().find("testsuite")
        counts = {key:int(suite.attrib[key]) for key in ["tests","failures","errors","skipped"]}
        assert counts == {"tests":18,"failures":0,"errors":0,"skipped":0}
        report["official_tests"]["counts"] = counts
    report["source_preservation"] = {"before":before,"after":source_state(source)}
    report["dependency_preservation"] = {"before":dependency_pins,"after":dependency_state(dependencies)}
    report["wall_ms"] = (time.perf_counter()-started)*1000
    report["passed"] = True
    report["pass_meaning"] = "documented_positive_component_contract_and_expected_boundary_observations; not full trainer or all inputs supported"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--dependency",type=Path,action="append",required=True,help="Existing isolated pinned dependency directory; may repeat.")
    parser.add_argument("--official-tests",action="store_true")
    args = parser.parse_args()
    source,output,dependencies,pins = checked_inputs(args.source,args.output,args.dependency)
    output.mkdir(parents=True,exist_ok=False)
    try:
        report = run(source,output,dependencies,args.official_tests,pins)
    except Exception as exc:
        report = {"passed":False,"exception_type":type(exc).__name__,"message":str(exc),"traceback_tail":traceback.format_exc()[-8000:]}
        (output/"result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(json.dumps({"passed":False,"exception_type":type(exc).__name__,"message":str(exc)[-2000:]}))
        return 2
    (output/"result.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({"passed":report["passed"],"fixture_groups":len(report["fixtures"]),"exhaustive_binary_trials":240,"upstream_observations":len(report["upstream_boundary_observations"]),"official_tests":report.get("official_tests",{})},indent=2))
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(json.dumps({"passed":False,"exception_type":type(exc).__name__,"message":str(exc)[-2000:]}))
        exit_code = 2
    raise SystemExit(exit_code)
