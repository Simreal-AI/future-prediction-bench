"""Execute the released VIP Allocator module using real NumPy/SciPy."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess

COMMIT = "f7bd18915467f50a0d8565b4f16afaef0741a96f"
SOURCE = "src/vip/allocation/bernoulli_allocation.py"


def check(checkout):
    checkout = Path(checkout).resolve()
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True)
    if head != COMMIT or dirty:
        raise ValueError("clean pinned VIP checkout required")
    path = checkout / SOURCE
    spec = importlib.util.spec_from_file_location("fpb_reviewed_official_vip_allocator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    probabilities = [0.01, 0.1, 0.2, 0.4, 0.5, 0.8, 0.95, 0.99]
    allocator = module.Allocator(allocation_rule="vip", lower=4, upper=12, budget_per_question=8)
    with contextlib.redirect_stdout(io.StringIO()):
        allocation = allocator.allocate(probabilities)
    if sum(allocation) != 64 or any(type(n) is not int or not 4 <= n <= 12 for n in allocation):
        raise RuntimeError("upstream_budget_bounds_or_total_failed")
    return {"schema_version": "official-vip-allocator-cpu-v1",
        "upstream": {"repository": "https://github.com/HieuNT91/VIP", "commit": head,
                     "source_path": SOURCE, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()},
        "execution_scope": "unmodified Allocator with real NumPy/SciPy",
        "success_probability_inputs": probabilities, "allocation": allocation,
        "total_rollouts": sum(allocation), "uniform_rollouts": [8] * 8,
        "upstream_accuracy_clamp_at_0_8": True,
        "success_predictor_trained": False, "model_rollout": False, "optimizer": False,
        "trainer_entrypoint_executed": False, "dependency_doubles": []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("new output required")
    result = check(args.checkout)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
