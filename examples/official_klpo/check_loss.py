"""Execute pinned, unmodified KLPO loss and native loss adapter on CPU.

This checks an author's fixed complete-trajectory fixture and exhaustive
independent auxiliary draws. It does not collect LLM rollouts, infer whether
arbitrary records are independent, or launch the Molt trainer/backend.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

COMMIT = "30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696"
SOURCE_FILES = ("klpo/__init__.py", "klpo/loss.py", "klpo/_validation.py",
                "klpo/molt.py", "klpo/budget.py", "tests/test_mc.py")


def load_upstream(checkout):
    checkout = Path(checkout).resolve()
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(checkout), "status", "--porcelain"], text=True)
    if head != COMMIT or dirty:
        raise ValueError("KLPO requires the clean pinned author checkout")
    if any(name == "klpo" or name.startswith("klpo.") for name in sys.modules):
        raise RuntimeError("Load KLPO in a fresh process to prevent module shadowing")
    for name in SOURCE_FILES:
        path = checkout / name
        if not path.is_file() or path.is_symlink():
            raise ValueError("KLPO source must be a regular file")
    sys.path.insert(0, str(checkout))
    module = importlib.import_module("klpo")
    adapter_module = importlib.import_module("klpo.molt")
    for name in ("klpo", "klpo.loss", "klpo._validation", "klpo.molt", "klpo.budget"):
        expected = checkout / ("klpo/__init__.py" if name == "klpo" else name.replace(".", "/") + ".py")
        if Path(sys.modules[name].__file__).resolve() != expected:
            raise RuntimeError("KLPO import origin mismatch")
    return module, adapter_module, {
        "repository": "https://github.com/yifanzhang-pro/KLPO", "commit": head,
        "source_sha256": {name: hashlib.sha256((checkout / name).read_bytes()).hexdigest()
                          for name in SOURCE_FILES},
        "modified_upstream_files": [], "dependency_doubles": [],
        "loss_execution": "normal import of unmodified klpo package",
        "native_adapter_execution": "klpo.molt.KLPOLoss only; no Molt backend imported",
    }


def fixture():
    """Author test_mc.py fixture: terminal two-token path, historical sampler."""
    import torch
    theta = torch.tensor([.5, -.7, .2], dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[[1., -.4, .2], [-.2, .3, 1.]],
                            [[.3, 1., -.5], [.7, -.3, .4]]], dtype=theta.dtype)
    current_full = (weights @ theta).log_softmax(-1)[None]
    sampler = torch.tensor([[[.25, .75], [.6, .4]]], dtype=theta.dtype)
    actions = torch.tensor([[1, 0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    current_actions = current_full.gather(-1, actions[..., None]).squeeze(-1)
    behavior_actions = sampler.log().gather(-1, actions[..., None]).squeeze(-1)
    terminal_reward = torch.tensor([.35], dtype=theta.dtype)
    return theta, current_full, sampler, (current_actions, behavior_actions, terminal_reward, mask)


def check_gradients(module, adapter_module):
    import torch
    theta, p_log, q, args = fixture()
    beta = .6
    rows = []
    for route, samples in (("token", 1), ("sequence", 2)):
        if route == "token":
            exact, _ = module.klpo_token_loss(*args, kl_estimator="full",
                conditional_log_probs=p_log, behavior_conditional_log_probs=q.log(), beta=beta)
        else:
            exact, _ = module.klpo_sequence_full_loss(*args, full_log_probs=p_log,
                behavior_full_log_probs=q.log(), beta=beta)
        expected_surrogate = p_log.new_zeros(())
        weight_sum = p_log.new_zeros(())
        first_bank = None
        # Cartesian enumeration includes repeats and uses historical q weights.
        # This is an exact expectation over draws independent of the fixed
        # complete rollout, not a statistical assertion about arbitrary logs.
        combinations = list(product(range(2), repeat=2 * samples))
        for values in combinations:
            ids = torch.tensor(values).reshape(1, 2, samples)
            records = {"mc_log_probs": p_log.gather(-1, ids),
                       "behavior_mc_log_probs": q.log().gather(-1, ids)}
            probability = records["behavior_mc_log_probs"].exp().prod()
            if route == "token":
                loss, _ = module.klpo_token_loss(*args, **records, beta=beta)
            else:
                loss, _ = module.klpo_sequence_mc_loss(*args, **records, beta=beta)
            expected_surrogate = expected_surrogate + probability * loss
            weight_sum = weight_sum + probability
            if first_bank is None:
                first_bank = (records, loss)
        expected_gradient, = torch.autograd.grad(exact, theta, retain_graph=True)
        mc_gradient, = torch.autograd.grad(expected_surrogate, theta, retain_graph=True)
        torch.testing.assert_close(mc_gradient, expected_gradient, atol=1e-12, rtol=1e-11)
        torch.testing.assert_close(weight_sum, weight_sum.new_ones(()))
        records, direct = first_bank
        native = adapter_module.KLPOLoss(beta=beta, route=route, kl_estimator="mc")
        native_loss, *telemetry = native(args[0], torch.full_like(args[1], -500.),
            torch.full_like(args[0], 99.), action_mask=args[3], rollout_log_probs=args[1],
            rewards=args[2], global_batch_size=1, dp_size=1,
            kl_log_probs=records["mc_log_probs"], behavior_kl_log_probs=records["behavior_mc_log_probs"])
        direct_gradient, = torch.autograd.grad(direct, theta, retain_graph=True)
        native_gradient, = torch.autograd.grad(native_loss, theta, retain_graph=True)
        torch.testing.assert_close(native_gradient, direct_gradient)
        if any(value.requires_grad for value in telemetry):
            raise RuntimeError("native telemetry unexpectedly retains gradients")
        rows.append({"route": route, "mc_samples_per_prefix": samples,
            "exhaustively_enumerated_auxiliary_banks": len(combinations),
            "expected_mc_gradient": mc_gradient.tolist(),
            "full_kl_gradient": expected_gradient.tolist(),
            "maximum_absolute_gradient_difference": float((mc_gradient - expected_gradient).abs().max()),
            "native_loss_adapter_gradient_matches_direct_loss": True,
            "native_ignores_ppo_old_logprobs_and_group_advantages": True,
            "complete_fixed_terminal_fixture": True,
            "real_torch_autograd": True})
    return rows


def check_guards(module, adapter_module):
    import torch
    _, p_log, q, args = fixture()
    ids = torch.tensor([[[0], [1]]])
    records = {"mc_log_probs": p_log.gather(-1, ids),
               "behavior_mc_log_probs": q.log().gather(-1, ids)}
    bad_behavior = args[1].clone()
    bad_behavior[0, 0] = float("nan")
    bad_records = records["behavior_mc_log_probs"].clone()
    bad_records[0, 0, 0] = float("inf")
    cases = {
        "nonboolean_action_mask": lambda: module.klpo_token_loss(*args[:3], args[3].float(), **records),
        "no_policy_tokens": lambda: module.klpo_token_loss(*args[:3], torch.zeros_like(args[3]), **records),
        "nonfinite_active_sampler_logprob": lambda: module.klpo_token_loss(args[0], bad_behavior, *args[2:], **records),
        "nonfinite_auxiliary_sampler_logprob": lambda: module.klpo_token_loss(*args, mc_log_probs=records["mc_log_probs"], behavior_mc_log_probs=bad_records),
        "missing_auxiliary_bank": lambda: module.klpo_token_loss(*args),
        "sequence_requires_two_independent_columns": lambda: module.klpo_sequence_mc_loss(*args, **records),
        "native_requires_actual_rollout_logprobs": lambda: adapter_module.KLPOLoss()(args[0], args[1], None,
            action_mask=args[3], rollout_log_probs=None, rewards=args[2], global_batch_size=1),
        "native_requires_global_trajectory_count": lambda: adapter_module.KLPOLoss()(args[0], args[1], None,
            action_mask=args[3], rollout_log_probs=args[1], rewards=args[2]),
    }
    results = {}
    for name, call in cases.items():
        try:
            call()
        except ValueError as exc:
            results[name] = {"rejected": True, "exception": "ValueError", "reason": str(exc)}
        else:
            raise RuntimeError("expected upstream rejection missing: " + name)
    return results


def run(checkout):
    import torch
    module, adapter_module, upstream = load_upstream(checkout)
    return {"schema_version": "official-klpo-loss-cpu-v1", "upstream": upstream,
        "torch_version": torch.__version__, "device": "cpu",
        "fixture": "author tests/test_mc.py two-token terminal path; exhaustive independent auxiliary draws",
        "loss_and_gradients": check_gradients(module, adapter_module),
        "upstream_guard_checks": check_guards(module, adapter_module),
        "offpolicy_current_and_sampler_differ": True,
        "complete_trajectory_not_inferred_from_mask": True,
        "arbitrary_sampling_independence_validated": False,
        "arbitrary_history_transform_matching_validated": False,
        "llm_rollout": False, "molt_trainer": False, "distributed_backend": False,
        "gpu_training": False, "inference_or_training_speedup_measured": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        parser.error("output must be a new file")
    if output.resolve().is_relative_to(Path(args.checkout).resolve()):
        parser.error("output must be outside the reviewed upstream checkout")
    result = run(args.checkout)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output), "upstream_commit": result["upstream"]["commit"],
        "loss_and_gradients": result["loss_and_gradients"],
        "guard_checks": len(result["upstream_guard_checks"])}, indent=2))
