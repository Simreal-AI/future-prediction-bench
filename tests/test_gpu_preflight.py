"""The GPU preflight checks offline inputs; it never proves that training ran."""

import copy
import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from future_prediction_bench.gpu_preflight import audit_gpu_job, main


def _write_ref(root, name, content):
    data = content.encode() if isinstance(content, str) else json.dumps(content, sort_keys=True).encode()
    (root / name).write_bytes(data)
    return {"path": name, "sha256": hashlib.sha256(data).hexdigest()}


def _task(index, split):
    return {"task_id": f"{split}-task-{index}", "cluster_id": f"cluster-{split}-{index}",
            "repository": f"https://github.com/example/{split}-repo-{index}",
            "source_commit": f"{index:x}" * 40, "source_sha256": f"{index:x}" * 64,
            "verifier_sha256": "b" * 64, "environment_image_sha256": "sha256:" + "c" * 64,
            "license": "MIT", "is_fixture": False}


def _job(root):
    train = [_task(1, "train"), _task(2, "train")]
    dev = [_task(3, "dev")]
    train_ref = _write_ref(root, "train.json", {"schema_version": "qwen8b-coding-manifest-0.1",
                                               "split": "train", "track": "coding", "records": train})
    dev_ref = _write_ref(root, "dev.json", {"schema_version": "qwen8b-coding-manifest-0.1",
                                           "split": "dev", "track": "coding", "records": dev})
    template = "{{ messages | tojson }}"
    model_revision = "e" * 40
    policy_revision = "d" * 64
    model_config = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
                    "hidden_size": 4096, "num_hidden_layers": 36,
                    "num_attention_heads": 32, "num_key_value_heads": 8,
                    "head_dim": 128, "vocab_size": 151936, "torch_dtype": "bfloat16"}
    samples = []
    for task in train:
        for sample_index in range(2):
            samples.append({"task_id": task["task_id"], "source_commit": task["source_commit"],
                            "group_id": "group-" + task["task_id"], "sample_index": sample_index,
                            "group_size": 2, "reward": float(sample_index), "input_ids": [1, 2, 3],
                            "assistant_loss_mask": [0, 0, 1],
                            "behavior_token_logprobs": [None, None, -.7],
                            "teacher_forced_token_logprobs": [None, None, -.71],
                            "truncated": False})
    rollouts = {"schema_version": "qwen8b-tokenized-rollouts-0.1",
                "record_type": "sampled_token_trajectories", "trainer_ready": False,
                "model_revision": model_revision, "tokenizer_revision": model_revision,
                "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                "enable_thinking": False, "policy_revision": policy_revision,
                "logprobs_mode": "processed_logprobs", "train_manifest_sha256": train_ref["sha256"],
                "samples": samples}
    plan = {"schema_version": "qwen8b-gpu-job-preflight-0.1",
            "model": {"repository": "Qwen/Qwen3-8B", "revision": model_revision,
                      "tokenizer_revision": model_revision,
                      "config": _write_ref(root, "config.json", model_config),
                      "tokenizer_config": _write_ref(root, "tokenizer_config.json", {"chat_template": template}),
                      "chat_template_sha256": rollouts["chat_template_sha256"],
                      "enable_thinking": False},
            "runtime": {"engine": "vllm", "version": "0.8.5", "logprobs_mode": "processed_logprobs",
                        "image_sha256": "sha256:" + "f" * 64,
                        "max_model_len": 4096, "max_completion_len": 256},
            "dataset": {"train": train_ref, "dev": dev_ref},
            "rollouts": _write_ref(root, "rollouts.json", rollouts),
            "trainer": {"framework": "trl", "version": "0.20.0",
                        "entrypoint": _write_ref(root, "train.py", "raise SystemExit('test only')\n"),
                        "algorithm": "grpo", "optimizer": "adamw_torch_fused",
                        "learning_rate": 1e-5, "max_steps": 10, "policy_revision": policy_revision,
                        "max_policy_lag": 0, "beta": 0, "scale_rewards": "none",
                        "loss_type": "dr_grpo"}}
    _write_ref(root, "plan.json", plan)
    return plan, rollouts, train, dev


class GPUPreflightTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.plan, self.rollouts, self.train, self.dev = _job(self.root)

    def audit(self, *, plan=None, rollouts=None, train=None, dev=None):
        plan = copy.deepcopy(self.plan if plan is None else plan)
        if train is not None:
            plan["dataset"]["train"] = _write_ref(self.root, "train.json", {
                "schema_version": "qwen8b-coding-manifest-0.1", "split": "train",
                "track": "coding", "records": train})
        if dev is not None:
            plan["dataset"]["dev"] = _write_ref(self.root, "dev.json", {
                "schema_version": "qwen8b-coding-manifest-0.1", "split": "dev",
                "track": "coding", "records": dev})
        if rollouts is not None:
            plan["rollouts"] = _write_ref(self.root, "rollouts.json", rollouts)
        _write_ref(self.root, "plan.json", plan)
        return audit_gpu_job(self.root / "plan.json")

    def test_valid_input_contract_still_never_claims_training(self):
        report = self.audit()
        self.assertEqual(report["errors"], [])
        self.assertTrue(report["input_contract_passed"])
        self.assertFalse(report["trainer_ready"])
        self.assertFalse(report["optimizer_update_observed"])
        self.assertEqual((report["train_tasks"], report["dev_tasks"], report["complete_groups"]), (2, 1, 2))
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--plan", str(self.root / "plan.json")]), 0)
        self.assertFalse(json.loads(output.getvalue())["gpu_executed"])

    def test_rejects_unpinned_or_mismatched_model_and_runtime(self):
        changed = copy.deepcopy(self.plan)
        changed["model"]["tokenizer_revision"] = "a" * 40
        self.assertEqual(self.audit(plan=changed)["errors"], ["tokenizer_model_revision_mismatch"])
        changed = copy.deepcopy(self.plan)
        changed["runtime"]["logprobs_mode"] = "raw_logprobs"
        self.assertEqual(self.audit(plan=changed)["errors"], ["unsupported_sampling_logprob_contract"])
        changed = copy.deepcopy(self.plan)
        changed["runtime"]["version"] = "0.8.4"
        self.assertEqual(self.audit(plan=changed)["errors"], ["qwen3_unsupported_vllm_version"])
        changed = copy.deepcopy(self.plan)
        changed["model"]["chat_template_sha256"] = "a" * 64
        self.assertEqual(self.audit(plan=changed)["errors"], ["chat_template_mismatch"])

    def test_rejects_manifest_hash_and_leakage(self):
        (self.root / "train.json").write_text("{}")
        self.assertEqual(self.audit()["errors"], ["train_manifest:file_hash_mismatch"])
        self.plan["dataset"]["train"] = _write_ref(self.root, "train.json", {
            "schema_version": "qwen8b-coding-manifest-0.1", "split": "train",
            "track": "coding", "records": self.train})
        dev = copy.deepcopy(self.dev)
        dev[0]["repository"] = self.train[0]["repository"]
        self.assertEqual(self.audit(dev=dev)["errors"], ["train_dev_repository_overlap"])
        dev = copy.deepcopy(self.dev)
        dev[0]["source_sha256"] = self.train[0]["source_sha256"]
        self.assertEqual(self.audit(dev=dev)["errors"], ["train_dev_source_sha256_overlap"])
        fixture = copy.deepcopy(self.train)
        fixture[0]["is_fixture"] = True
        self.assertEqual(self.audit(train=fixture)["errors"], ["train_public_fixture_not_training_data"])

    def test_rejects_repository_aliases_across_splits(self):
        for alias in ("https://github.com:443/example/train-repo-1",
                      "https://www.github.com/example/train-repo-1/",
                      "https://github.com/example/train-repo-1.git"):
            with self.subTest(alias=alias):
                dev = copy.deepcopy(self.dev)
                dev[0]["repository"] = alias
                self.assertEqual(self.audit(dev=dev)["errors"],
                                 ["train_dev_repository_overlap"])
        dev = copy.deepcopy(self.dev)
        dev[0]["repository"] = "https://github.com/example/train-%72epo-1"
        self.assertEqual(self.audit(dev=dev)["errors"], ["dev_repository_url"])

    def test_rejects_text_only_export_and_missing_logprobs(self):
        text_export = {"trainer_ready": False, "groups": [{"samples": [{"text": "x"}]}]}
        self.assertEqual(self.audit(rollouts=text_export)["errors"], ["rollout_export_fields"])
        changed = copy.deepcopy(self.rollouts)
        changed["samples"][0]["behavior_token_logprobs"][-1] = None
        self.assertEqual(self.audit(rollouts=changed)["errors"], ["missing_or_nonfinite_behavior_logprob"])
        changed = copy.deepcopy(self.rollouts)
        changed["samples"][0]["behavior_token_logprobs"][0] = -.1
        self.assertEqual(self.audit(rollouts=changed)["errors"], ["observation_or_context_has_policy_loss"])

    def test_rejects_policy_staleness_misalignment_and_incomplete_groups(self):
        changed = copy.deepcopy(self.plan)
        changed["trainer"]["max_policy_lag"] = 1
        self.assertEqual(self.audit(plan=changed)["errors"], ["stale_policy_requires_off_policy_validation"])
        changed = copy.deepcopy(self.rollouts)
        changed["samples"][0]["teacher_forced_token_logprobs"][-1] = -1.5
        self.assertEqual(self.audit(rollouts=changed)["errors"], ["behavior_teacher_logprob_mismatch"])
        changed = copy.deepcopy(self.rollouts)
        changed["samples"].pop()
        self.assertEqual(self.audit(rollouts=changed)["errors"], ["incomplete_or_mixed_rollout_group"])
        changed = copy.deepcopy(self.rollouts)
        changed["samples"][0]["source_commit"] = "0" * 40
        self.assertEqual(self.audit(rollouts=changed)["errors"], ["rollout_task_source_commit_mismatch"])

    def test_refuses_unhashed_entrypoint_and_symlink_escape(self):
        changed = copy.deepcopy(self.plan)
        changed["trainer"]["entrypoint"] = {"path": "train.py", "sha256": "0" * 64}
        self.assertEqual(self.audit(plan=changed)["errors"], ["trainer_entrypoint:file_hash_mismatch"])
        changed = copy.deepcopy(self.plan)
        changed["trainer"]["entrypoint"]["path"] = "../train.py"
        self.assertEqual(self.audit(plan=changed)["errors"], ["trainer_entrypoint:unsafe_path"])
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside) / "train.py"
            target.write_text("pass\n")
            (self.root / "external.py").symlink_to(target)
            changed = copy.deepcopy(self.plan)
            changed["trainer"]["entrypoint"] = {
                "path": "external.py", "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
            self.assertEqual(self.audit(plan=changed)["errors"],
                             ["trainer_entrypoint:missing_or_external_file"])

    def test_failed_cli_is_machine_readable(self):
        changed = copy.deepcopy(self.plan)
        changed["trainer"]["optimizer"] = "none"
        self.audit(plan=changed)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--plan", str(self.root / "plan.json")]), 2)
        report = json.loads(output.getvalue())
        self.assertFalse(report["input_contract_passed"])
        self.assertFalse(report["trainer_ready"])
        self.assertEqual(report["errors"], ["optimizer_not_explicitly_supported"])

    def test_decoder_recursion_error_is_machine_readable_and_fail_closed(self):
        with patch("future_prediction_bench.gpu_preflight.json.loads",
                   side_effect=RecursionError), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--plan", str(self.root / "plan.json")]), 2)
        report = json.loads(output.getvalue())
        self.assertEqual(report["errors"], ["plan:invalid_json"])
        self.assertFalse(report["input_contract_passed"])
        self.assertFalse(report["trainer_ready"])

    def test_deep_json_and_huge_integer_fail_as_machine_readable_input_errors(self):
        (self.root / "plan.json").write_text("[" * 2000 + "0" + "]" * 2000)
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(["--plan", str(self.root / "plan.json")]), 2)
        report = json.loads(output.getvalue())
        # Decoder recursion limits differ across Python builds; both paths reject input.
        self.assertIn(report["errors"], (["plan:invalid_json"], ["plan_fields"]))
        self.assertFalse(report["input_contract_passed"])

        changed = copy.deepcopy(self.plan)
        changed["trainer"]["learning_rate"] = 10 ** 400
        self.assertEqual(self.audit(plan=changed)["errors"], ["learning_rate"])


if __name__ == "__main__":
    unittest.main()
