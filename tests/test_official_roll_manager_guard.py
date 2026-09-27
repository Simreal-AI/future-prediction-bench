"""Source-level ROLL manager tests; no Ray cluster, model, or optimizer."""

import asyncio
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
import types
import unittest
from pathlib import Path
from threading import Lock
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

try:
    import torch
except ImportError:  # Optional, only the exact-source tensor probe needs it.
    torch = None

from examples.official_roll.gem_bridge import RealWorldGemBridge, RewardNotVerified
from examples.official_roll.manager_guard import (
    guarded_proxy_env_manager_class, require_verified_harbor_result,
)
from examples.official_roll.sample_guard import (
    require_behavior_sample_lineage, require_recorded_behavior_logprobs,
    require_trainable_roll_sample,
)
from future_prediction_bench.realworld import RealWorldEnv


PINNED_SHA = "192b1a01ea61c113b2deb543f7b115783038dff8"


class TensorFixture:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def tolist(self):
        return [row[:] for row in self.rows]


def trainable_sample(reward):
    return types.SimpleNamespace(
        label="optimizer_ready_batch",
        batch={
            "input_ids": TensorFixture([[10, 11, 12, 0]]),
            "attention_mask": TensorFixture([[1, 1, 1, 0]]),
            "prompt_mask": TensorFixture([[1, 0, 0, 0]]),
            "response_mask": TensorFixture([[0, 1, 1, 0]]),
            "scores": TensorFixture([[0.0, 0.0, reward, 0.0]]),
            "infer_logprobs": TensorFixture([[-0.7, -0.2, 0.0]]),
        },
        non_tensor_batch={
            "episode_scores": [reward], "step_scores": [reward],
        },
    )


def bridge_with_status(status, reward, *, actions=1):
    bridge = RealWorldGemBridge(lambda seed: None)
    bridge.env = types.SimpleNamespace(
        status=status, reward=reward, actions_used=actions,
        reason=None, episode_id="source-level-episode", next_verify_at=None,
    )
    return bridge


class ManagerGuardContractTests(unittest.TestCase):
    def test_missing_behavior_logprobs_never_become_masked_or_zero_filled(self):
        recorded = [{"response_ids": [2, 3], "logprobs": [-0.2, -0.3]}]
        self.assertIs(require_recorded_behavior_logprobs(recorded), recorded)
        for invalid in (
            [{"response_ids": [2, 3], "logprobs": None}],
            [{"response_ids": [2, 3], "logprobs": [-0.2]}],
            [{"response_ids": [2, 3], "logprobs": [-0.2, float("nan")]}],
            [{"response_ids": [], "logprobs": []}],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(RewardNotVerified):
                require_recorded_behavior_logprobs(invalid)

    def test_trainable_sample_rejects_placeholder_and_corrupted_fields(self):
        for reward in (0.0, 1.0):
            sample = trainable_sample(reward)
            self.assertIs(require_trainable_roll_sample(
                sample, expected_reward=reward), sample)
        cases = (
            ("no_response", lambda sample: sample.batch["response_mask"].rows[0].__setitem__(slice(None), [0, 0, 0, 0])),
            ("response_on_padding", lambda sample: sample.batch["response_mask"].rows[0].__setitem__(3, 1)),
            ("missing_logprobs", lambda sample: sample.batch.pop("infer_logprobs")),
            ("nan_logprob", lambda sample: sample.batch["infer_logprobs"].rows[0].__setitem__(0, float("nan"))),
            ("bad_token_reward", lambda sample: sample.batch["scores"].rows[0].__setitem__(2, 0.0)),
            ("bad_episode_reward", lambda sample: sample.non_tensor_batch["episode_scores"].__setitem__(0, 0.0)),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                sample = trainable_sample(1.0)
                mutate(sample)
                with self.assertRaises(RewardNotVerified) as caught:
                    require_trainable_roll_sample(sample, expected_reward=1.0)
                self.assertEqual(caught.exception.status, "untrainable_sample")

    def test_dict_gate_requires_exact_verified_reward(self):
        graded_zero = bridge_with_status("graded", 0.0)
        accepted = {"status": "Finished", "score": 0.0, "step_scores": [0.0]}
        self.assertIs(require_verified_harbor_result(accepted, graded_zero), accepted)
        for invalid in (
            {"status": "Finished", "score": 1.0, "step_scores": [0.0]},
            {"status": "Finished", "score": 0.0, "step_scores": []},
            {"status": "Finished", "score": 0.0, "step_scores": [float("nan")]},
            {"status": "Failed", "score": 0.0, "step_scores": [0.0]},
            "not a result mapping",
        ):
            with self.subTest(invalid=invalid), self.assertRaises(RewardNotVerified):
                require_verified_harbor_result(invalid, graded_zero)
        for status in ("active", "pending", "void", "missed"):
            with self.subTest(status=status), self.assertRaises(RewardNotVerified):
                require_verified_harbor_result(accepted, bridge_with_status(status, None))

    def test_sample_lineage_rejects_rewritten_tokens_and_logprobs(self):
        history = [{"response_ids": [11, 12], "logprobs": [-0.7, -0.2]}]
        valid = trainable_sample(1.0)
        self.assertIs(require_behavior_sample_lineage(valid, history), valid)

        sample = trainable_sample(1.0)
        sample.batch["input_ids"].rows[0][1] = 99
        with self.assertRaises(RewardNotVerified) as token_error:
            require_behavior_sample_lineage(sample, history)
        self.assertEqual(token_error.exception.reason, "response_lineage_mismatch")

        sample = trainable_sample(1.0)
        sample.batch["infer_logprobs"].rows[0][0] = -0.6
        with self.assertRaises(RewardNotVerified) as logprob_error:
            require_behavior_sample_lineage(sample, history)
        self.assertEqual(logprob_error.exception.reason, "response_lineage_mismatch")

        with self.assertRaises(RewardNotVerified) as missing_error:
            require_behavior_sample_lineage(trainable_sample(1.0), [])
        self.assertEqual(missing_error.exception.reason,
                         "response_lineage_length_mismatch")

    def test_factory_checks_upstream_types(self):
        with self.assertRaises(TypeError):
            guarded_proxy_env_manager_class(object(), ray_get=lambda x: x)
        with self.assertRaises(TypeError):
            guarded_proxy_env_manager_class(type("Manager", (), {}), ray_get=None)


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as error:
            raise AttributeError(key) from error


class RemoteMethod:
    def __init__(self, func):
        self.func = func

    def remote(self, *args):
        return self.func(*args)


class Queue:
    def __init__(self):
        episodes = iter((3, None))
        self.items = []
        self.get_episode_id = RemoteMethod(lambda group_id, env_id: next(episodes))
        self.put = RemoteMethod(lambda *args: self.items.append(args))


class OfficialSourceManagerTests(unittest.TestCase):
    def _load_official(self):
        checkout_arg = os.environ.get("FPB_UPSTREAM_ROLL")
        if not checkout_arg:
            self.skipTest("Set FPB_UPSTREAM_ROLL to the pinned official ROLL checkout")
        checkout = Path(checkout_arg).resolve()
        sha = subprocess.check_output(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        self.assertEqual(sha, PINNED_SHA)
        paths = (
            "roll/pipeline/agentic/agent_runner/base.py",
            "roll/pipeline/agentic/agent_runner/gem_runner.py",
            "roll/pipeline/agentic/env_manager/proxy_env_manager.py",
            "roll/utils/import_utils.py",
        )
        dirty = subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain", "--", *paths],
            text=True,
        )
        self.assertEqual(dirty, "", "Audited upstream manager/runner files must be clean")

        def module(name, **attributes):
            item = types.ModuleType(name)
            item.__dict__.update(attributes)
            if name in ("roll", "roll.pipeline", "roll.pipeline.agentic",
                        "roll.pipeline.agentic.env", "roll.pipeline.agentic.agent_runner",
                        "roll.pipeline.agentic.env_manager", "roll.distributed",
                        "roll.distributed.scheduler", "roll.utils", "sglang",
                        "sglang.srt", "sglang.srt.function_call",
                        "sglang.srt.entrypoints", "sglang.srt.entrypoints.openai",
                        "fastapi"):
                item.__path__ = []
            return item

        packages = (
            "roll", "roll.pipeline", "roll.pipeline.agentic",
            "roll.pipeline.agentic.env", "roll.pipeline.agentic.agent_runner",
            "roll.pipeline.agentic.env_manager", "roll.distributed",
            "roll.distributed.scheduler", "roll.utils", "sglang",
            "sglang.srt", "sglang.srt.function_call",
            "sglang.srt.entrypoints", "sglang.srt.entrypoints.openai",
        )
        modules = {name: module(name) for name in packages}
        modules["omegaconf"] = module("omegaconf", DictConfig=AttrDict)
        modules["ray"] = module("ray", get=lambda value: value)
        modules["tensordict"] = module("tensordict", List=list, TensorDict=dict)
        modules["transformers"] = module("transformers", PreTrainedTokenizer=object)
        modules["fastapi"] = module("fastapi", Request=object)
        modules["fastapi.responses"] = module("fastapi.responses", JSONResponse=object)
        modules["sglang.srt.function_call.function_call_parser"] = module(
            "sglang.srt.function_call.function_call_parser", FunctionCallParser=object)
        modules["sglang.srt.entrypoints.openai.protocol"] = module(
            "sglang.srt.entrypoints.openai.protocol", Tool=object)
        modules["roll.pipeline.agentic.env"].gem = module("gem")
        modules["roll.pipeline.agentic.env"].gem.make = lambda **kwargs: None
        modules["roll.pipeline.agentic.llm_proxy"] = module(
            "roll.pipeline.agentic.llm_proxy", create_llm_proxy=lambda **kwargs: None,
            BaseLLMProxy=object)
        modules["roll.pipeline.agentic.env_manager.base_env_manager"] = module(
            "roll.pipeline.agentic.env_manager.base_env_manager",
            BaseEnvManager=type("BaseEnvManager", (), {
                "__init__": lambda self: setattr(self, "current_step", 0),
            }))
        modules["roll.pipeline.agentic.env_manager.message_tracker"] = module(
            "roll.pipeline.agentic.env_manager.message_tracker",
            MessageTracker=type("MessageTracker", (), {
                "__init__": lambda self, *args, **kwargs: None,
            }))
        modules["roll.distributed.scheduler.protocol"] = module(
            "roll.distributed.scheduler.protocol", DataProto=object)
        modules["roll.distributed.scheduler.rollout_scheduler"] = module(
            "roll.distributed.scheduler.rollout_scheduler", GroupQueueManager=object)
        modules["roll.distributed.scheduler.router"] = module(
            "roll.distributed.scheduler.router", RouterManager=object)
        modules["roll.pipeline.agentic.agentic_config"] = module(
            "roll.pipeline.agentic.agentic_config", EnvManagerConfig=object,
            AgenticConfig=object)
        modules["roll.utils.functionals"] = module(
            "roll.utils.functionals", pad_to_length=lambda value, **kwargs: value)
        modules["roll.utils.logging"] = module(
            "roll.utils.logging", get_logger=lambda: logging.getLogger("roll-source-test"))
        modules["roll.utils.str_utils"] = module(
            "roll.utils.str_utils", contains_renderable_field=lambda *args: False)
        modules["roll.utils.import_utils"] = module(
            "roll.utils.import_utils",
            safe_import_class=lambda path: getattr(
                importlib.import_module(path.rsplit(".", 1)[0]), path.rsplit(".", 1)[1]))

        def load(name, path):
            spec = importlib.util.spec_from_file_location(name, checkout / path)
            self.assertIsNotNone(spec)
            item = importlib.util.module_from_spec(spec)
            sys.modules[name] = item
            spec.loader.exec_module(item)
            return item

        return modules, load

    def test_pinned_upstream_tensor_formulation_and_placeholder_rejection(self):
        """Run the unmodified ROLL formatter on real CPU torch tensors."""
        if torch is None:
            self.skipTest("PyTorch is required for the pinned tensor probe")

        modules, load = self._load_official()

        class TensorDictFixture(dict):
            def __init__(self, value, *, batch_size):
                super().__init__(value)
                self.batch_size = (batch_size,)

        class DataProtoFixture:
            def __init__(self, *, batch=None, non_tensor_batch=None, meta_info=None):
                self.batch = batch
                self.non_tensor_batch = non_tensor_batch
                self.meta_info = meta_info or {}

            @classmethod
            def concat(cls, samples):
                if len(samples) != 1:
                    raise AssertionError("This exact fixture is one trajectory")
                return samples[0]

        modules["tensordict"].TensorDict = TensorDictFixture
        modules["roll.distributed.scheduler.protocol"].DataProto = DataProtoFixture
        with patch.dict(sys.modules, modules):
            load("roll.pipeline.agentic.agent_runner.base",
                 "roll/pipeline/agentic/agent_runner/base.py")
            load("roll.pipeline.agentic.agent_runner.gem_runner",
                 "roll/pipeline/agentic/agent_runner/gem_runner.py")
            proxy = load("roll.pipeline.agentic.env_manager.proxy_env_manager",
                         "roll/pipeline/agentic/env_manager/proxy_env_manager.py")
            guarded = guarded_proxy_env_manager_class(proxy.ProxyEnvManager,
                                                      ray_get=lambda value: value)

            for grade in (0.0, 1.0):
                with self.subTest(verified_grade=grade):
                    manager = object.__new__(guarded)
                    manager.agent_runner = types.SimpleNamespace(
                        env=bridge_with_status("graded", grade))
                    manager.env_config = AttrDict(tag="fixture", group_id=1, env_id=2)
                    manager.pipeline_config = types.SimpleNamespace(
                        sequence_length=4, exp_name="source-level")
                    manager.episode_id = 3
                    manager.group_seed = 4
                    manager.reward_granularity = "binary"
                    manager.trajectory_mode = "traj"
                    manager.tokenizer = types.SimpleNamespace(pad_token_id=0)
                    manager.message_tracker = types.SimpleNamespace(
                        get_trajectory_data=lambda mode: [{
                            "token_ids": [101, 102, 103, 104],
                            "response_masks": [0, 0, 1, 1],
                            "logprobs": [0.0, 0.0, -0.5, -0.7],
                            "messages": [],
                        }])
                    manager.history = [{
                        "response_ids": [103, 104], "logprobs": [-0.5, -0.7],
                    }]
                    manager.tools = None
                    manager.traj_start_time = time.time()
                    manager.log_stats = {key: [] for key in (
                        "step_rt", "pure_infer_time", "env_exec_time",
                        "proxy_overhead")}
                    manager.logger = logging.getLogger("roll-tensor-source-test")
                    sample = manager.formulate_rollouts({
                        "status": "Finished", "score": grade,
                        "step_scores": [grade],
                    })
                    self.assertEqual(sample.batch["response_mask"].tolist(),
                                     [[False, False, True, True]])
                    self.assertEqual(sample.batch["prompt_mask"].tolist(),
                                     [[True, True, False, False]])
                    self.assertEqual(sample.batch["infer_logprobs"].shape,
                                     (1, 3))
                    for observed, expected in zip(
                        sample.batch["infer_logprobs"].tolist()[0],
                        (0.0, -0.5, -0.7),
                    ):
                        self.assertAlmostEqual(observed, expected, places=6)
                    self.assertTrue(torch.isclose(
                        sample.batch["scores"].sum(), torch.tensor(grade)))
                    self.assertEqual(sample.non_tensor_batch["episode_scores"][0],
                                     grade)

                    # Upstream creates a no-policy-token zero placeholder
                    # when MessageTracker has no trajectory, even if the
                    # environment's true grade is one. Our guard blocks it.
                    manager.message_tracker = types.SimpleNamespace(
                        get_trajectory_data=lambda mode: [])
                    with self.assertRaises(RewardNotVerified) as caught:
                        manager.formulate_rollouts({
                            "status": "Finished", "score": grade,
                            "step_scores": [grade],
                        })
                    self.assertEqual(caught.exception.status,
                                     "untrainable_sample")

    def test_official_proxy_loop_and_config_factory_gate(self):
        modules, load = self._load_official()
        checkout = Path(os.environ["FPB_UPSTREAM_ROLL"]).resolve()
        with patch.dict(sys.modules, modules):
            base = load("roll.pipeline.agentic.agent_runner.base",
                        "roll/pipeline/agentic/agent_runner/base.py")
            gem = load("roll.pipeline.agentic.agent_runner.gem_runner",
                       "roll/pipeline/agentic/agent_runner/gem_runner.py")
            load("roll.utils.import_utils", "roll/utils/import_utils.py")
            proxy = load("roll.pipeline.agentic.env_manager.proxy_env_manager",
                         "roll/pipeline/agentic/env_manager/proxy_env_manager.py")
            self.assertEqual(Path(proxy.__file__).resolve(), checkout /
                             "roll/pipeline/agentic/env_manager/proxy_env_manager.py")

            factory_module = types.ModuleType("roll_fixture_factory")
            factory_module.factory = lambda seed: None
            with patch.dict(sys.modules, {"roll_fixture_factory": factory_module}):
                configured = importlib.import_module(
                    "examples.official_roll.configured_runner")
                manager_module = importlib.import_module(
                    "examples.official_roll.verified_proxy_env_manager")
                self.assertTrue(issubclass(configured.VerifiedRealWorldGEMRunner,
                                           gem.GEMRunner))
                self.assertTrue(issubclass(manager_module.VerifiedProxyEnvManager,
                                           proxy.ProxyEnvManager))

                def build_manager(manager_cls=None):
                    queue = Queue()
                    env_config = AttrDict({
                        "group_id": 2, "env_id": 7, "group_seed": 10,
                        "proxy_port": 8000,
                        "agent_runner_cls":
                            "examples.official_roll.configured_runner.VerifiedRealWorldGEMRunner",
                        "config": AttrDict({"fpb_env_factory":
                                             "roll_fixture_factory:factory"}),
                        "max_steps": 1, "agent_system_template": "Act",
                        "agent_template": "{observation}",
                    })
                    manager = (manager_cls or manager_module.VerifiedProxyEnvManager)(
                        worker_config=types.SimpleNamespace(llm_proxy=None),
                        pipeline_config=types.SimpleNamespace(),
                        env_config=env_config, tokenizer=object(),
                        generate_scheduler=None, output_queue=queue,
                        thread_lock=Lock(),
                    )
                    self.assertIsInstance(manager.agent_runner.env,
                                          RealWorldGemBridge)
                    self.assertIs(manager.agent_runner.env.env_factory,
                                  factory_module.factory)
                    return manager, queue

                with self.assertRaises(ValueError):
                    configured.resolve_env_factory("missing_colon")
                with self.assertRaises(ValueError):
                    configured.resolve_env_factory("os:invalid.attr")

                score_cases = (
                    ("graded", 1.0, 1.0, True),
                    ("graded", 0.0, 0.0, True),
                    ("active", None, 0.0, False),
                    ("pending", None, 0.0, False),
                    ("void", None, 0.0, False),
                    ("missed", None, 0.0, False),
                    ("graded", 1.0, 0.0, False),
                )
                for status, reward, result_score, accepted in score_cases:
                    with self.subTest(status=status, reward=reward,
                                      result_score=result_score):
                        manager, queue = build_manager()
                        manager.agent_runner.env.env = bridge_with_status(
                            status, reward).env
                        def fixture_job(seed, score=result_score):
                            manager.history.append({
                                "response_ids": [11, 12],
                                "logprobs": [-0.7, -0.2],
                            })
                            return base.EpisodeResult("Finished", score, [score])
                        manager.agent_runner.run_job = fixture_job
                        manager.formulate_calls = []
                        def make_sample(self, result):
                            self.formulate_calls.append(result)
                            return trainable_sample(result["score"])
                        # Only sample tensor construction is stubbed; the
                        # official manager constructor and run loop execute.
                        with patch.object(proxy.ProxyEnvManager,
                                          "formulate_rollouts", make_sample):
                            if accepted:
                                manager.run_rollout_loop(types.SimpleNamespace(
                                    meta_info={"seed": 100}))
                                self.assertEqual(len(manager.formulate_calls), 1)
                                self.assertEqual(
                                    [getattr(item[3], "label", None) for item in queue.items],
                                    ["optimizer_ready_batch", None])
                            else:
                                with self.assertRaises(RewardNotVerified):
                                    manager.run_rollout_loop(types.SimpleNamespace(
                                        meta_info={"seed": 100}))
                                self.assertEqual(manager.formulate_calls, [])
                                self.assertEqual([item[3] for item in queue.items],
                                                 [None])
                                self.assertEqual(queue.items[0][1], 3)

                # The unmodified manager accepts a runner's false zero and
                # hands it to sample construction. This is the negative
                # control for our pre-formulation gate.
                stock, stock_queue = build_manager(proxy.ProxyEnvManager)
                stock.agent_runner.env.env = bridge_with_status("pending", None).env
                stock.agent_runner.run_job = lambda seed: base.EpisodeResult(
                    "Finished", 0.0, [0.0])
                with patch.object(proxy.ProxyEnvManager, "formulate_rollouts",
                                  lambda self, result: "unsafe_zero_batch"):
                    stock.run_rollout_loop(types.SimpleNamespace(meta_info={"seed": 100}))
                self.assertEqual([item[3] for item in stock_queue.items],
                                 ["unsafe_zero_batch", None])

                # Actual pending exceptions from the guarded runner happen
                # before formulate_rollouts; the manager must still signal
                # queue completion for its claimed episode and re-raise.
                pending, pending_queue = build_manager()
                def pending_runner(seed):
                    raise RewardNotVerified(status="pending", reason="not_due",
                                            episode_id="source-level-episode")
                pending.agent_runner.run_job = pending_runner
                with patch.object(proxy.ProxyEnvManager, "formulate_rollouts",
                                  side_effect=AssertionError("must not formulate")):
                    with self.assertRaises(RewardNotVerified) as caught:
                        pending.run_rollout_loop(types.SimpleNamespace(
                            meta_info={"seed": 100}))
                self.assertEqual(caught.exception.status, "pending")
                self.assertEqual(pending.last_unverified_status, "pending")
                self.assertEqual([item[3] for item in pending_queue.items], [None])

    def test_official_request_tracker_runner_and_formatter_end_to_end(self):
        """Execute the pinned ROLL path with deterministic CPU model responses.

        The model, HTTP transport, and unavailable ROLL infrastructure are
        doubles. Request processing, MessageTracker, GEMRunner, manager loop,
        and tensor formulation run from the clean pinned upstream checkout.
        """
        if torch is None:
            self.skipTest("PyTorch is required for the pinned CPU rollout probe")
        modules, load = self._load_official()
        checkout = Path(os.environ["FPB_UPSTREAM_ROLL"]).resolve()
        extra_paths = (
            "roll/pipeline/agentic/env_manager/message_tracker.py",
            "roll/pipeline/agentic/env_manager/token_mask_utils.py",
        )
        self.assertEqual(subprocess.check_output(
            ["git", "-C", str(checkout), "status", "--porcelain", "--", *extra_paths],
            text=True,
        ), "", "Audited upstream tracker/tokenizer utility files must be clean")

        class TensorDictFixture(dict):
            def __init__(self, values, *, batch_size):
                super().__init__(values)
                self.batch_size = (batch_size,)

        class DataProtoFixture:
            def __init__(self, *, batch=None, non_tensor_batch=None, meta_info=None):
                self.batch = batch
                self.non_tensor_batch = non_tensor_batch
                self.meta_info = meta_info or {}

            @classmethod
            def concat(cls, samples):
                if len(samples) != 1:
                    raise AssertionError("Expected exactly one non-fork trajectory")
                return samples[0]

        class TinyTokenizer:
            pad_token_id = 0

            def encode(self, text):
                return [10] if text == "\n" else [31 + len(text) % 31]

            def apply_chat_template(self, messages, *, add_generation_prompt=False,
                                    **kwargs):
                ids = []
                for message in messages:
                    content = message.get("content", "")
                    ids.append(40 + len(str(content)) % 59)
                if add_generation_prompt:
                    ids.append(300)
                return ids

            def decode(self, ids, **kwargs):
                return {
                    (201, 202): '{"action":"read_file","path":"main.py"}',
                    (203, 204): '{"action":"submit"}',
                }[tuple(ids)]

        class Adapter:
            def __init__(self, grade):
                self.grade = grade
                self.actions = []
                self.verify_calls = 0

            def reset(self, specification, *, now):
                return {"workspace": "visible fixture"}

            def step(self, action, *, now):
                self.actions.append(action)
                if action["action"] == "submit":
                    return {"observation": {"status": "submitted"}, "terminated": True}
                return {"observation": {"status": "read"}, "terminated": False}

            def verify(self, *, now):
                self.verify_calls += 1
                return {"status": "resolved", "reward": self.grade,
                        "evidence": {"secret": "verifier-only"},
                        "available_at": now.isoformat()}

            def get_state(self):
                return {"actions": len(self.actions)}

            def close(self):
                pass

        class ChatMessageAdapter:
            def __init__(self, kind):
                pass

            def validate_python(self, message):
                data = dict(message)
                return types.SimpleNamespace(model_dump=lambda: data)

        def pad_to_length(value, *, length, pad_value):
            if value.shape[-1] > length:
                raise AssertionError("Fixture exceeded configured sequence length")
            return torch.nn.functional.pad(
                value, (0, length - value.shape[-1]), value=pad_value)

        modules["tensordict"].TensorDict = TensorDictFixture
        modules["roll.distributed.scheduler.protocol"].DataProto = DataProtoFixture
        modules["roll.utils.functionals"].pad_to_length = pad_to_length
        modules["pydantic"] = types.ModuleType("pydantic")
        modules["pydantic"].TypeAdapter = ChatMessageAdapter
        modules["sglang.srt.entrypoints.openai.protocol"].ChatCompletionMessageParam = dict
        modules["roll.datasets"] = types.ModuleType("roll.datasets")
        modules["roll.datasets"].__path__ = []
        modules["roll.datasets.collator"] = types.ModuleType("roll.datasets.collator")
        modules["roll.datasets.collator"].DataCollatorWithPaddingForMM = object

        started = datetime(2030, 1, 1, 10, tzinfo=timezone.utc)
        for grade, delayed, missing_logprobs in (
            (1.0, False, False),
            (0.0, False, False),
            (1.0, True, False),
            (1.0, False, True),
        ):
            with self.subTest(grade=grade, delayed=delayed,
                              missing_logprobs=missing_logprobs):
                adapters = []

                def make_env(seed):
                    adapter = Adapter(grade)
                    adapters.append(adapter)
                    specification = {
                        "schema_version": "realworld-0.1",
                        "task_id": f"roll-source-e2e-{seed}",
                        "event_id": f"roll-source-event-{seed}",
                        "cluster_id": f"roll-source-cluster-{seed}",
                        "split": "train", "prompt": "Read a file and submit.",
                        "issued_at": started.isoformat(),
                        "action_deadline": (started + timedelta(hours=2)).isoformat(),
                        "outcome_not_before": started.isoformat(),
                        "verify_after": (started + timedelta(seconds=60 if delayed else 0)).isoformat(),
                        "tool_manifest": [
                            {"name": "read_file", "description": "Read a file."},
                            {"name": "submit", "description": "Submit the candidate."},
                        ],
                        "reward_contract": {"id": "source-e2e", "description": "Private grade.",
                                            "min_reward": 0.0, "max_reward": 1.0},
                        "budgets": {"max_actions": 3, "max_wall_seconds": 7200,
                                    "verification_cooldown_seconds": 10,
                                    "max_verifications": 3},
                        "is_fixture": True,
                        "metadata": {"private_fixture_key": "never-show-me"},
                    }
                    return RealWorldEnv(specification, adapter, clock=lambda: started,
                                        monotonic_clock=lambda: 0.0)

                class Proxy:
                    def __init__(self):
                        self.calls = []

                    def generate(self, *, messages, lm_input, generation_config):
                        step = len(self.calls)
                        self.calls.append((messages, lm_input, generation_config))
                        response = [201, 202] if step == 0 else [203, 204]
                        batch = {"responses": torch.tensor([response], dtype=torch.long)}
                        if not missing_logprobs or step == 0:
                            values = [-0.2, -0.3] if step == 0 else [-0.4, -0.5]
                            batch["infer_logprobs"] = torch.tensor([values])
                        return DataProtoFixture(batch=TensorDictFixture(batch, batch_size=1))

                model = Proxy()
                modules["roll.pipeline.agentic.llm_proxy"].create_llm_proxy = (
                    lambda **kwargs: model)
                with patch.dict(sys.modules, modules):
                    base = load("roll.pipeline.agentic.agent_runner.base",
                                "roll/pipeline/agentic/agent_runner/base.py")
                    gem = load("roll.pipeline.agentic.agent_runner.gem_runner",
                               "roll/pipeline/agentic/agent_runner/gem_runner.py")
                    load("roll.pipeline.agentic.env_manager.token_mask_utils",
                         "roll/pipeline/agentic/env_manager/token_mask_utils.py")
                    tracker = load("roll.pipeline.agentic.env_manager.message_tracker",
                                   "roll/pipeline/agentic/env_manager/message_tracker.py")
                    proxy = load("roll.pipeline.agentic.env_manager.proxy_env_manager",
                                 "roll/pipeline/agentic/env_manager/proxy_env_manager.py")
                    self.assertEqual(Path(tracker.__file__).resolve(), checkout /
                                     "roll/pipeline/agentic/env_manager/message_tracker.py")
                    self.assertEqual(Path(proxy.__file__).resolve(), checkout /
                                     "roll/pipeline/agentic/env_manager/proxy_env_manager.py")

                    from examples.official_roll.guarded_runner import guarded_gem_runner_class

                    GuardedRunner = guarded_gem_runner_class(gem.GEMRunner)

                    class Runner(GuardedRunner):
                        def __init__(self, *args, **kwargs):
                            super().__init__(*args, env_factory=make_env, **kwargs)

                    runner_module = types.ModuleType("roll_source_e2e_runner")
                    runner_module.Runner = Runner
                    sys.modules["roll_source_e2e_runner"] = runner_module
                    manager_class = guarded_proxy_env_manager_class(
                        proxy.ProxyEnvManager, ray_get=lambda value: value)
                    env_config = AttrDict({
                        "group_id": 2, "env_id": 7, "group_seed": 10,
                        "proxy_port": 8000,
                        "agent_runner_cls": "roll_source_e2e_runner.Runner",
                        "config": AttrDict({"reward_granularity": "binary",
                                            "trajectory_mode": "traj"}),
                        "max_steps": 3, "max_tokens_per_step": 8,
                        "agent_system_template": "Act", "agent_template": "{observation}",
                    })
                    generating_args = types.SimpleNamespace(
                        max_new_tokens=8, to_dict=lambda: {"max_new_tokens": 8})
                    queue = Queue()
                    manager = manager_class(
                        worker_config=types.SimpleNamespace(
                            llm_proxy=None, generating_args=generating_args),
                        pipeline_config=types.SimpleNamespace(
                            sequence_length=128, exp_name="source-e2e"),
                        env_config=env_config, tokenizer=TinyTokenizer(),
                        generate_scheduler=None, output_queue=queue,
                        thread_lock=Lock(),
                    )

                    class Response:
                        def __init__(self, body):
                            self.body = body

                        def raise_for_status(self):
                            pass

                        def json(self):
                            return self.body

                    class Client:
                        def __init__(self, *, timeout):
                            self.timeout = timeout

                        def __enter__(self):
                            return self

                        def __exit__(self, *args):
                            pass

                        def post(self, url, *, json, headers):
                            self_test.assertTrue(url.endswith("/v1/chat/completions"))
                            self_test.assertEqual(headers["Authorization"], "Bearer 7")
                            return Response(asyncio.run(manager._process_request_dict(json)))

                    self_test = self
                    with patch.object(base.httpx, "Client", Client):
                        if delayed or missing_logprobs:
                            with self.assertRaises(RewardNotVerified) as caught:
                                manager.run_rollout_loop(DataProtoFixture(
                                    meta_info={"seed": 100}))
                            self.assertEqual(caught.exception.status,
                                             "pending" if delayed else "untrainable_sample")
                            self.assertEqual([item[3] for item in queue.items], [None])
                        else:
                            manager.run_rollout_loop(DataProtoFixture(
                                meta_info={"seed": 100}))
                            sample = queue.items[0][3]
                            self.assertIsInstance(sample, DataProtoFixture)
                            self.assertIsNone(queue.items[1][3])
                            mask = sample.batch["response_mask"][0]
                            positions = mask.nonzero().flatten().tolist()
                            self.assertEqual(sample.batch["input_ids"][0, mask].tolist(),
                                             [201, 202, 203, 204])
                            actual_logprobs = [
                                sample.batch["infer_logprobs"][0, index - 1].item()
                                for index in positions
                            ]
                            for actual, expected in zip(
                                actual_logprobs, [-0.2, -0.3, -0.4, -0.5]):
                                self.assertAlmostEqual(actual, expected, places=6)
                            self.assertAlmostEqual(sample.batch["scores"].sum().item(),
                                                   grade)
                            self.assertEqual(sample.non_tensor_batch["episode_scores"][0],
                                             grade)

                    self.assertEqual(len(model.calls), 2)
                    self.assertTrue(all(call[1].batch["input_ids"].dtype == torch.long
                                        for call in model.calls))
                    self.assertEqual(len(manager.history), 2)
                    self.assertEqual(len(manager.message_tracker.step_records), 2)
                    self.assertEqual([action["action"] for action in adapters[0].actions],
                                     ["read_file", "submit"])
                    self.assertEqual(adapters[0].verify_calls, 0 if delayed else 1)
                    self.assertNotIn("verifier-only", json.dumps(manager.history))


if __name__ == "__main__":
    unittest.main()
