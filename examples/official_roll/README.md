# ROLL environment protocol bridge (experimental)

This example borrows the **actual environment call shape** from Alibaba ROLL at
commit [`192b1a01ea61c113b2deb543f7b115783038dff8`](https://github.com/alibaba/ROLL/tree/192b1a01ea61c113b2deb543f7b115783038dff8). Its
[`GEMRunner`](https://github.com/alibaba/ROLL/blob/192b1a01ea61c113b2deb543f7b115783038dff8/roll/pipeline/agentic/agent_runner/gem_runner.py)
calls `env.reset(seed=seed)` for `(observation_text, info)`, then calls
`env.step(action_text)` for `(observation_text, reward, terminated, truncated,
info)` and sums numeric step rewards. The
[`AgentRunner` result contract](https://github.com/alibaba/ROLL/blob/192b1a01ea61c113b2deb543f7b115783038dff8/roll/pipeline/agentic/agent_runner/base.py)
contains a scalar episode `score`.

[`RealWorldGemBridge`](gem_bridge.py) maps a fresh `RealWorldEnv` per integer
seed to that text-mode protocol. Its caller supplies a deterministic
`env_factory(seed)` with the frozen task and trusted adapter. The policy sends
one JSON action object as text. An active, nonterminal action returns an
intermediate `0.0`; a submitted episode returns a terminal scalar only when
trusted verification reports `graded`. A verified failure may legitimately
return `0.0`. A `pending`, `not_due`, `void`, `missed`, or interrupted episode
instead raises `RewardNotVerified`, so those states never masquerade as a
verified zero. The trusted host may later call `collect_verified_reward()`
without submitting again; stock GEMRunner does not use that method.

[`guarded_runner.py`](guarded_runner.py) adds the next safety boundary.
`guarded_gem_runner_class(official.GEMRunner)` builds a local subclass whose
`run_job()` calls the **unmodified** official method, then checks that the
bridged `RealWorldEnv` is graded and its reward, action count, and numeric step
scores agree with the returned `EpisodeResult`. A max-step or inference-error
exit without submission raises `RewardNotVerified` before the result can be
passed to a manager. A legitimately verified `0.0` still passes. The subclass
needs an operator-supplied `env_factory(seed)` constructor argument; ROLL's
stock YAML class loader does not supply that argument.

[`configured_runner.py`](configured_runner.py) supplies that constructor
argument through ROLL's `agent_runner_cls` setting. The operator configures
`config.fpb_env_factory` as an importable `module:function` that accepts an
integer seed and returns a fresh, frozen `RealWorldEnv`. The trusted factory
and this source package must be importable in every ROLL worker process.
[`verified_proxy_env_manager.py`](verified_proxy_env_manager.py) supplies an
`env_manager_cls` setting. It subclasses the actual ROLL `ProxyEnvManager` and
checks the runner's result dict against the trusted grade **before** calling
ROLL's `formulate_rollouts`; an unverified status or mismatched zero cannot
reach its zero-score fallback. After the upstream formatter returns,
[`sample_guard.py`](sample_guard.py) checks the actual token, attention,
response, prompt, policy-logprob, and reward tensors before queue insertion.
It rejects ROLL's no-response placeholder, malformed masks, nonfinite sampled
logprobs, and reward disagreement, including when the verified grade is zero.
It also rejects a recorded inference response whose behavior logprobs are
missing or misaligned: ROLL's message tracker otherwise zero-fills or masks
that response. This adapter therefore requires a generation backend that
provides token logprobs. The gate also matches every queued generated token
and shifted behavior logprob to the response recorded by the inference
handler. Rewritten or duplicated responses fail closed. This ordered check
supports single-branch trajectory and step samples; forked branches need
explicit lineage before they can be accepted.
This is a data-integrity gate, not proof that the sample is optimizer-ready.
The class path
contains `proxy_env_manager`, which is also how the pinned ROLL
`EnvironmentWorker` decides to start its proxy server.

For a ROLL installation with this source checkout on `PYTHONPATH`, the
relevant environment template fields are:

```yaml
env_manager_cls: examples.official_roll.verified_proxy_env_manager.VerifiedProxyEnvManager
agent_runner_cls: examples.official_roll.configured_runner.VerifiedRealWorldGEMRunner
config:
  fpb_env_factory: your_trusted_module:make_frozen_realworld_env
```

These are the class-loader fields, not a complete runnable ROLL training
configuration. The factory is operator-owned because it must supply the
task/adapter registry and trusted verification roots. The example package is
included in the source materials but not in this project's Python wheel, so
workers need the source checkout or an operator-built package containing it.

The pinned upstream manager's `run_rollout_loop` has no recovery path around
`run_job()` or `formulate_rollouts()`. Our subclass catches
`RewardNotVerified`, sends `None` for that claimed episode to ROLL's output
queue, then re-raises so the failure is visible. In the pinned
`GroupQueue.put`, `None` occupies one member of a group; `get_batch` removes
such members and marks a group complete only when all expected members have
arrived. This prevents an unverified episode from becoming a training sample,
but it is **not** a complete delayed-reward scheduler: with too few verified
episodes, ROLL may return a partial/empty batch or fail rather than refill the
requested batch. A durable handoff and later trajectory/logprob reassembly
are still required for real asynchronous training.

Run the offline contract tests from the repository root:

```bash
python3 -m pytest -q tests/test_official_roll_bridge.py tests/test_official_roll_manager_guard.py
```

If you have cloned the audited official commit, run the same tests against its
real runner and manager source files:

```bash
FPB_UPSTREAM_ROLL=/path/to/ROLL-at-192b1a01 \
  python3 -m pytest -q tests/test_official_roll_bridge.py tests/test_official_roll_manager_guard.py
```

The optional test checks the checkout's exact Git SHA, loads the official
runner source **without changing it**, substitutes only unavailable imports
(`gem`, `omegaconf`, and file-writing ROLL logger) and fake LLM responses,
loads ROLL's actual string utility module, then calls
official `GEMRunner.run_job()`. On the pinned checkout, stock runner behavior
was: verified submission returned `Finished` with `score=1.0` and step scores
`[0.0, 1.0]`; a not-yet-due submission raised `RewardNotVerified` with no
score; a one-step run that never submitted returned `Finished` with
`score=0.0`. The guarded subclass blocked the unsubmitted zero and an
inference-error zero, while accepting a genuinely verified zero. No model
API, GPU, Ray worker, or optimizer ran.

The source-level manager test also checks the exact upstream commit and clean
manager/runner files, executes the unmodified `ProxyEnvManager.__init__` and
`run_rollout_loop` methods with only missing infrastructure dependencies and
sample tensor construction stubbed, and selects both local classes through
ROLL's actual import/config constructor shape. It observed that the stock
manager handed a pending episode's false zero to sample construction; the
verified manager rejected active, pending, void, missed, and mismatched
results before sample construction, preserved genuine verified zero, posted
only a `None` queue marker for rejection, and surfaced the exception. A
separate source-level probe executes the **unmodified** upstream
`ProxyEnvManager.formulate_rollouts` on real CPU PyTorch tensors (with small
`TensorDict`/`DataProto` infrastructure doubles). It checks the resulting
response/prompt masks, shifted behavior logprobs, and grade for both verified
zero and one; the local gate rejects upstream's no-token placeholder for both.

Another source-level test executes a complete two-action rollout through the
unmodified upstream `GEMRunner.run_job`, `ProxyEnvManager.run_rollout_loop`,
`_process_request_dict`, `MessageTracker`, token-content utility, and
`formulate_rollouts`. A deterministic CPU model double returns two response
token pairs with known behavior logprobs; an HTTP transport double routes the
runner's request into the real async request handler. The test checks that
those tokens and shifted logprobs reach the queued real PyTorch sample
unchanged for verified rewards `1.0` and `0.0`. Delayed verification and
missing second-step logprobs emit only a `None` completion marker. Unavailable
Ray, model, tokenizer, and `TensorDict`/`DataProto` infrastructure are
doubled. No live model, socket, optimizer, or GPU runs in this test.

**Integration boundary:** This is a local protocol bridge and fail-closed
ROLL runner/manager customization, not a trained ROLL model, registered
`gem.make` environment, or a live model-generated log-probability pipeline.
The full ROLL package was not installed or run; the optional smoke executes
its unmodified runner, request handler, tracker, and manager methods under
minimal dependency stubs.
The pinned stock GEMRunner returns `Finished` with a sum of rewards when its
maximum step count is reached or an inference response carries an error, even
if no submission occurred. Its
[`ProxyEnvManager`](https://github.com/alibaba/ROLL/blob/192b1a01ea61c113b2deb543f7b115783038dff8/roll/pipeline/agentic/env_manager/proxy_env_manager.py)
does not supply a delayed-reward queue and has zero-score fallback paths in
sample construction. A real training integration still needs a durable
pending-reward path and batch refill/trajectory reassembly after delayed
verification. It
must then be tested with the installed ROLL stack. No speed or training-quality
gain is claimed here.

The example contains no copied upstream code. See the linked official files
for ROLL's implementation and licensing terms.
