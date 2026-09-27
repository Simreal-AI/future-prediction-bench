# KLPO source review

Reviewed on 2026-09-27. The [author report](https://yifanzhang-pro.github.io/KLPO/KLPO.pdf)
is dated September 18 and revised September 20, 2026. The clean official source
checkout is [`yifanzhang-pro/KLPO`](https://github.com/yifanzhang-pro/KLPO/tree/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696),
commit `30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696`, Apache-2.0.

## Actual code executed

`examples/official_klpo/check_loss.py` imports the unmodified author package
normally. It executes token MC-KL, sequence MC-KL, their Full-KL references,
and `klpo.molt.KLPOLoss` with real PyTorch/autograd on CPU. Executing that
adapter class does not execute the separately installed Molt trainer/backend.

The author's `tests/test_mc.py` two-token terminal fixture supplies a fixed
historical sampler and a different current policy. Exact enumeration of the
independent auxiliary draws gives maximum absolute gradient differences of
`1.11e-16` for token MC-KL (`M=1`, 4 banks) and `6.66e-16` for sequence MC-KL
(`M=2`, 16 banks), relative to the corresponding Full-KL gradients. The native
adapter's gradient equals the direct loss gradient for each checked route.
These are numerical identity checks, not a learning or throughput benchmark.

Eight failure cases verify existing upstream input guards. Four optional
project tests additionally check context masking, detached historical records,
and a nonzero singleton-response update. Separately, 133 selected tests in the
clean author checkout passed; one optional native-Molt source-contract test
was skipped because that backend was not supplied. No GPU was used.

The [CPU loss report](measurements/official_klpo_loss_cpu_2026-09-27.json)
records the exact fixture, source hashes, gradients and rejection checks.
The unmodified author `examples/train_toy.py` also completed **four real
CPU AdamW updates** on its four-action toy, with sampler lag 0/1/2/3 and fresh
auxiliary draws. The [toy report](measurements/official_klpo_cpu_toy_2026-09-27.json)
records the command and each update. This toy is separate from our LLM and
sandbox environments and provides no task-success or training-speed result.
Its final realized KL slightly exceeded the supplied predicted-KL budget,
consistent with a prediction rather than a hard guarantee.

## Concrete integration direction

The useful change is a separate complete-trajectory training route which can
consume one resolved response without a same-prompt reward group. The
[actual loss](https://github.com/yifanzhang-pro/KLPO/blob/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696/klpo/loss.py)
uses frozen collection log-probabilities and terminal returns; its default
correction averages independent auxiliary tokens at each visited prefix.
Auxiliary draws are tokens, not additional complete response continuations.

Our existing action/event masks provide a starting point, but current text
training packets explicitly lack token IDs and behavior log-probabilities.
The existing GPU preflight also requires groups of at least two responses.
A KLPO route must therefore be separate from that group contract: retain one
complete resolved episode, exact tokenization, sampler revision and sampling
configuration, generated-token mask, action sampler log-probabilities, and
independent auxiliary token IDs with their original sampler log-probabilities.
Current-model scoring must remain differentiable and use the same stored
histories and token IDs. The adapter can then execute the official loss and
normalize by the complete global response count across microbatches.

This collector/trainer route is not implemented by the CPU example. Neither
mask shape nor finite log-probabilities prove sampling independence, complete
termination, or correct historical sampler identity. These require collector
provenance and audits. Tool observations and prompt/padding tokens must remain
outside the policy loss. Outcome-selected branch data should not be silently
treated as unfiltered sampler trajectories; the report discusses selection
changing the replay distribution.

## Relevant source boundaries

[`klpo/_validation.py`](https://github.com/yifanzhang-pro/KLPO/blob/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696/klpo/_validation.py)
checks tensor shape, active values and mask contracts and detaches historical
records. It does not establish arbitrary logging provenance.
[`klpo/molt.py`](https://github.com/yifanzhang-pro/KLPO/blob/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696/klpo/molt.py)
uses actual rollout scores and raw rewards, and includes global trajectory
normalization. The optional `budget.py` primitives require the complete
optimizer displacement/JVP and predict current-to-proposed policy KL; they do
not guarantee realized KL or provide a historical-sampler admission gate.

The pinned [training guide](https://github.com/yifanzhang-pro/KLPO/blob/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696/docs/training.md)
requires synchronous collection in its current native Molt backend and says
GPU/paper-scale reproduction is unvalidated. MC capture currently materializes
full sampler probabilities before reducing transport records; it is not a
fused GPU sampler. Fixed-record reuse is an empirical surrogate after adaptive
updates; fresh conditional-unbiased estimates need fresh draws from the
historical sampler. Thus this source supports a testable alternative loss and
record contract, without demonstrating asynchronous training speed here.

## Reproduce the source checks

See [`examples/official_klpo/README.md`](../examples/official_klpo/README.md) for
the project command. The selected unmodified author tests were run with:

```sh
python3 -m pytest -q tests/test_loss.py tests/test_mc.py \
  tests/test_population.py tests/test_combinations.py \
  tests/test_budget.py tests/test_molt.py
```

Run this command from the clean author checkout with real PyTorch and pytest.
It checks mathematical/adapter code; it does not execute CUDA/vLLM training.
