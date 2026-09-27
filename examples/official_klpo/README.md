# Official KLPO loss checks

This example imports the author's unmodified `klpo` package and executes its
token/sequence MC-KL losses, exact Full-KL losses, and `klpo.molt.KLPOLoss`
adapter with real CPU PyTorch/autograd. It does not import or launch the Molt
trainer, replace dependencies, or collect model rollouts.

The reviewed source is
[`yifanzhang-pro/KLPO`](https://github.com/yifanzhang-pro/KLPO/tree/30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696)
at commit `30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696`. The source is Apache-2.0
and is not vendored in this package. Supply a clean checkout and PyTorch 2.2+:

```sh
git clone https://github.com/yifanzhang-pro/KLPO.git /path/to/KLPO
git -C /path/to/KLPO checkout --detach 30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696
python3 -m examples.official_klpo.check_loss \
  --checkout /path/to/KLPO --output /new/path/klpo-check.json
FPB_KLPO_CHECKOUT=/path/to/KLPO python3 -m unittest discover \
  -s tests -p test_official_klpo.py -v
```

The output records source hashes and exact import provenance. The author's
two-token terminal fixture is scored against a fixed historical sampler.
Enumerating all 4 auxiliary banks for token MC-KL (`M=1`) and all 16 for
sequence MC-KL (`M=2`) checks that each expected gradient matches its Full-KL
reference. It also checks that the native loss adapter uses actual collection
log-probabilities and terminal rewards, rather than PPO old-policy scores or
group advantages.

Eight rejection checks exercise upstream validation of missing/nonfinite
records, nonboolean/empty action masks, insufficient sequence MC columns, and
missing native sampler/global-batch contracts. Optional tests verify that
context tokens, historical sampler records, and rewards receive no gradients.

The fixture's terminal status and independent auxiliary distribution are known
by construction. The upstream tensor API cannot infer complete-trajectory
provenance, sampling independence, sampler-version identity, or identical
history/sampling transforms from arbitrary arrays. Different current and
historical policies are valid off-policy inputs. A mismatch in their collection
metadata requires a separate collector validation layer.

These checks establish CPU loss execution and gradient correctness on a small
fixture. They establish no inference, rollout, GPU-training, or learning speed
increase. The source and practical integration boundaries are documented in
[`docs/KLPO_SOURCE_REVIEW.md`](../../docs/KLPO_SOURCE_REVIEW.md).
