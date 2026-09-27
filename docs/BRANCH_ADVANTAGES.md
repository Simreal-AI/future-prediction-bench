# Branch-local return preparation for real-world RL

The real-world environment can checkpoint an active coding episode and run independent sibling suffixes from that state. [`prepare_sibling_advantages`](../future_prediction_bench/branch_advantages.py) now consumes the resulting **graded train trajectories** and calculates a leave-one-out return advantage for each sibling: its reward minus the mean reward of the other siblings. This is a host-side text-trajectory preparation step, not an optimizer or a demonstrated learning gain.

The preparation requires the collection plan's exact sibling count, current policy ID, and timezone-aware preparation cutoff. It rejects missing siblings, changed policy IDs, rewards unavailable by that cutoff, mixed task/reward hashes, different checkpoint or visible prefixes, duplicate episode/branch IDs, altered visible-event digests, and a branch with no post-checkpoint policy action. The checkpoint lineage now records the count of visible prefix events. The output lists only **post-checkpoint policy-action event hashes** as candidates for the local sibling signal; all shared-prefix events remain context. No reward, branch metadata, or hidden verifier output is returned to the policy during its episode.

```python
from future_prediction_bench.branch_advantages import prepare_sibling_advantages
from datetime import datetime, timezone

prepared = prepare_sibling_advantages(
    [good_branch.export_trajectory(), bad_branch.export_trajectory()],
    expected_siblings=2,
    current_policy_id="checkpoint-001",
    as_of=datetime.now(timezone.utc),
)
# A reward-1 branch and reward-0 sibling get +1 and -1 local advantages.
assert prepared["trainer_ready"] is False
```

The [Branching Policy Optimization preprint](https://arxiv.org/html/2607.14171) motivates sampling from common intermediate sandbox states and comparing siblings at the branch point. It reports gains on its own WebShop, ALFWorld, and SWE-bench Verified training setups. Our implementation takes only the local sibling-baseline arithmetic and frozen-state cohort check. It does **not** implement that paper's entropy-based branch selection, prefix-credit propagation, PPO loss, token likelihood ratios, or reported training results. The shared prefix is identical across siblings, so its local contrast alone cannot tell which earlier action should have reached that state; a separate upstream credit estimator would be required.

Before a model update, the runner must preserve exact model requests/responses, tokenize with a pinned chat template, derive policy-token and observation masks, record behavior token log probabilities, and choose a supported optimizer and policy-staleness rule. Equal-reward siblings correctly yield zero local advantage. The sample above is a fixture arithmetic check; only new, disjoint repository tasks and matched model/token budgets can establish whether branching improves learning or wall-clock training throughput.
