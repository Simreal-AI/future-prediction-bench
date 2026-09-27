# Rollout allocation and checkpoint research, 2026-09-27

Three concrete code paths now separate **where continuations are spent**,
**how scored siblings receive credit**, and **whether sandbox state needs
capture**. None has completed policy training in this benchmark environment. A rollout-count budget is not
a token, GPU-time, or wall-time budget; a changed allocation alone cannot
prove faster learning.

## Executed official code

| Source | Exact revision and execution | Evidence and boundary |
| --- | --- | --- |
| [Crab](https://github.com/open-agent-infra/crab) | `9607d61a41dc44358cf078c4b438bfd971c8ee9d`; real upstream Linux process monitor and host policy, no doubles | [Actual x86 QEMU/TCG probe](OFFICIAL_CRAB_MICROVM.md): 8 → 3 full saves, 18.101 → 16.257 s median controlled workload. No CRIU/ZFS, eBPF filesystem monitor, or software grader. |
| [Tree-GRPO](https://github.com/AMAP-ML/Tree-GRPO), [paper](https://arxiv.org/abs/2509.21240) | `19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59`; two unchanged upstream advantage-function ASTs with real CPU PyTorch | [CPU report](measurements/official_tree_advantages_cpu_2026-09-27.json), [runbook](../examples/official_tree_grpo/README.md): inter-plus-inner and sequential normalization, sibling ordering and masks. Synthetic scored tensors; no full trainer. |
| [VIP](https://github.com/HieuNT91/VIP), [paper](https://arxiv.org/abs/2602.01601) | `f7bd18915467f50a0d8565b4f16afaef0741a96f`; unmodified `Allocator` with real NumPy/SciPy | [CPU report](measurements/official_vip_allocator_cpu_2026-09-27.json), [runbook](../examples/official_vip/README.md): exact 64-rollout allocation `[5,11,12,12,12,4,4,4]` for eight supplied probabilities. No trained predictor or trainer entry point. |

The Tree-GRPO bridge uses explicit question and tree identities and distinct
response/policy-loss masks. A tree cannot mix unrelated questions; each tree
needs at least two scored responses. Context tokens and padding cannot become
policy-loss targets. Eight optional tests exercise the actual upstream
functions when real PyTorch and the clean pinned checkout are available.
These checks prepare credit tensors; they do not supply tokenization,
behavior-policy log probabilities, an optimizer, or proof of on-policy data.

VIP's allocator executes separately from its training package. In the pinned
release, the package initializer imports an `allocate_rollout` symbol absent
from allocation exports. Direct allocator loading avoids that unresolved
entry point; it does not establish successful trainer integration. The input
probabilities and 0.8 clamp are recorded so the plan can be reproduced.

## Independent TRACE allocation implementation

[TRACE](https://arxiv.org/html/2606.11119), equations 12–16, motivates
[`allocate_contrast`](../future_prediction_bench/contrast_allocation.py).
This is an **independent equation-level implementation**; no author code
release has been established. Given a supplied success probability `p` and
`k` continuations, the implemented objectives are:

- Root: `1 - p**k - (1-p)**k`, the probability of obtaining both binary
  outcomes. A selected root receives at least two continuations.
- Prefix with factual reward 1: `1 - p**k`; prefix with factual reward 0:
  `1 - (1-p)**k`, the probability of a contrasting outcome. A selected prefix
  receives at least one continuation.

Bounded dynamic programming finds an exact integer-budget plan under a
per-anchor cap. Deterministic ordering resolves equal objectives reproducibly.
Six tests in [`test_contrast_allocation.py`](../tests/test_contrast_allocation.py)
include exhaustive-search agreement for both stages, ordering invariance,
budget feasibility, and input rejection. Terminal anchors and unresolved
factual rewards cannot be expanded. The result marks the probability
predictor untrained and training speedup unmeasured. The formula currently
requires binary resolved reward; it is not a general allocator for categorical
forecast probability vectors.

## Additional sources reviewed

[KLPO's official repository](https://github.com/yifanzhang-pro/KLPO) is now
pinned at `30c0ae8c3fa8f56213d6b57bc88b18ebee8ed696`. Its unmodified token/sequence
loss and native loss adapter execute with real CPU PyTorch/autograd; exact
auxiliary-draw enumeration agrees with Full-KL gradients within `6.67e-16`.
The author's four-action CPU toy completed four actual AdamW updates.
These checks do not collect LLM rollouts or train in this environment. The
[source review](KLPO_SOURCE_REVIEW.md) records the required historical
sampler records and independent auxiliary draws, plus the pinned backend's
synchronous-collection restriction and unvalidated GPU reproduction.

[Exact checking of agent execution edits](https://arxiv.org/abs/2608.22928)
has [author-linked code](https://github.com/eunomia-bpf/agent-check-restore-safety),
pinned at `c3fbdae3675b7e83b9b1e261eea37d9d4c066d60`. Nineteen upstream
`test_exact_history_realization.py` tests passed in the local checkout.
This is source-level checking of execution-edit semantics, not a deployed
sandbox guard or a speed measurement. It provides a useful separate
correctness criterion when extending checkpoint/fork behavior.

A separate [experimental host operation journal](RESTORE_EFFECT_GATEWAY.md)
now applies a narrower independent rule: unresolved external outcomes block
snapshot/restore, while completed stable calls replay stored results. Nine
SQLite/fake-VM unit tests cover that control logic and versioned provider keys.
This is not the upstream exact checker or an actual provider/QEMU integration.

[Checkpoint handoff](https://arxiv.org/abs/2609.19636) is a further September
2026 reference; an author implementation has not been established. It has
not been replicated. [Branching Policy Optimization](https://arxiv.org/abs/2607.14171)
remains a motivation for the separate [sibling-return preparation](BRANCH_ADVANTAGES.md),
with no reproduced policy-training result.

RollArt concerns scheduling generation, environment interaction, and reward
work across a training system. It is distinct from a statistical allocation
rule such as VIP or TRACE, and from guest checkpointing. Existing
[same-host scheduling measurements](ROLLART_CONTROL_PLANE.md) and the
[official-code audit](OFFICIAL_CODE_INTEGRATION_AUDIT.md) retain their original
boundaries. No ratios from those separate experiments are multiplied together.

## Required next comparisons

Connecting these mechanisms requires a scored, versioned model collector
with behavior log probabilities, supported tree identities and loss masks,
and an actual optimizer. Compare uniform versus adaptive budgets at matched
model, tasks, token/GPU budget, and held-out success; report both environment
cost and full trained-token or task-success throughput. A privileged x86-64
Linux host can test Crab's real CRIU/ZFS/eBPF backend without a GPU. A model
training comparison, for example with a Qwen 8B policy, separately needs a
suitable GPU host. Neither requirement is replaced by the present CPU and
controlled-VM checks.
