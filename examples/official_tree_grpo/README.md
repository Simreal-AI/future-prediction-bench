# Pinned Tree-GRPO outcome advantages on CPU

The optional bridge executes two **unmodified function ASTs** from the
[official Tree-GRPO repository](https://github.com/AMAP-ML/Tree-GRPO), pinned
to [`19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59`](https://github.com/AMAP-ML/Tree-GRPO/commit/19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59).
The functions are `compute_grpo_outcome_advantage` and
`compute_tree_grpo_outcome_advantage` in `verl/trainer/ppo/core_algos.py`.
This is direct function execution using real PyTorch; it does not import or
execute the distributed trainer. Other module imports are omitted rather
than replaced with doubles. No upstream source is vendored.

With real PyTorch installed in the selected Python environment:

```bash
git clone https://github.com/AMAP-ML/Tree-GRPO.git runs/official-tree-grpo
git -C runs/official-tree-grpo checkout 19bf3fa0c74d8b9ba70619a09a2fd480c31b9d59
python3 -m examples.official_tree_grpo.check_advantages \
  --checkout runs/official-tree-grpo \
  --output runs/official-tree-cpu/measurement.json
```

The source checkout must be clean, and output must not already exist. The
[published CPU result](../../docs/measurements/official_tree_advantages_cpu_2026-09-27.json)
uses synthetic scored tensors and records the source hash and PyTorch
version. It exercises `tree` inter-plus-inner advantages and `tree_2norm`
sequential normalization, confirms sibling reward ordering, excludes prefix
context from policy-loss targets, and masks padding. Equal-reward siblings
have zero local contrast under sequential normalization. Cross-question
trees and singleton trees are rejected.

[`OfficialTreeAdvantageBridge`](../../future_prediction_bench/official_tree_advantages.py)
also validates tensor shape, finiteness, binary masks, nonempty loss targets,
and cohort identities. Eight optional tests in
[`test_official_tree_advantages.py`](../../tests/test_official_tree_advantages.py)
exercise the bridge when its dependencies and pinned checkout are available.
There is no learned policy, model rollout, behavior logprob collector,
complete loss, or optimizer update. Outputs retain `trainer_ready=false`;
CPU advantage correctness is not evidence of training acceleration.
