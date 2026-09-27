# Original AReaL staleness manager on CPU

`check_staleness.py` validates a clean pinned upstream AReaL checkout, imports
the original package normally, executes its real `StalenessManager`, and can
run the unchanged official CPU test suite. The version input is an explicit
integer control-plane counter; no model tokens or gradients are generated.

See [the source review](../../docs/AREAL_SOURCE_REVIEW.md) for the commit,
source hashes, execution results, dependency scope, and a concrete admission
gate integration contract for our rollout pipeline.

```bash
python3 examples/official_areal/check_staleness.py \
  --source runs/official-areal \
  --output runs/official-areal-cpu-check \
  --upstream-tests
```

The upstream package and genuine dependencies must import. The source and
official tests are not patched, extracted through ASTs, or replaced. This
probe measured capacity-control correctness and does not establish inference
or GPU training throughput. Upstream AReaL is Apache-2.0 licensed.

Source and output must be separate: the probe rejects either path containing
the other before creating any output. It disables imported bytecode writes
and pytest cache writing to preserve the pinned checkout.
