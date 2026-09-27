# Pinned VIP rollout allocation on CPU

This example directly loads the unmodified `Allocator` implementation from
the [official VIP repository](https://github.com/HieuNT91/VIP), commit
[`f7bd18915467f50a0d8565b4f16afaef0741a96f`](https://github.com/HieuNT91/VIP/commit/f7bd18915467f50a0d8565b4f16afaef0741a96f),
in `src/vip/allocation/bernoulli_allocation.py`. Real NumPy and SciPy execute
the allocation; no dependency doubles or vendored upstream files are used.

```bash
git clone https://github.com/HieuNT91/VIP.git runs/official-vip
git -C runs/official-vip checkout f7bd18915467f50a0d8565b4f16afaef0741a96f
python3 -m examples.official_vip.check_allocator \
  --checkout runs/official-vip \
  --output runs/official-vip-cpu/measurement.json
```

Install NumPy and SciPy in the selected Python environment first. The
checkout must be clean, and output must be new. The
[published result](../../docs/measurements/official_vip_allocator_cpu_2026-09-27.json)
uses supplied success probabilities `[0.01, 0.1, 0.2, 0.4, 0.5, 0.8, 0.95,
0.99]`. With an exact 64-rollout budget and bounds of 4–12 per question,
the upstream allocator returns **`[5, 11, 12, 12, 12, 4, 4, 4]`**, compared
with uniform `[8, 8, 8, 8, 8, 8, 8, 8]`. The run checks the budget total and
integer bounds and records the source SHA-256. The upstream implementation
clamps probability inputs at 0.8 for this rule.

These probabilities are inputs to the probe; no success predictor is trained
or evaluated. No model rollouts or optimizer step run, so the changed budget
distribution does not establish a learning or wall-time benefit. The pinned
package initializer imports `allocate_rollout`, which is absent from its
allocation module's exports. This example intentionally loads the reviewed
allocator file directly and does **not** claim that the package or trainer
entry point executes successfully. Further integration must fix or use a
verified supported entry point before comparing training outcomes.
