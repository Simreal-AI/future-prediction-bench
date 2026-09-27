# Prepared full-VM templates for RealWorldEnv

[`PreparedMicroVMTemplate`](../future_prediction_bench/prepared_microvm.py) is an opt-in environment setup path for the pinned public Boltons 26.0.0 coding fixture. It closes the gap between a standalone QEMU snapshot benchmark and actual [`RealWorldEnv`](../future_prediction_bench/realworld.py) episodes: every spawned child receives its own writable qcow2 disk and restored CPU, RAM, and devices, then uses the ordinary file tools, action budget, submission snapshot, and host-private verifier. The default cold-start adapter is unchanged. With no extra contract, preparation retains the default **full-VM restore before every hidden case**. An explicitly supplied stateless contract can be composed with preparation as a separate verifier mode; both cold and prepared sides of that comparison use the same contract.

The trusted host starts from a pristine qcow2 whose digest matches the asset manifest and task artifact binding. Preparation checks the exact published Boltons source archive digest, seed workspace tree, `strutils.py`, and 14-case verifier digest. It checks the guest file against the host seed, rejects unexpected live user-space processes, and exports a full QEMU `savevm` checkpoint before any policy action. It writes RAM/tmpfs and ext4 markers into the parent *after* export; each child must prove both markers absent. The parent closes before any child starts. Reopening requires the caller-held prepared ID; every spawn checks the manifest, boot-artifact digests, seed files, verifier, visible check, and exact frozen task digest, then verifies each cloned child disk against the template digest **before boot**. The sidecar records host paths and remains under ignored `runs/`, outside the release archive. Digests detect changes against a trusted ID before the child boots; they are not signatures or a multi-tenant artifact service.

Each child runs `RealWorldEnv.reset` only after a template clone and restore. Reset rechecks the pristine guest source and process census, so inherited policy edits or background workers fail setup rather than receiving a reward. The policy-visible opening matches the cold adapter. The hidden expected outputs remain on the host, and VM or verifier failures retain the existing infrastructure/pending behavior. Template setup and child spawn are trusted-host operations, never policy tools.

After creating the pinned task and assets as described in [the microVM guide](MICROVM_ENV.md), run the paired benchmark:

```bash
python3 -m examples.realworld_boltons26.benchmark_prepared_env \
  --task-dir runs/boltons-vm-task \
  --assets-dir runs/boltons-vm-assets \
  --output runs/boltons-prepared-realworld \
  --repetitions 3
```

The script refreshes only the public fixture's expired local action window, freezes one task binding for all conditions, and alternates cold and prepared order. Repair and untouched baseline each run as complete independent episodes. Each wall time starts before that episode's disk provisioning and ends after `RealWorldEnv.verify`: it includes VM boot or template restore, policy-tool execution of the *scripted* actions, submitted-state snapshot, all 14 hidden cases, and an isolation probe. The one-time parent boot and export is measured separately and charged in the amortized ratio. Policy inference, model sampling, gradient updates, and RL convergence are absent.

On the Apple M3 / 8 GB development host, three alternating repetitions returned exactly **14/14 cases and reward 1.0** for the repair and **7/14 cases and reward 0.0** for the baseline under both conditions. Opening observations, every policy action observation, and all per-case evidence matched exactly. The six cold episodes had median **8.117 s**; the six template episodes had median **3.647 s**, a **2.226×** steady-state time ratio. One-time template preparation took **6.100 s**. Across the six episodes per condition, cold time totaled **48.922 s** and prepared time including setup **27.176 s**, a **1.800×** amortized ratio. Repair medians were 8.226/3.726 s (cold/prepared); baseline medians were 7.629/3.008 s. These are one-host infrastructure results on one already-known repair, not model-training acceleration or millisecond full-VM cloning. The [sanitized measurement report](measurements/prepared_realworld_episodes_v0.6.0.json) records the paired timings, hashes, outcomes, and exact action/case parity.

The complete child path remains seconds because it clones and starts QEMU, restores a full snapshot, executes policy actions, and grades the task. Faster guest-local process forks or a stateless hidden-case batch have a narrower isolation contract and are evaluated separately. A production trainer would also need controlled policy rollout workers, behavior log probabilities, update steps, scheduling, and repeated tasks before claiming training throughput.

## Optional stateless-verifier composition

The same prepared template can be used with the [task-declared stateless verifier](STATELESS_VERIFIER.md). This is an explicit second condition, not an automatic change to the default. The frozen task and prepared manifest bind the exact stateless task-source, contract, and trusted guest-helper SHA-256 digests in addition to the ordinary task, verifier, and asset digests. Each child checks those files again before its policy episode. The submitted snapshot must pass the existing process-quiescence gate, and every hidden case receives a separate guest mount/PID namespace. The host still holds expected outputs and scores exact bytes.

```bash
python3 -m examples.realworld_boltons26.benchmark_prepared_env \
  --task-dir runs/boltons-vm-task \
  --assets-dir runs/boltons-vm-assets \
  --stateless-contract examples/realworld_boltons26/stateless_contract.json \
  --preinstall-stateless-helper \
  --output runs/boltons-prepared-stateless \
  --repetitions 3
```

The `--preinstall-stateless-helper` flag seals only the **generic, trusted** guest helper in the clean parent before export. Preparation checks the helper's bytes and proves the case-code file is absent; each child rechecks both before its episode and again before submission. Hidden case source is uploaded only after submission, and expected outputs never leave the host. Omitting this flag retains upload-at-submit behavior.

In three alternating repair/baseline pairs on the same host, cold stateless episodes had median **7.029 s** and prepared episodes with the generic helper preinstalled had median **2.122 s**, a **3.313×** steady-state time ratio. Preparation took **6.286 s**; totals over six episodes per condition were **41.702 s cold** and **18.986 s prepared including setup**, a **2.197×** amortized ratio. Repair medians were 7.239/2.379 s (cold/prepared), and baseline medians were 6.605/1.845 s. All pairs matched opening observations, every policy action observation, and every hidden case exactly: repair **14/14, reward 1.0**; baseline **7/14, reward 0.0**. The [sanitized composed report](measurements/prepared_stateless_episodes_v0.6.0.json) records the paired outcomes. The stateless verifier's case-isolation contract is narrower than a full VM restore per case, so this ratio belongs only to the explicitly declared fixture.

A separate [five-repetition stability report](measurements/prepared_stateless_stability_5pair_v0.6.0.json) repeated that same composed condition on the same host and fixture, with **10 cold and 10 prepared complete episodes**. Median cold/prepared time was **6.824/1.935 s**, a **3.525×** steady-state ratio. This ten-episode median combines five repaired episodes with four policy actions and five untouched baseline episodes with two: repair medians were **7.101/2.323 s** cold/prepared (**3.056×**), while baseline medians were **6.583/1.603 s** (**4.107×**). Each branch ratio divides its own cold and prepared medians. Template setup took **6.353 s**; across those 10 episodes per condition, cold time totaled **68.447 s** and prepared time including setup **26.415 s**, a **2.591×** ratio. Each repaired/baseline condition pair retained exact opening, policy action observation, 14-case evidence, and reward parity. The earlier three-pair and this five-repetition run are reported separately; neither set is pooled into a larger estimate, and neither measures policy inference or training.

The separate [v2 bounded-edit comparison](SMALL_EDIT_MICROVM.md) freezes a different task/tool contract and runs 40 full QEMU episodes. Its bounded `replace_text` method measured **6.577/1.728 s** mixed-branch cold/prepared medians, a **3.800×** ratio of total times at steady state and **2.751×** after charging its 6.610-s preparation to 10 prepared episodes. The [v2 report](measurements/prepared_small_edit_v0.7.0.json) is not pooled with the v1 reports above.

The [later template-hash A/B](measurements/prepared_template_hash_ab_v0.8.0.json) uses that same v2 task to isolate one host-side spawn change. Five alternating pairs (80 full episodes) compared the former three source-template SHA reads per spawn against a child-check-only path. The latter still validates each cloned child disk before boot. Prepared episodes totaled **36.632 → 31.280 s** across 20 episodes per variant (**1.171×** throughput), with identical policy observations, all 14 case outcomes, reward, source digest, and child isolation. This result does not measure a faster `savevm` or `loadvm` primitive.

After preparing the pinned v2 task and rebound assets as in [the bounded-edit guide](SMALL_EDIT_MICROVM.md), reproduce the controlled comparison with:

```bash
python3 -m examples.realworld_boltons26.benchmark_template_hash_ab \
  --task-dir runs/boltons-v2-task \
  --assets-dir runs/boltons-v2-assets \
  --stateless-contract examples/realworld_boltons26/stateless_contract_v2.json \
  --output runs/boltons-v2-template-hash-ab --pairs 5
```

The public harness freezes one task deadline per run, alternates legacy/candidate order, checks all matched observations and case outcomes, and writes complete per-condition reports plus `comparison.json` under the ignored output directory. Its legacy arm adds the former three source hashes around the same child-digest-before-boot spawn path. A separate one-pair calibration recorded a **0.0679 s median** for each source-template SHA pass; that calibration is reported separately in the [A/B measurement](measurements/prepared_template_hash_ab_v0.8.0.json).

To isolate the helper preinstallation itself from boot and template savings, a [matched prepared-template A/B](../examples/realworld_boltons26/benchmark_preinstalled_helper.py) creates two clean templates and alternates complete episodes:

```bash
python3 -m examples.realworld_boltons26.benchmark_preinstalled_helper \
  --task-dir runs/boltons-vm-task \
  --assets-dir runs/boltons-vm-assets \
  --stateless-contract examples/realworld_boltons26/stateless_contract.json \
  --output runs/boltons-helper-ab \
  --repetitions 3
```

Across three matched pairs, uploading the helper at submit had median complete-episode time **2.256 s**, versus **2.080 s** with helper-only preinstallation (**1.085×**). Including each template's own one-time setup, total time over six episodes was **19.771 versus 19.164 s** (**1.032×**). The [path-free A/B report](measurements/prepared_helper_ab_v0.6.0.json) retains exact opening, action-observation, and per-case parity with all timing samples. This small observed difference on one host is not a model-training speedup.

Each final three-pair timing report above establishes parity of the opening observation, **every policy action observation**, and host-private case evidence. Separate one-pair QEMU runs independently confirmed the strengthened action-observation checks for the default verifier, the composed stateless verifier, and the helper-only A/B. The [action-observation parity report](measurements/prepared_action_observation_parity_v0.6.0.json) records those earlier checks and their raw-report digests. The timing ratios come from the final three-pair reports.
