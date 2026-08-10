# EXP-027: CutMix-Off, RandAugment-On Refinement Window

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-027
- **Commit**: (pending — committed on loop success)
- **Outcome**: failed — valid no-improvement at `93.69%`

## Implementation Notes

### Summary

Added a module-level forkserver-picklable collator holding a synchronized policy flag. It snapshots the flag once, preserves the accepted fork-RNG-isolated 50% alpha-1 CutMix while enabled, and returns explicit provenance. The parent flips the flag once after the first completed step at 70% counted time without breaking the iterator; N1/M7 and all workers persist until the unchanged 80% weak-loader rebuild.

### Surprises & Discoveries

The adversarial plan review predicted that a locked `multiprocessing.Value` embedded in a forkserver-pickled collator would fail. A guarded two-worker DataLoader probe on the actual Python/PyTorch environment disproved that concern: it observed `[True, True, True, True, False, False, False, False]` across the live flip and shut both workers cleanly.

### Decisions

- Count policy-on hard, policy-on CutMix, policy-off hard, and post-request policy-on drain separately because target rank cannot distinguish natural eligible hard batches from deliberately disabled mixing.
- Enforce the 70.5% drain deadline in production, while the preflight additionally enforces the registered 24-delivery limit.
- Generate an EXP027-owned immutable corpus once rather than depending on EXP026's ignored local corpus.

## Experimental Adjustments

- Plan review required the lifecycle gate to recreate iterators at real epoch boundaries and removed the hard dependency on EXP026's ignored corpus. Both changes strengthen diagnostic fidelity without changing the candidate.

## Pre-Production Evidence

- Immutable corpus SHA-256 `3af2f47731c1184b18cad3d51849619cdda08fe30c2a0a73f1ba8ad08d70d70e`; all 200 source/target/RNG-state digests and eight worker identities validated. Natural semantic replay produced 97/200 CutMix decisions.
- Common-prefix exact continuation passed with candidate/control terminal loss-EMA ratio `0.939069` and zero candidate-only concentration events; all BN counters and momentum state were complete.
- Lifecycle passed: 7,495/15,000 eligible deliveries were CutMix (`49.967%`); the live flip drained after 8 deliveries, all eight workers observed off, then 5,000 hard N1/M7 deliveries completed. Eight strong/eight weak workers stopped, weak rebuild was `2.893s`, and no child remained.
- Five-pair timing passed: mean counted ratio `1.006886`, worst pair `1.016473`, control/candidate CV `0.846%/0.828%`, projected `26,714` steps and `332.98s` total. Every candidate drained after 14 deliveries, reached all workers, and produced 86 hard-off batches; CutMix was `51.261%` while eligible.
- Maximum candidate projected wall/count was `1.036746`; maximum candidate-control delta `0.003113`; peak allocation `598.676 MiB`; allocation growth zero; every weak rebuild was below `3.10s`.

Evidence hashes:
- Preflight report: `b810bef47929dd180836c61da736e7d62923c1214da9b8e9c334a8592a95698c`.
- Timing report: `0c5208b72e518a571b2f1d59d14099361a265b5ca513159b310856a4e542462a`.

## Run Log

### Run 1

Metadata:
- **Job ID**: local process (single scored run)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: completed, exit 0, no retry
- **Started**: 2026-08-06T17:41:29Z
- **Ended**: 2026-08-06T17:47:28Z

Description:
- One seed-42 H20 run of the accepted recipe with CutMix disabled at 70% while RandAugment remains active until 80%, conditional on semantic, immutable-corpus, worker propagation/lifecycle, and paired timing gates. Formal success requires `best_test_acc>=94.25%` with at least 26,629 steps and no scored retry.

Observations:

- The request fired at `70.002%` and the last eligible prefetched batch arrived at `70.051%`; 13 post-request policy-on batches drained before 2,688 hard N1/M7 batches.
- Accuracy dropped to `82.96%` at the request epoch, recovered to only `88.76%` at the 80.0% switch (0.97 below EXP010), and reached `93.00%` first weak.
- The weak tail peaked at `93.69%` in epoch 69 and finished `93.57%`; final NLL `0.2031` was worse than EXP010's `0.1934`.

Key Metrics:

- `best_test_acc=93.69%`; baseline `94.15%`; delta `-0.46` points; required `94.25%`.
- `final_test_acc=93.57%`; `final_test_loss=0.2031`; `training_seconds=300.0`; `total_seconds=333.5`; `startup_seconds=1.0`.
- `num_steps=26986`; `num_epochs=70`; `num_params=1,073,962`; `peak_vram_mb=598.7`.
- Production log SHA-256: `6341b24361945bece4d70b22877a7dfe09b002f63099483442050774f6dbb2aa`.

## Verification Results

### Conditions Checked

- PASS integrity: exit zero; ten finite summary fields; one H20; only `train.py` tracked-modified; one scored run; 300.0 counted/333.5 total seconds; 26,986 steps; 1,073,962 parameters.
- FAIL primary metric: `93.69% < 94.25%`; classify as no-improvement and do not retry.
- PASS transition protocol: one request at 70.002%, last policy-on at 70.051%, one 80.0% switch, eight stopped workers, 13 delivered drain batches, and 2,688 hard N1/M7 bridge batches.
- PASS count identity: 9,524 policy-on hard + 9,411 CutMix + 2,688 policy-off hard = 21,623 strong batches; eligible CutMix fraction `49.702%`.
- PASS evaluator cadence: 19 unique evaluation epochs, each at most once.

### Informational Metrics

- Switch/first weak/best/final accuracy: `88.76% / 93.00% / 93.69% / 93.57%`.
- Final NLL `0.2031`; best-final gap `0.12` points; 598.7 MiB peak; 26,986 updates.

## Errors & Dead Ends

## Human Notes

> Autopilot session; no human intervention requested.
