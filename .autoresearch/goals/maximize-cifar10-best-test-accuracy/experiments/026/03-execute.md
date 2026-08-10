# EXP-026: Exact-Corpus Balanced Mixup/CutMix Retry

## Execution

Overall Status & Info:
- **Created**: 2026-08-06
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/maximize-cifar10-best-test-accuracy-026
- **Commit**: (pending — committed on loop success)
- **Outcome**: no-improvement — valid `94.22%` best missed the `94.25%` gate

## Implementation Notes

### Summary

Changed only the strong worker policy: one fork-RNG-isolated draw now selects 50% hard, 25% alpha-1 CutMix, or 25% alpha-0.4 Mixup. The loop explicitly unpacks strong triples versus weak pairs, validates provenance/target rank, and reports all three counts at the unchanged switch. Model, optimizer, schedule, evaluator, weak loader, seed, and 1,073,962 parameters remain unchanged.

### Surprises & Discoveries

Direct seeded semantic tests prove the candidate is bitwise equal to the accepted policy for `u<0.25` CutMix and `u>=0.5` hard branches; only the registered middle quarter differs.

### Decisions

- Return an integer policy kind from the worker collator because target rank cannot distinguish CutMix from Mixup.
- Keep the categorical draw and selected torchvision transform in one CPU `fork_rng` region, preserving the accepted worker-transform RNG isolation contract.

## Experimental Adjustments

- Preflight controller attempt 1 reached the external 240-second cap after writing and validating the complete 200-batch immutable corpus and semantic report (`control=[103 hard, 97 CutMix]`, `candidate=[103 hard, 47 CutMix, 50 Mixup]`). No scored training ran and no controller failure was observed. The retry reuses that content-addressed corpus instead of regenerating it, leaving additional wall-clock budget for the required 20,000-batch lifecycle gate.
- Preflight controller attempt 2 also reached 240 seconds with corpus regeneration removed. An isolated exact-replay control child completed successfully in 12.06 seconds, localizing the excess runtime to the 20,000-batch real-loader lifecycle exercise. The final controller retry keeps all 20,000 batches, adds auditable stage progress, and raises only the external diagnostic timeout.
- The final 480-second preflight completed successfully. The subsequent timing controller's original 300-second wrapper completed its conditioner, four full alternating pairs, and trial-5 control, then exited 124 before trial-5 candidate. This produced no timing verdict and consumed no scored run. The diagnostic retry keeps the registered workload and fresh-process design unchanged, persists each complete pair incrementally, and raises only the outer wrapper to the measured runtime.
- Timing attempt 2 completed all five pairs. Its first analysis charged the one-time ~3-second strong-to-weak transition against only ~13.7 seconds of sampled GPU work, making raw probe wall/count 1.25-1.27 and the <=1.07 production-overhead gate structurally impossible. Raw results otherwise passed every gate. The accounting correction preserves all measurements and the registered threshold, separates measured per-step waits from one-time transition overhead, and projects both over the registered 300-second production horizon before verdict.

## Run Log

### Run 1

Metadata:
- **Job ID**: local process (single scored run)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.5-gpt-5-6-sol-adversarial/run.log`
- **WandB**: N/A
- **Status**: completed, exit 0, no retry
- **Started**: 2026-08-06T16:57:46Z
- **Ended**: 2026-08-06T17:03:25Z

Description:
- One seed-42 H20 run of balanced hard/CutMix/Mixup geometry on the accepted recipe, conditional on natural immutable pre-policy replay, exact safety, lifecycle, and real-loader timing gates. Formal success requires `best_test_acc>=94.25%` without changing total mixed probability.

Observations:

- The strong trajectory reached `89.68%` at 60%, then fell to `88.13%` at the 80.0% switch, `1.60` points below accepted EXP010's `89.73%` switch.
- The hard weak tail recovered immediately to `93.37%`, `0.21` above EXP010's first weak checkpoint, but final NLL `0.1975` remained worse than EXP010's `0.1934`.
- Best accuracy reached `94.22%` at epoch 68, then regressed to `94.05%` at the final epoch. The extra `370` updates over EXP010 did not clear the accuracy margin.

Key Metrics:

- `best_test_acc=94.22%`; baseline `94.15%`; delta `+0.07` points; required `94.25%`.
- `final_test_acc=94.05%`; `final_test_loss=0.1975`; `training_seconds=300.0`; `total_seconds=332.4`; `startup_seconds=1.0`.
- `num_steps=27268`; `num_epochs=70`; `num_params=1,073,962`; `peak_vram_mb=598.7`.
- Run log SHA-256: `8c44f6afff741e904ba5b6727e4d63e9b56f56d2e9b168efa9a67aec3732ea82`.

## Verification Results

### Conditions Checked

- Immutable corpus: 200 unfiltered natural records, SHA-256 `4386e6915d0bf3bb1f1f5dfdc6c36581758308d23df89eef2dd41202bb2e41e3`; all source/input/target/RNG-state digests valid.
- Semantic counts: control `[103 hard, 97 CutMix, 0 Mixup]`; candidate `[103 hard, 47 CutMix, 50 Mixup]`; shared branches bitwise equal, surrounding RNG unchanged, all targets valid.
- Exact 200-step safety: pass; no candidate-only concentration event; candidate/control terminal loss-EMA ratio `0.934313`; complete BN counters and momentum state.
- 20,000 real collations: `[10059 hard, 5038 CutMix, 4903 Mixup]` = `[50.295%, 25.190%, 24.515%]`; eight strong and eight weak workers stopped, weak rebuild `2.912s`, zero live children.
- Five-pair real-loader timing: mean counted ratio `1.003966`, worst pair `1.027406`, control/candidate CV `1.249%/0.473%`, projected `26,791` steps, projected total `332.11s`.
- Candidate loader/memory: all median waits `<0.91%` and p95 waits `<1.40%` of GPU time; minimum headroom `56.04x`; maximum projected wall/count `1.038259`; maximum projected wall/count delta `0.002019`; peak `598.68 MiB`; allocation growth zero; every weak rebuild `<5s`; no child leak.
- Timing candidate geometry across 5,500 natural strong batches: `[2740 hard, 1305 CutMix, 1455 Mixup]` = `[49.818%, 23.727%, 26.455%]`.
- Production completed once with exit zero and all ten finite summary fields. It used exactly one H20, 300.0 counted seconds, 332.4 total seconds, and 27,268 steps.
- Primary condition failed: `94.22% < 94.25%`. All integrity and runtime conditions passed, so the result is a valid no-improvement, not invalid or crash.
- Exactly one switch occurred at 80.0%; eight strong workers stopped. Production geometry was `[10963 hard, 5450 CutMix, 5423 Mixup] / 21836` = `[50.206%, 24.958%, 24.836%]`.
- Nineteen evaluation epochs were unique and each was evaluated once. First weak accuracy was `93.37%`; best/final were `94.22%/94.05%`; final NLL was `0.1975`.

### Informational Metrics

- Preflight report SHA-256: `acf73dbe76897add301ddc42476f7977e10749150672cd6f521bce95ab05f30b`.
- Semantic report SHA-256: `99615d4f180ac198b6737dc1ec4277b4c40b344736a444c4b08c2d7384837134`.
- Timing report SHA-256: `f312d5fbded346e853cac4a423c4a5b3e71f79e5eba7d8787fe2f2d20dc3679f`.
- Production log SHA-256: `8c44f6afff741e904ba5b6727e4d63e9b56f56d2e9b168efa9a67aec3732ea82`.

## Errors & Dead Ends

- Controller attempt 1 exited 124 at the outer timeout during the long lifecycle exercise. The timeout cleanup reported semaphore resource-tracker warnings, but a process audit found no surviving controller or worker processes.
- Controller attempt 2 likewise exited 124 with no failure trace. The timeout wrapper again cleaned up all worker processes; this was not a candidate-policy veto.
- Timing attempt 1 exited 124 at its outer 300-second cap after nine of ten measured arms. A process audit is required before retry; incomplete in-memory pair data are not reused.
- Timing attempt 2 initially reported only wall/count failures because its analysis concentrated one-time transition overhead into the short probe region. The raw report showed all comparison and resource gates passing. Reanalysis over the registered production horizon corrected the accounting and passed without rerunning an arm or changing a threshold.

## Human Notes

> Autopilot session; no human intervention requested.
