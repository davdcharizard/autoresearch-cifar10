# EXP-005: Clean-Gated Last-Mini-Batch Self-Distillation

## Execution

Overall Status & Info:
- **Created**: 2026-08-05
- **Autonomy**: autopilot
- **Experiment Branch**: tree-autoresearch/maximize-cifar10-best-test-accuracy-exp-005
- **Base Node**: 004
- **Commit**: `49f6f64`
- **Outcome**: failed — valid run completed, but `best_test_acc=95.28%` missed the 95.50% parent-relative gate

## Implementation Notes

### Summary

Implemented the reviewed DLB plan in `train.py` only. The CIFAR-10 dataset now returns raw identities, and a private seed-42 sampler partitions 49,920 identities into 390 half-batches and emits 389 adjacent overlapping batches. Clean-clean transitions add the published temperature-three, weight-one detached KL; mixed batches preserve the parent CutMix loss and invalidate the teacher. The period-two clean-tail SAM path uses the same incoming teacher in both passes and publishes only primary-forward outgoing logits after the sole optimizer update. Startup and final audit lines expose the recipe, cache state transitions, activity, and mismatches.

### Surprises & Discoveries

- The inherited EXP-004 `train.py` passes `ruff check` but is not `ruff format --check` clean. The reviewed plan therefore omits automatic formatting to avoid unrelated parent-code churn.
- PyTorch worker base seeding requires the dedicated worker generator to be passed through `DataLoader(generator=...)`; the batch sampler's private generator controls only index permutations.
- A two-batch worker smoke exited while pinned-memory prefetch remained in flight and printed a shutdown-thread warning. Repeating the smoke through the complete 389-batch epoch shut workers down cleanly and verified every adjacent overlap; this was a smoke-lifecycle artifact, not a training-code defect.

### Decisions

- Cache indices remain on CPU, so identity assertions do not add device transfers. Cached logits remain detached FP32 tensors on GPU and are only about 5 KiB.
- `dlb_cache_invalidations` counts mixed batches that actually discard a live clean teacher, rather than all mixed batches. Publications count every successful clean batch, including bootstrap batches that do not yet consume a teacher.
- The result will be attributed to the combined repeat-view sampler plus clean-gated KL mechanism. DLB-specific isolation would require a future fixed-seed `alpha=0` control.

## Experimental Adjustments

- None.

## Run Log

### Run 1

Metadata:
- **Job ID**: local PID 329724 (exec session 25171)
- **Log file(s)**: `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/tree-v0-gpt-5-6-sol/run.log`
- **WandB**: N/A
- **Status**: completed
- **Started**: 2026-08-05 13:08:02 UTC
- **Ended**: 2026-08-05 13:14:24 UTC

Description:
- One fixed-seed local run of the clean-gated DLB intervention on physical GPU 0. The run preserves the EXP-004 WRN, front-loaded CutMix, period-two clean-tail SAM, and 300-second charged budget while adding the reviewed overlapping sampler and temporal consistency loss. Expected success is `best_test_acc >= 95.50%` with zero overlap mismatches and at least 24,500 steps as a throughput diagnostic.

Observations:
- Preflight passed: `py_compile` and `ruff check` exited 0; the semantic smoke produced 389 batches from 390 chunks and 49,920 unique identities with correct tau-three detached KL; the complete DataLoader smoke found exact 128-identity overlap and 128/128 independently transformed repeated views.
- GPU-0 smoke passed on the 2,748,890-parameter full WRN: one ordinary forward, two SAM forwards, exact 0.050000 perturbation, one BatchNorm update on the SAM step, exact restoration, CUDA RNG parity, and primary-only cache publication.
- Run 1 started successfully and wrote the expected GPU, model, complete DLB/CutMix/SAM config, 300-second budget, and `Batches per epoch: 389` headers to `run.log` before training progress.
- The run completed exit 0 with 66 evaluations for 66 natural epochs, zero overlap mismatches, and a complete summary. It retained 25,492 optimizer steps and finished below the 600-second outer limit. (source: `run.log` L138-L151)
- DLB was active on 10,037 batches / 1,284,736 repeated examples; 15,247 clean batches published teachers, 5,173 live caches were invalidated by mixing, and 66 epoch resets occurred. (source: `run.log` L140)

Key Metrics:
- `best_test_acc`: 95.28% versus parent 95.40% and required 95.50%; research failure (source: `run.log` L142)
- `final_test_acc`: 95.22%; best-to-final gap 0.06 points (source: `run.log` L143)
- `final_test_loss`: 0.1455 versus parent 0.1654, an improvement of 0.0199 despite lower accuracy (source: `run.log` L144)
- `training_seconds`: 300.0; `total_seconds`: 382.4; `startup_seconds`: 1.1 (source: `run.log` L145-L147)
- `peak_vram_mb`: 1,190.5; unchanged from the parent to one decimal (source: `run.log` L148)
- `num_epochs`: 66; `num_steps`: 25,492; `num_params`: 2,748,890 (source: `run.log` L149-L151)
- CutMix: 10,245/20,645 = 0.4962; SAM: 2,424/4,847 = 0.5001, first step 20,646 at progress 0.7500 (source: `run.log` L138-L139)

## Verification Results

### Conditions Checked

- **Parent and scope — passed**: tree query returned EXP-004 metric 95.40 at commit `1a8d0de`; branch was EXP-005 at that commit and `git diff --name-only` returned only `train.py`.
- **Static implementation — passed**: `py_compile`, `ruff check`, and `git diff --check` all exited 0; protected files and evaluator were unchanged.
- **Deterministic semantics — passed**: 389 batches, 390 chunks, 49,920 unique identities, exact adjacent overlap, private RNG reproducibility, 128/128 fresh transformed views, temperature-three KL agreement, detached teacher, and repeated-half-only gradients.
- **GPU/SAM integration — passed**: physical GPU 0 was an NVIDIA H20 with 97,871 MiB; the full-WRN smoke measured 0.050000 perturbation, one BatchNorm update, exact restore, CUDA RNG parity, and primary-only cache publication.
- **Full-run protocol integrity — passed**: exit 0; 300.0 charged and 382.4 total seconds; 389 batches per natural epoch; 66 evals for 66 epochs; CutMix 0.4962; SAM 0.5001 beginning at even step 20,646/progress 0.7500; DLB nonzero with zero mismatches; complete summary and unchanged parameter count. (source: `run.log` L138-L151)
- **Primary metric — failed**: 95.28% is below the necessary 95.50% threshold by 0.22 points and below the 95.40% parent by 0.12 points. Verification stopped at this necessary-condition failure; no retry or parameter adjustment is allowed. (source: `run.log` L142)
- **Cleanup — passed**: all log evidence was recorded in `03-execute.md` and `04-analysis.md`, then `run.log` was removed before the next experiment.

### Informational Metrics

- Formal optional collection skipped after the primary necessary condition failed. All available values are preserved inline under Run 1 Key Metrics for failure analysis.

## Errors & Dead Ends

### 2026-08-05 — Valid DLB run missed the accuracy gate
- Error: `best_test_acc=95.28% < required 95.50%`
- Root cause: Research outcome, not an execution defect. The combined repeat-view sampler plus clean-gated KL produced lower accuracy than EXP-004 despite lower final loss and nearly identical optimizer exposure; mechanism analysis is deferred to `04-analysis.md`.
- Source: `run.log` L138-L151
- Do NOT retry: Do not reroll the seed, tune `tau`/`alpha`, redefine epochs, or rerun the same configuration based on this result.

## Human Notes

> The user requires physical GPU 0 and Claude as the sole adversarial reviewer. Both EXP-005 adversarial reviews used Claude; no fallback reviewer was used.
