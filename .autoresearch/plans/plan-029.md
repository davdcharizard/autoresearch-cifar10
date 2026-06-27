# Plan EXP-029: Clean-data BN running-stat recalibration before every eval
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-029.md

## Milestones

### Milestone 1: Recalibration implemented and sanity-checked
- [ ] On branch `autoresearch/exp-029` (cut from `autoresearch/dev` @ 1990397), edit `train.py`:
  1. New constants after LABEL_SMOOTHING: `BN_RECAL_BATCHES = 16` (16×512 = 8,192 clean train images), `BN_RECAL_BATCH_SIZE = 512`.
  2. **Startup (uncharged, before the timed loop, after the train_loader is built)**: build the clean recalibration tensor once —
     `clean_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])`;
     `clean_set = datasets.CIFAR10(DATASET_DIR, train=True, download=True, transform=clean_tf)`;
     stack the first `BN_RECAL_BATCHES × 512` images (fixed slice `[0:8192]`, deterministic — no sampling, no seed interaction) via a temporary `DataLoader(clean_set, batch_size=512, shuffle=False, num_workers=4)` iterated `BN_RECAL_BATCHES` times; store as a list of GPU tensors in channels_last (`x.to(device).to(memory_format=torch.channels_last)`). ~101MB VRAM. Delete the temporary loader.
  3. New module-level helper:
     ```python
     def recalibrate_bn(net, clean_batches):
         momenta = {}
         for m in net.modules():
             if isinstance(m, nn.BatchNorm2d):
                 m.reset_running_stats()
                 momenta[m] = m.momentum
                 m.momentum = None  # exact cumulative average over the clean batches
         was_training = net.training
         net.train()
         with torch.no_grad():
             for xb in clean_batches:
                 with torch.autocast("cuda", dtype=torch.bfloat16):
                     net(xb)
         for m, mom in momenta.items():
             m.momentum = mom  # restore 0.1 BEFORE the next compiled training step (guard safety)
         if not was_training:
             net.eval()
     ```
  4. At the eval call site (currently `test_loss, test_acc = evaluator.evaluate(base_model, device)`): insert `recalibrate_bn(base_model, clean_batches)` on the line immediately before. **Forward passes go through `base_model` (eager)** — never the compiled wrapper — so no recompile/guard churn.
  5. NOTHING inside the timed step body changes — the diff must not touch lines between `t0 = time.time()` and `total_training_time += dt`.
- [ ] Sanity: `uv run python -c "import ast; ast.parse(open('train.py').read())"`; `grep -n "recalibrate_bn\|BN_RECAL" train.py` shows the helper, the constants, the startup build, and exactly ONE call site (before the single eval call); confirm no edits inside the timed region (`git diff` review).

### Milestone 2: Run 1 launched with gates
- [ ] GPU-0 zero-compute-apps pre-check passes inside the composite launcher.
- [ ] Composite: pre-check → `rm -f run.log` → launch → 15s watchdog: contention kill (4 consecutive >30ms), early-dt gate (3 consecutive >27.0ms within first 7 ticks — dt must be IDENTICAL to baseline 22.4ms; any regression means the change leaked into the timed region or compile broke → treat as code error, not contention), STARTUP_KILL tick 10 (startup grows by the clean-tensor build, est. +3–6s → still ≪150s), NaN/inf guard → `wait` → summary.
- [ ] **Recalibration-sanity gate (new, this experiment)**: first eval must appear by tick ~4 and read ≥ 25%. A collapsed ep1 eval (<25%, vs family ~38) with healthy train loss means the recalib helper corrupted the stats (e.g., ran on wrong tensor/mode) → kill, classify as code error, fix and resubmit (counts toward the 2-retry limit).

### Milestone 3: Trajectory readout and completion
- [ ] Early readout: ep1–10 evals vs baseline family (ep1 ~38, ep5 ~64, ep10 ~78). The hypothesis predicts a visible upward shift from ep1. A trajectory tracking the family exactly (±0.15) predicts a noise-band outcome; let it complete either way — the plateau is the measurement.
- [ ] Completion: rc=0, total ≤600s (est. ~510–540s: baseline 493 + 139 recalibs × ~0.15–0.3s), eval_lines = num_epochs ≈ 139, params 4,286,026 (recalib adds NO params — buffers only), post-hoc profile clean.

## Code Changes
- **train.py** (only file): constants + startup clean-tensor build + `recalibrate_bn` helper + one-line insertion before the eval call. The timed step body, model, optimizer, schedule, loaders, and compile path are untouched.
- Why this tests the hypothesis: the ONLY behavioral change is which BN running stats the evaluator consumes — clean-distribution estimates instead of augmented-distribution estimates. Training cannot see the change (train mode uses batch stats; weights never touched by the no-grad recalib forwards). Any metric movement is attributable to eval-time stat alignment.
- Risks/edge cases: (a) momentum restore must happen before the next compiled step — handled inside the helper; (b) `reset_running_stats()` also zeroes `num_batches_tracked` — only consumed when momentum is None, i.e., only during recalib itself; (c) recalib runs under bf16 autocast for speed — BN stat accumulation is fp32 internally (autocast keeps BN fp32), deviation vs the evaluator's numerics is O(1e-3) relative, negligible against the distribution shift being corrected; (d) the evaluator manages train/eval modes itself (current baseline never calls .eval()) — helper restores the entry mode defensively; (e) VRAM +~101MB clean tensor + recalib activations — soft constraint, expect peak ~1750MB.

## Configuration Changes
- BN_RECAL_BATCHES: 16 (8,192 images — per-channel stats over 8192×32×32 ≈ 8.4M samples/channel, far past convergence; SWA re-estimation uses the full set but stats saturate orders of magnitude earlier)
- BN_RECAL_BATCH_SIZE: 512 (matches training batch; channels_last)
- All training constants byte-identical to baseline.

## Execution Environment
- Method: local composite background Bash (pre-check + launch + inline watchdog + wait + summary), branch `autoresearch/exp-029`, GPU 0 (`CUDA_VISIBLE_DEVICES=0`).
- Resources: VRAM ~1750MB; 8 loader workers (+4 temporarily at startup for the clean tensor build).
- Estimated runtime: ~510–540s total (startup ~16–19s, training 300s charged, evals+recalibs ~210–230s wall). Under the 600s cap with margin.
- Log output: `run.log` via `uv run train.py > run.log 2>&1`; watchdog WIN lines; post-hoc awk profile authoritative.
- Tool skill: none (local).

## Abort Criteria
- **Startup gate**: no step lines by tick 10 (150s) → kill.
- **Early-dt gate**: 3 consecutive windows >27.0ms within first 7 ticks → kill (dt must equal baseline; regression = code leak into timed region, classify as code error).
- **Contention kill**: 4 consecutive windows >30ms → kill, contaminated, rerun once.
- **Recalib-sanity**: ep1–3 evals < 25% with sane train loss → kill, code error (stats corruption), fix + resubmit (≤2 retries).
- **Divergence**: NaN/inf loss → kill (would be a genuine surprise — training is untouched — so treat as code error and inspect).
- **Wall cap**: >600s → kill, failure.

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** @ 1990397; bar = 96.81. σ context (EXP-027): baseline mean ≈96.57, σ ≈0.16.

1. **best_test_acc ≥ 96.81**:
   - Command: `grep "^best_test_acc:" run.log`. Empty ⇒ crash classification.
   - Pre-condition: post-hoc profile — `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if(ms>27) c++; n++; s+=ms} {p1=$1; p2=$2} END{printf "windows>27ms: %d of %d | mean win %.1f ms | expected epochs %.1f\n", c, n, s/n, 139*22.4/(s/n)}'` — require ≤2 slow windows AND epochs within ±3 of expected. Contaminated ⇒ rerun once.
   - Integrity sub-check on a bar-pass: gain must be a trajectory-wide/plateau-LEVEL shift (final-7 median ≥ 96.6 via `tr '\r' '\n' < run.log | grep "eval ep" | tail -7`), not a single-eval spike; `grep "^num_params:" run.log` = 4,286,026; num_epochs within ±3 of 139 (training untouched); training_seconds = 300.0.
2. **Completes within budget**: rc=0 AND `grep "^total_seconds:" run.log` ≤ 600.
3. **Validation ≤ once/epoch**: `tr '\r' '\n' < run.log | grep -c "eval ep"` ≤ num_epochs. (Recalibration touches no test data — it is not validation.)

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- peak_vram_mb (expect ~1750; soft constraint)
- num_epochs (expect ≈139 — unchanged), startup_seconds (expect +3–6s for the clean tensor build), total_seconds (recalib wall cost = total − baseline 493)
- Trajectory shift at ep1/5/10 vs family (~38/~64/~78) — the per-epoch magnitude of the alignment effect, the key mechanistic readout for analysis regardless of verdict
