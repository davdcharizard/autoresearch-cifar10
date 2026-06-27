# Plan EXP-033: Augmentation taper — revert the final 12% of budget to the original-ResNet transform (crop+flip only)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-033.md

## Milestones

### Milestone 1: Dual-loader taper implemented and sanity-checked
- [x] On branch `autoresearch/exp-033` (cut from `autoresearch/dev` @ 1990397), edit `train.py`:
  1. New constant after LABEL_SMOOTHING: `AUG_TAPER_FRAC = 0.88  # revert to original-ResNet light aug (crop+flip) after this`.
  2. After the `train_tf` Compose: `light_tf = transforms.Compose([RandomCrop(32, padding=4), RandomHorizontalFlip(), ToTensor(), Normalize(mean, std)])` — the He-2015 original CIFAR recipe = baseline transform MINUS TrivialAugmentWide and RandomErasing.
  3. After `train_loader`: `tail_set = datasets.CIFAR10(DATASET_DIR, train=True, download=True, transform=light_tf)` and `tail_loader = DataLoader(tail_set, ...)` with IDENTICAL loader args (batch 512, shuffle, 8 workers, pin_memory, drop_last, persistent_workers). Workers spin up lazily at first tail epoch (~2s, uncharged).
  4. Epoch loop head — replace `for inputs, targets in train_loader:` with:
     ```python
     epoch_loader = (
         train_loader
         if total_training_time / TIME_BUDGET_S < AUG_TAPER_FRAC
         else tail_loader
     )
     for inputs, targets in epoch_loader:
     ```
     NOTHING else changes: LR cosine anneals to 0 exactly as baseline (full anneal preserved), timed step byte-identical, eval = `base_model` once per epoch, model/optimizer untouched.
- [x] Sanity: AST parse OK; `git diff` shows exactly the 4 edit sites (30 insertions / 1 deletion); timed step body untouched; both loaders share batch 512 / drop_last (97 steps/epoch).

### Milestone 2: Run 1 launched with gates
- [ ] Launch gates (EXP-032 infra lesson): zero GPU-0 compute apps AND host 1-min load < 60, poll 30s up to 2h.
- [ ] Composite watchdog (15s ticks, 44): contention 4 consecutive windows >27ms → CONTENTION_KILL; STARTUP_KILL tick 10; NaN/inf guard; wall cap 600s; divergence (any eval <15% after ep5).
- [ ] Early readout: dt ≈ 22.4ms confirmed in first 2 minutes (taper changes nothing before 88%).

### Milestone 3: Taper readout and completion
- [ ] Taper begins ~ep 122–124 (88% of charged budget). Expect: eval jump ≥ +0.15 within 2 taper epochs (EXP-025's fully-clean analogue was +0.35); then a SUSTAINED climb/plateau riding the final anneal. Watch test_loss in the last ~6 epochs — a clear upward reversal = light pressure insufficient (EXP-025's overfit signature returning; let it complete, the plateau decides).
- [ ] Completion: rc=0, total ≤600s (est. ~495s — baseline 493 + ~2s tail-worker spin-up), eval_lines = num_epochs (~139), params 4,286,026.

## Code Changes
- **train.py** (only file): 1 constant, 1 transform Compose, 1 dataset + 1 loader, epoch-loader selection. Timed step, schedule, model, optimizer, eval all byte-identical to baseline.
- Why this tests the hypothesis: the ONLY change is the training distribution of the final ~17 epochs — from heavy (TA+RE) to the original-paper crop+flip — while the anneal completes and training pressure persists. It interpolates the measured endpoints: baseline (full pressure, plateau ~96.6) and EXP-025 (zero pressure: +0.35 jump then overfit, net −0.87). Decision-boundary class per EXP-032's diagnosis.
- Risks/edge cases: (a) light-aug gradient noise is lower late — data-side, distinct from the bracketed optimizer-side noise axis; flagged for analysis; (b) second CIFAR10 dataset object re-reads the same cached files (+~1s startup, no re-download — data/ exists); (c) +8 lazy workers (16 total) — fine at 180 cores with load-gated launch; (d) BN stats track the light distribution within ~7 batches of the switch — training continues, EXP-029-safe; (e) if the +0.35 was specific to FULLY clean data (no crop/flip, test-transform), the jump may be ≈0 — falsifies cleanly.

## Configuration Changes
- AUG_TAPER_FRAC: 0.88 (~17 epochs of tail) — anchors: EXP-025 switched at 0.85 and the alignment jump completed within ~2–3 epochs while overfit needed ~5 to appear; 0.88 keeps the tail long enough to harvest a sustained plateau but shorter than the measured overfit runway. NOT tuned; a miss leaves 0.80/0.93 interiors — flagged.
- light_tf: crop(32,4)+flip+normalize — external anchor: the original ResNet paper's exact CIFAR augmentation (He et al. 2015 § 4.2). Keeps LS 0.1 and all loss/optimizer settings.
- All baseline constants unchanged.

## Execution Environment
- Method: local composite background Bash (dual launch gates + launch + watchdog + wait + summary), branch `autoresearch/exp-033`, GPU 0, `uv run train.py > run.log 2>&1`.
- Resources: VRAM ~1613MB (unchanged); up to 16 loader workers (8 active at a time + 8 lazy tail workers).
- Estimated runtime: ~495s total. ~105s margin under the cap.
- Log output: `run.log`; post-hoc awk profile authoritative.
- Tool skill: none (local).

## Abort Criteria
- **Startup gate**: no step lines by tick 10 (150s) → kill.
- **Contention**: 4 consecutive windows >27ms → kill, contaminated, rerun once after gates re-clear (EXP-032 precedent: gates = GPU-free AND load<60).
- **Divergence**: NaN/inf loss, or any eval <15% after epoch 5 → kill, code error.
- **Wall cap**: >600s → kill, failure.
- **Crash** (rc≠0): code-error fix + resubmit per execute-skill rules (max 2).
- NOTE: a post-taper eval DROP is NOT an abort (it is the falsification signal — let the run complete; the final plateau is the result).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** @ 1990397; bar = 96.81. σ context: mean ≈96.57, σ ≈0.16.

1. **best_test_acc ≥ 96.81**:
   - Command: `grep "^best_test_acc:" run.log`. Empty ⇒ crash classification.
   - Pre-condition (profile): `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); n++; s+=ms; if(ms>27)c++} {p1=$1;p2=$2} END{printf "%d win, mean %.1f ms, slow>27: %d\n", n, s/n, c}'` — require ≤2 slow windows AND mean ≤24ms AND num_epochs within ±4 of 139. Contaminated ⇒ rerun once (gates re-cleared first).
   - Integrity sub-check on a bar-pass: best must come from a POST-TAPER eval (ep ≥ ~122) with the post-taper trail sitting above the pre-taper family level (sustained shift, not a single-eval anomaly); `grep "^num_params:" run.log` = 4,286,026; training_seconds = 300.0; eval_lines = num_epochs.
2. **Completes within budget**: rc=0 AND `grep "^total_seconds:" run.log` ≤ 600.
3. **Validation ≤ once/epoch**: `tr '\r' '\n' < run.log | grep -c "eval ep"` ≤ num_epochs.

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- **Taper jump**: evals at the first 3 post-taper epochs vs the mean of the last 3 pre-taper evals (the alignment-gain datum at LIGHT pressure; EXP-025's fully-clean datum was +0.35) — recorded regardless of verdict
- **Overfit check**: test_loss trend over the final 6 epochs (rising = pressure insufficient; the bracketing datum)
- total_seconds (~495 expected), startup_seconds (~14, +1s for second dataset), peak_vram_mb (~1613), num_epochs (~139)
