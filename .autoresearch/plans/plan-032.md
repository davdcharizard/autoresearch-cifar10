# Plan EXP-032: SWA tail — freeze cosine at 85%, equal-average iterates, eval BN-re-estimated SWA model
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-032.md

## Milestones

### Milestone 1: SWA tail implemented and sanity-checked
- [x] On branch `autoresearch/exp-032` (cut from `autoresearch/dev` @ 1990397), edit `train.py`:
  1. New import after the optim import: `from torch.optim.swa_utils import AveragedModel, update_bn` (verified present in torch 2.9.1).
  2. New constant after LABEL_SMOOTHING: `SWA_START_FRAC = 0.85  # freeze the cosine here; equal-average end-of-epoch iterates after this point`.
  3. After optimizer creation (before warmup, uncharged startup): `swa_model = AveragedModel(base_model)` — eager deepcopy, weights shared-by-copy, +~17MB VRAM.
  4. Timed step — ONE added branch right after `lr_now = lr_at(progress)` (charged, trivial):
     ```python
     if progress >= SWA_START_FRAC:
         lr_now = lr_at(SWA_START_FRAC)  # constant SWA tail LR ≈ 0.030 (cosine frozen where it stands)
     ```
  5. Per-epoch eval block — replace the unconditional `test_loss, test_acc = evaluator.evaluate(base_model, device)` with:
     ```python
     if total_training_time / TIME_BUDGET_S >= SWA_START_FRAC:
         swa_model.update_parameters(base_model)
         update_bn(train_loader, swa_model, device=device)
         test_loss, test_acc = evaluator.evaluate(swa_model, device)
     else:
         test_loss, test_acc = evaluator.evaluate(base_model, device)
     ```
     Eval print line, best_acc tracking, `if epoch == 1: gc.collect()` all UNCHANGED (eval-line format must stay grep-identical). Still exactly ONE eval per epoch.
- [x] Sanity: AST parse OK; `git diff` shows exactly the 5 edit sites; confirm (a) the ONLY in-step change is the lr-freeze branch, (b) all SWA ops (update_parameters, update_bn, eval) sit OUTSIDE the timed region, (c) pre-85% training math byte-identical to baseline.

### Milestone 2: Run 1 launched with gates
- [ ] GPU-0 zero-compute-apps pre-check (wait-for-free poll up to 60 min if busy).
- [ ] Composite watchdog (15s ticks, 44 ticks): contention 4 consecutive windows >27ms → CONTENTION_KILL; STARTUP_KILL tick 10; NaN/inf guard; wall cap 600s; **SWA-bug gate** — once pct >87, if the latest eval line shows test_acc < 92.0 → SWA_BUG_KILL (a working BN re-estimation cannot sit >3pp below the converged family; a broken one reproduces EXP-029's −10.9 signature).
- [ ] Early readout: dt ≈ 22.4ms confirmed within first 2 minutes (this run changes nothing before 85%).

### Milestone 3: SWA-phase readout and completion
- [ ] SWA evals begin ~ep 118–120 (85% of charged budget). Expect the first SWA eval ≈ raw family level (average of 1 snapshot, re-estimated BN), then a climb ABOVE the family as snapshots accumulate. Failure to exceed the family within ~8 SWA epochs = averaging gain < forfeited anneal; let it complete — the plateau decides.
- [ ] Completion: rc=0, total ≤600s (est. ~535s: 493 baseline + ~21 update_bn passes ×~2s), eval_lines = num_epochs (~139), params 4,286,026.

## Code Changes
- **train.py** (only file): 1 import, 1 constant, 1 startup AveragedModel, 1-line lr freeze in-step, eval-block branch. Model, optimizer, schedule before 85%, data pipeline, eval semantics untouched.
- Why this tests the hypothesis: the only training-math change is the tail LR (cosine→0 replaced by constant ≈0.030 from 85%), which is the canonical SWA requirement for basin sampling; the evaluated artifact in the tail is the equal-weight iterate average with BN running stats re-estimated on the AUGMENTED train loader — repairing the exact flaw diagnosed in EXP-011 (averaged weights + live BN buffers) using the EXP-029 law (stats must match training-time constants).
- Risks/edge cases: (a) `update_bn` iterates the FULL train_loader (97 batches, forward-only, train mode, momentum=None cumulative stats) and handles (input, target) tuples natively — ~2s/call, ~21 calls ≈ +42s wall; (b) consuming extra loader passes advances worker augmentation RNG → subsequent epochs see different draws than baseline (reshuffle-class noise, within σ); (c) AveragedModel is an nn.Module wrapper — evaluator calls .eval()/forward on it directly; first SWA eval = single snapshot ≈ raw model (BN re-est only); (d) frozen tail LR makes RAW iterates noisier — by design; the SWA average must beat what the forfeited anneal would have produced; (e) compiled `model` still trains; `base_model` (eager alias, shared weights) feeds update_parameters.

## Configuration Changes
- SWA_START_FRAC: 0.85 — anchor: Izmailov et al. start SWA at ~75–80% of training; our family converges ~80% of budget; 0.85 leaves ~20 snapshot epochs. lr_at(0.85) ≈ 0.030 ≈ canonical CIFAR swa_lr 0.05 scaled to schedule. NOT tuned — a miss leaves interior (0.75/0.9, other tail LRs) unbracketed; flag for honest analysis.
- All baseline constants unchanged. No new packages (torch.optim.swa_utils is core).

## Execution Environment
- Method: local composite background Bash (GPU-0 wait-for-free pre-check + launch + inline watchdog + wait + summary), branch `autoresearch/exp-032`, GPU 0 (`CUDA_VISIBLE_DEVICES=0`), `uv run train.py > run.log 2>&1`.
- Resources: VRAM ~1630MB (baseline 1613 + SWA copy); 8 loader workers (baseline; dt 22.4ms is loader-fed at this pace — 16-worker lever held in reserve if Run 1 shows wall pressure).
- Estimated runtime: ~535s total (300 charged + startup ~14 + ~139 evals + ~21 update_bn passes). Under the 600s cap with ~65s margin.
- Log output: `run.log`; watchdog WIN lines; post-hoc awk profile authoritative.
- Tool skill: none (local).

## Abort Criteria
- **Startup gate**: no step lines by tick 10 (150s) → kill.
- **Contention**: 4 consecutive windows >27ms → kill, contaminated, rerun once (eval-side-immunity judgment clause per EXP-029/030 precedent).
- **SWA-bug gate**: pct >87 AND latest eval <92.0 → kill, code-error class (BN re-estimation failure; EXP-029 signature).
- **Divergence**: NaN/inf loss, or any eval <15% after epoch 5 → kill, code error.
- **Wall cap**: >600s → kill, failure (if caused by update_bn cost: Run 2 with update_bn thinned to every 2nd SWA epoch + 16 workers — wall-side-only fix, EXP-031 precedent).
- **Crash** (rc≠0): code-error fix + resubmit per execute-skill rules (max 2).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** @ 1990397; bar = 96.81. σ context: mean ≈96.57, σ ≈0.16.

1. **best_test_acc ≥ 96.81**:
   - Command: `grep "^best_test_acc:" run.log`. Empty ⇒ crash classification.
   - Pre-condition (profile): `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); n++; s+=ms; if(ms>27)c++} {p1=$1;p2=$2} END{printf "%d win, mean %.1f ms, slow>27: %d\n", n, s/n, c}'` — require ≤2 slow windows AND mean ≤24ms AND num_epochs within ±4 of 139. Contaminated ⇒ rerun once.
   - Integrity sub-check on a bar-pass: best must come from a TAIL (SWA) eval with the SWA trail sitting above the pre-SWA family (plateau-LEVEL shift, not an anomalous single eval); `grep "^num_params:" run.log` = 4,286,026; training_seconds = 300.0; eval_lines = num_epochs.
2. **Completes within budget**: rc=0 AND `grep "^total_seconds:" run.log` ≤ 600.
3. **Validation ≤ once/epoch**: `tr '\r' '\n' < run.log | grep -c "eval ep"` ≤ num_epochs.

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- SWA trail: every eval from ep ~118 on, vs pre-SWA family level — the averaging-gain datum (recorded regardless of verdict)
- First-SWA-eval delta vs last raw eval (isolates the BN re-estimation effect at n=1 snapshots)
- total_seconds (update_bn wall cost datum: ~21 passes), peak_vram_mb (expect ~1630), startup_seconds (~14)
- num_epochs (expect ~139 ± 2 — dt unchanged)
