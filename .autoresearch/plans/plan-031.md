# Plan EXP-031: Progressive resizing 24→32 (in-step GPU downsample, switch at 50% of budget)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-031.md

## Milestones

### Milestone 1: Dual-resolution training implemented and sanity-checked
- [x] On branch `autoresearch/exp-031` (cut from `autoresearch/dev` @ 1990397), edit `train.py`:
  1. New constants after LABEL_SMOOTHING: `LOW_RES = 24  # phase-1 input resolution`, `RES_SWITCH_FRAC = 0.5  # fraction of budget at LOW_RES`.
  2. `model = torch.compile(model)` → `model = torch.compile(model, dynamic=False)` — forces static per-shape specialization so the phase-2 (32×32) graph is compiled exactly like the baseline's, protecting the NUMERICS EQUIVALENCE law for the entire anneal half; both shape graphs are built in startup warmup.
  3. Warmup loop: after the existing 3 iterations at 32×32, add `warm_x24 = F.interpolate(warm_x, size=LOW_RES, mode="bilinear")` and 3 more forward/backward iterations on it (same pattern), then the existing `zero_grad`/synchronize. Both inductor graphs + cudnn.benchmark tunings land in startup.
  4. Timed step body — insert immediately after the `for g in optimizer.param_groups: g["lr"] = lr_now` block (so `progress` is already computed; charged, honest):
     ```python
     if progress < RES_SWITCH_FRAC:
         inputs = F.interpolate(inputs, size=LOW_RES, mode="bilinear")
     ```
     NOTHING else in the step changes. Transforms, loader, schedule, optimizer, eval all byte-identical to baseline.
- [x] Sanity: AST parse OK; `git diff` shows exactly the 4 edit sites; confirm the interpolate sits INSIDE the timed region (after `t0`, before autocast) and phase 2 executes the unmodified baseline path (the `if` is false for progress ≥ 0.5).

### Milestone 2: Run 1 launched with phase-aware gates
- [ ] GPU-0 zero-compute-apps pre-check passes (wait-for-free poll if busy, per EXP-030 pattern).
- [ ] Composite watchdog (15s ticks) with PHASE-AWARE thresholds keyed on the step line's pct_done:
  - **Phase 1 (pct < 50)**: early-dt gate — 3 consecutive windows >19.0ms within the first 7 ticks → EARLY_DT_KILL (at ≥19ms the epoch gain is <+12 ≈ +0.2, arithmetically sub-bar); contention — 4 consecutive >26ms → CONTENTION_KILL.
  - **Phase 2 (pct ≥ 50)**: contention — 4 consecutive >30ms → CONTENTION_KILL; dt must return to ≈22.4 (a phase-2 mean >24 in the post-hoc profile = compile/numerics regression, code-error class).
  - **Switch-stall guard**: any single window >100ms once pct >45 → SWITCH_STALL_KILL (a charged mid-run compile; should be impossible with the dual-shape warmup).
  - STARTUP_KILL tick 10; NaN/inf guard.
- [ ] Early readout: phase-1 dt and projected epochs recorded in exp-log within the first 2 minutes.

### Milestone 3: Trajectory readout and completion
- [ ] Post-switch check (~52–60% progress): evals should climb toward the baseline family within ~10 epochs of the switch (full-res adaptation). Failure to rejoin = toll dominating; let it complete — the plateau decides.
- [ ] Completion: rc=0, total ≤600s (est. ~460–490s: 300 training + ~145 eval at ~170 epochs + startup ~16), eval_lines = num_epochs (expect 160–185), params 4,286,026 (no model change), per-segment post-hoc profile clean.

## Code Changes
- **train.py** (only file): 2 constants, `dynamic=False` on compile, dual-shape warmup, 2-line in-step downsample branch. Model, optimizer, schedule, data pipeline, eval untouched.
- Why this tests the hypothesis: the only change is input resolution for the first half of the TIMED budget — cheaper steps → more steps — with the downsample cost honestly charged inside the step. Phase 2 is numerically the baseline regime.
- Risks/edge cases: (a) `F.interpolate` on a channels_last fp32 tensor — supported; output inherits the memory format (and compile re-normalizes layout regardless); (b) `dynamic=False` may recompile per shape — exactly what we want, both shapes warmed; (c) BN batch stats adapt instantly at the switch, running stats within ~7 batches (momentum 0.1) — the next eval (≤1 epoch later) sees adapted stats; (d) eval is always 32×32 — phase-1 evals read low (train/test res mismatch), cosmetic under the max-statistic; (e) pad-shortcut slicing handles 24→12→6 spatial flow (even sizes).

## Configuration Changes
- LOW_RES: 24 (FLOPs ×0.5625; fastai/MosaicML recipes use ~0.7–0.75 linear scale → 24/32 = 0.75)
- RES_SWITCH_FRAC: 0.5 (MosaicML default ramps to full res by ~mid-training; single fixed switch keeps this one-variable-ish)
- All baseline constants unchanged. Both new constants are external-recipe anchors, NOT tuned — a miss leaves interior points (28px, 0.3/0.65 splits) unexplored; flagged for honest analysis.

## Execution Environment
- Method: local composite background Bash (GPU-0 wait-for-free pre-check + launch + phase-aware inline watchdog + wait + summary), branch `autoresearch/exp-031`, GPU 0 (`CUDA_VISIBLE_DEVICES=0`).
- Resources: VRAM ~1615MB (phase-1 activations smaller); 8 loader workers.
- Estimated runtime: ~460–490s total. Under the 600s cap (eval count rises to ~170 but stays bounded).
- Log output: `run.log` via `uv run train.py > run.log 2>&1`; watchdog WIN lines tagged with pct; post-hoc per-segment awk profile authoritative.
- Tool skill: none (local).

## Abort Criteria
- **Startup gate**: no step lines by tick 10 (150s) → kill (startup grows ~+3s for the second warmup shape).
- **Early-dt gate (phase 1)**: 3 consecutive windows >19.0ms within first 7 ticks → kill; classify code error (downsample not engaging / unexpected kernel cost), not contention.
- **Contention**: 4 consecutive windows >26ms (pct<50) or >30ms (pct≥50) → kill, contaminated, rerun once (eval-side-immunity judgment clause available per EXP-029/030 precedent).
- **Switch-stall**: any single window >100ms at pct>45 → kill, code error (charged compile leak).
- **Divergence**: NaN/inf loss, or any eval <15% after epoch 5 → kill, code error.
- **Wall cap**: >600s → kill, failure.
- **Crash** (rc≠0): code-error fix + resubmit per execute-skill rules (max 2).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** @ 1990397; bar = 96.81. σ context: mean ≈96.57, σ ≈0.16.

1. **best_test_acc ≥ 96.81**:
   - Command: `grep "^best_test_acc:" run.log`. Empty ⇒ crash classification.
   - Pre-condition (per-segment profile): `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); if($2<50){n1++;s1+=ms;if(ms>22)c1++}else{n2++;s2+=ms;if(ms>27)c2++}} {p1=$1;p2=$2} END{printf "P1: %d win, mean %.1f ms, slow>22: %d | P2: %d win, mean %.1f ms, slow>27: %d\n", n1, s1/n1, c1, n2, s2/n2, c2}'` — require ≤2 slow windows in EACH segment AND phase-2 mean ≤24ms (numerics-equivalence guard) AND num_epochs ≥ 150 (throughput delivery; below 150 the mechanism failed). Contaminated ⇒ rerun once.
   - Integrity sub-check on a bar-pass: plateau-LEVEL shift (final-7 median ≥ 96.6); `grep "^num_params:" run.log` = 4,286,026; training_seconds = 300.0; the gain must come with epochs ≥155 (the claimed mechanism), not from an anomalous single eval.
2. **Completes within budget**: rc=0 AND `grep "^total_seconds:" run.log` ≤ 600.
3. **Validation ≤ once/epoch**: `tr '\r' '\n' < run.log | grep -c "eval ep"` ≤ num_epochs.

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- Phase-1 dt mean (the resolution cost curve datum: 24px on H20) and num_epochs (expect 160–185) — recorded regardless of verdict
- peak_vram_mb (expect ≤1615), startup_seconds (expect ~16), total_seconds
- Post-switch rejoin: epochs from switch until evals re-enter the baseline family ±1pp band at matched epoch — the adaptation-cost datum
- Phase-1 eval trail vs family (expected LOW from res mismatch — cosmetic; do not interpret as toll)
