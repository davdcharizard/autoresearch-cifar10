# Plan EXP-051: Confidence-weighted CE — detached w = p_true^0.7, batch-mean-normalized (GCE-style aug-noise filtering)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-051.md

Baseline at planning time (exp-index.sh baseline): **96.71** @ 1990397 → bar = **96.81**. σ context (EXP-027): mean ≈ 96.57, σ ≈ 0.16. Tests EXP-050's destroyed-view hypothesis from the suppression side; sign-flip burden discharged by construction (suppresses the population 050 amplified).

Projection: signatures byte-identical to baseline — params **4,286,026**, dt 22.2–22.7ms (added ops: softmax+gather+pow+normalize on 512×10 under no_grad, fused, ≪ 0.1ms), ~139 epochs, ~13.4–13.5k steps. Printed train loss stays family-scale (mean-1 weights).

## Milestones

### Milestone 1: Code change implemented and passing CPU sanity
- [x] `train.py`: add constant `GCE_Q = 0.7` after `LABEL_SMOOTHING`; replace the timed-loop loss with the weighted form inside the existing autocast block:
  ```python
  outputs = model(inputs)
  ce = F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING, reduction="none")
  with torch.no_grad():
      p_true = outputs.softmax(1).gather(1, targets.unsqueeze(1)).squeeze(1)
      w = p_true.float() ** GCE_Q
      w = w / w.mean()
  loss = (w * ce).mean()
  ```
  Warmup loss mirrored identically with `warm_y` (graph identity).
- [x] Diff check: 1 file, +16/−4 — constant + two loss sites only
- [x] CPU sanity all pass first run: (a) params 4,286,026; (b) q=0 identity to 1e-6; (c) w(p^0.7) = (0.0398, 0.6156, 0.9647), mean(w)=1.000000; (d) per-row gradient = w_i × unweighted (allclose); (e) smoke 2.24→0.85

### Milestone 2: Gated launch on GPU 0 confirmed running
- [x] Reused `/tmp/exp046_composite.sh` AS-IS
- [x] GATE_DECISION D0=22.7ms ≤ 23 — GATES_CLEAR poll 1 (apps=0, load=32), pid 1767871 launched 03:36:52

### Milestone 3: Run resolved — full completion OR pre-registered branch
- [x] Trajectory family-like early as predicted; depressed still-climbing tail (intervention effect); no abort signals
- [x] No kills — gates clear poll 1, slow_streak 0 throughout
- [x] Completed: RC=0, full summary block (best 95.32, 139 ep, 13,417 steps, 469.7s total)

### Milestone 4: Verification executed (first-failure-stop) and exp-log updated
- [x] Integrity pre-condition PASS; Condition 1 FAIL (95.32 < 96.81) — first-failure-stop; results in exp-log-051.md

## Code Changes
- **train.py** (only file): (1) new constant `GCE_Q = 0.7`; (2) timed-loop loss → per-sample CE+LS reweighted by detached, mean-normalized p_true^0.7; (3) warmup loss mirrored. Why this tests the hypothesis: w → 0 as p_true → 0 sends augmentation-destroyed views' wrong-direction gradients toward zero (the exact population EXP-050's uniform margin amplified to a −2.4σ loss), while genuine samples keep ≈ full gradient; eval untouched, so any gain is realized boundary placement. Risks/edge cases: weights MUST be detached (no_grad block) — backprop through w would change the gradient geometry away from GCE's; `p_true.float()` avoids bf16 pow precision issues in the weight (weight applied to fp32 ce vector — `ce` from F.cross_entropy under autocast is fp32); mean-1 normalization keeps integrated heat family-equal (heat axis closed, must not confound); early-training near-uniform weights make the intervention self-gating (no cold-start instability).

## Configuration Changes
- `GCE_Q`: new, 0.7 — the canonical Generalized Cross Entropy exponent (Zhang & Sabuncu 2018), validated across noise rates on CIFAR-10; reproduces GCE's gradient geometry (∂loss/∂logits scaled by p^q) while preserving the recipe's LS target semantics via the detached-weight form. All other constants unchanged.

## Execution Environment
- Method: local, composite background script `/tmp/exp046_composite.sh` reused verbatim (validated 5× incl. EXP-049/050): dual launch gates (GPU-0 zero compute apps AND load < 60, poll 30s×240) → `rm -f run.log` → `uv run train.py > run.log 2>&1 &` → watchdog 44×15s (GATE_KILL D0 > 26ms; CONTENTION_KILL 4 consecutive > max(D0×1.25, 26); STARTUP_KILL tick 12; NaN; divergence eval < 15% after ep5; WALL_CAP tick 44)
- Resources: GPU 0 only (H20); VRAM ~1.6GB; CIFAR-10 cached in `data/`
- Estimated runtime: ~470–505s clean
- Log output: `<project root>/run.log` + composite stdout (background task output file)
- Tool skill: none (local)

## Abort Criteria
- NaN loss or eval < 15% after epoch 5 → research failure, no retry
- GATE_KILL / CONTENTION_KILL / STARTUP_KILL → infra: relaunch byte-identically when gates clear (max 2, then Outcome failed)
- Wall ≥ ~660s → kill, failure (goal cap 600s)

## Verification Protocol

### Verification Procedure
First-failure-stop. **Integrity pre-condition**: pristine profile — ≥200-step windows mean ≤ 23.5ms, none > 27; num_epochs 136–142; num_steps within ~1% of family ledger 13,428–13,515; printed params == 4,286,026; training_seconds == 300.0; eval lines ≤ num_epochs. Numerics by trajectory + plateau criterion (TEST evals; single-eval bands unreliable — EXP-010/048). final_test_loss is informational, not an integrity gate (the weighting may legitimately shift converged logit scale; EXP-050 precedent). Contention-caused integrity failure → rerun (infra), not a verdict.

1. **best_test_acc ≥ 96.81** (baseline 96.71 + 0.1pp): `grep "^best_test_acc:" run.log` (timeout 10s; empty ⇒ crash ⇒ `tail -n 50 run.log`). Pass ≥ 96.81. **Replicate branch**: read ∈ [96.70, 96.80] with pristine profile → replicate pair (two byte-identical gated runs; improvement only if mean ≥ 96.81). Fail < 96.70 (or replicate mean < 96.81).
2. **Within budget**: composite rc == 0 AND `grep "^total_seconds:" run.log` ≤ 600 (timeout 10s)
3. **Eval cadence**: `grep -c "eval ep" run.log` ≤ num_epochs value (timeout 10s)

Cleanup per goal Procedure: delete `run.log` at loop end (analyze housekeeping).

### Informational Metrics (Optional)
- final_test_loss: `grep "^final_test_loss:" run.log` (family ~0.185; direction informs the mechanism reading — noise filtering should NOT produce 050's CE-collapse signature)
- num_epochs / num_steps: ledger cross-check (expect family — confirms weighting is throughput-free)
- peak_vram_mb: `grep "^peak_vram_mb:" run.log`
- Plateau shape (last-8 evals): converged-flat expected; level vs EXP-027 mean is the dose-response datum resolving branches (i)/(ii)/(iii)
