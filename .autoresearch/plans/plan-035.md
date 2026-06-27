# Plan EXP-035: Periodic SAM — sharpness-aware ascent-descent every 5th step (ρ=0.05, BN-protected)
- **Created**: 2026-06-10
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-035.md

## Milestones

### Milestone 1: SAM step implemented and sanity-checked
- [x] On branch `autoresearch/exp-035` (cut from `autoresearch/dev` @ 1990397), edit `train.py`:
  1. New constants after LABEL_SMOOTHING: `SAM_RHO = 0.05` and `SAM_EVERY = 5  # full SAM ascent-descent on every 5th step (LookSAM-style periodic)`.
  2. In the timed step, after the existing `loss.backward()` insert the SAM branch, executed when `step % SAM_EVERY == 0` (pre-increment step counter — steps 0, 5, 10, …):
     ```python
     if step % SAM_EVERY == 0:
         with torch.no_grad():
             sam_params = [p for p in base_model.parameters() if p.grad is not None]
             grad_norm = torch.norm(
                 torch.stack([p.grad.norm(p=2) for p in sam_params]), p=2
             )
             scale = SAM_RHO / (grad_norm + 1e-12)
             e_ws = [p.grad * scale for p in sam_params]
             for p, e_w in zip(sam_params, e_ws):
                 p.data.add_(e_w)
         optimizer.zero_grad(set_to_none=True)
         for m in bn_modules:
             m.momentum = 0.0  # freeze running stats on the perturbed pass
         with torch.autocast("cuda", dtype=torch.bfloat16):
             sam_loss = F.cross_entropy(
                 base_model(inputs), targets, label_smoothing=LABEL_SMOOTHING
             )
         sam_loss.backward()
         for m in bn_modules:
             m.momentum = 0.1  # restore default BN momentum
         with torch.no_grad():
             for p, e_w in zip(sam_params, e_ws):
                 p.data.sub_(e_w)
     optimizer.step()
     ```
     (replaces the bare `optimizer.step()`; the existing zero_grad/forward/backward before it are untouched — non-SAM steps are byte-identical to baseline).
  3. Before the training loop: `bn_modules = [m for m in base_model.modules() if isinstance(m, nn.BatchNorm2d)]`.
  4. Warmup block: after the existing 3 compiled warmup iterations, add ONE eager warmup pass (`base_model(warm_x)` fwd + backward under autocast) so cudnn.benchmark algo selection for the eager path lands in startup, not in step 0; existing zero_grad + synchronize already follow.
  5. Loss logging unchanged — `train_loss_f = loss.item()` keeps referencing the FIRST-pass (unperturbed) loss so the trace stays comparable to baseline.
  6. NOTHING else changes: model, transforms, loaders, schedule, eval, summary all byte-identical.
- [x] Sanity: AST parse OK; `git diff` confirms edits only at the 4 sites (constants, bn_modules line, warmup add, step branch); eval path (`evaluator.evaluate(base_model, device)`) untouched.

### Milestone 2: Run 1 launched with gates and SAM-aware watchdog
- [ ] Launch gates: zero GPU-0 compute apps AND host 1-min load < 60, poll 30s up to 2h.
- [ ] Composite watchdog (15s ticks, 44): contention 4 consecutive windows >33ms (≈1.2× expected 27.6); cost-model GATE_KILL if windowed dt >31ms on 3 consecutive windowed ticks within the first 10 (epochs would fall below ~105, pre-losing the arithmetic); STARTUP_KILL tick 10; NaN/inf; divergence eval <15% after ep5; wall cap 600s.
- [ ] Early readout: windowed dt ≈ 26.5–28.5ms (windows span ≥150 steps, so per-window values are the 4:1 mixed average).

### Milestone 3: Completion and readout
- [ ] Full run: rc=0, total ≈ 440–470s, epochs ≈ 108–118, eval_lines = num_epochs, params 4,286,026 (architecture unchanged).
- [ ] No pre-registered fallback: a GATE_KILL on the cost model means the eager second pass is far costlier than measured ratios imply — record the dt datum and stop (k or ρ retunes are new experiments, not adjustments).

## Code Changes
- **train.py** (only file): 2 constants; 1 bn_modules list; 1 eager warmup pass; 1 SAM branch in the timed step. Why this tests the hypothesis: on every 5th step the descent direction comes from the WORST-CASE point in a ρ-ball (explicit sharpness pressure) while 4/5 steps and everything else stay byte-identical — any plateau-level shift vs baseline isolates the flat-minima mechanism at a wall-cost (~26 epochs) below the published gain it must beat.
- Risks/edge cases: (a) eager second pass costlier than the 1.22× compile ratio suggests (it also skips inductor fusions on elementwise) → GATE_KILL catches; (b) `p.grad * scale` allocates ~17MB fp32 per SAM step — VRAM headroom is 60+GB, fine; (c) first-pass grads are populated for ALL params (conv/linear/BN affines) since backward is unchanged — canonical SAM perturbs all of them; (d) BN momentum toggling happens OUTSIDE any compiled graph (base_model is eager) — no guard recompiles; the compiled first pass never sees attribute changes; (e) sam_loss is intentionally not logged into the EMA trace.

## Configuration Changes
- SAM_RHO = 0.05 — the canonical CIFAR value (Foret et al. Table 1 uses ρ=0.05 for CIFAR-10 WRN); NOT tuned; interior values flagged for a follow-up only if the sign is positive but sub-bar.
- SAM_EVERY = 5 — LookSAM's validated k; the only dose whose epoch-deficit (−~26) sits below the literature gain (+0.5–0.8 fixed-epoch, shrinking under heavy aug).
- All baseline recipe constants unchanged.

## Execution Environment
- Method: local composite background Bash (dual launch gates → `rm -f run.log` → `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` → 44×15s watchdog → wait → rc + summary greps + eval tails), branch `autoresearch/exp-035`, GPU 0 only.
- Resources: VRAM ~1650MB (baseline + e_ws copies); 8 loader workers.
- Estimated runtime: ~440–470s total (300 charged + ~11s startup + ~113 epochs × ~1.17s evals/stalls). ≥130s margin under the cap.
- Log output: `run.log` (no tee); post-hoc awk profile authoritative; delete run.log after analysis.
- Tool skill: none (local).

## Abort Criteria
- **Startup gate**: no step lines by tick 10 (150s) → kill.
- **Cost-model GATE_KILL**: windowed dt >31ms ×3 consecutive within first 10 ticks at clean load → kill (~90–150s); no fallback, record dt datum.
- **Contention**: 4 consecutive windows >33ms → kill, contaminated, rerun once after both gates re-clear.
- **Divergence**: NaN/inf loss, or any eval <15% after epoch 5 → kill (SAM ascent at peak LR destabilizing would show here).
- **Wall cap**: >600s → kill, failure.
- **Crash** (rc≠0): code-error fix + resubmit per execute-skill rules (max 2).

## Verification Protocol

### Verification Procedure
First-failure-stop, in order. Baseline from `exp-index.sh baseline`: **96.71** @ 1990397; bar = **96.81**. σ context (EXP-027): baseline mean ≈96.57, σ ≈0.16.

1. **best_test_acc ≥ 96.81**:
   - Command: `grep "^best_test_acc:" run.log`. Empty ⇒ crash classification.
   - Pre-condition (profile): `tr '\r' '\n' < run.log | grep -E "^step [0-9]+" | sed -E 's/^step 0*([0-9]+) ep [0-9]+ \(([0-9.]+)%\).*/\1 \2/' | awk 'NR>1{ms=($2-p2)*3000/($1-p1); n++; s+=ms; if(ms>33)c++} {p1=$1;p2=$2} END{printf "%d win, mean %.1f ms, slow>33: %d\n", n, s/n, c}'` — require ≤2 slow windows AND mean ≤31ms AND num_epochs ≈ 139×22.4/mean_dt ±4. Contaminated ⇒ rerun once (gates re-cleared first).
   - Integrity sub-check: `grep "^num_params:" run.log` = 4,286,026 (architecture untouched); training_seconds = 300.0; eval_lines = num_epochs; on a bar-pass additionally confirm the plateau (last ~15 evals) sits above the baseline band rather than a single-eval spike.
   - Timeout: greps instant; missing run.log = infrastructure failure.
2. **Completes within budget**: rc=0 AND `grep "^total_seconds:" run.log` ≤ 600.
3. **Validation ≤ once/epoch**: eval_lines ≤ num_epochs.

On first failure: stop, classify, proceed to analyze.

### Informational Metrics (Optional)
- **Measured SAM-step cost**: mean windowed dt → back out per-SAM-step cost = 5×mean − 4×22.4 (the sharpness-axis pricing datum, recorded regardless of verdict).
- **Plateau level & test_loss**: last-15-evals mean and final_test_loss vs baseline family (~96.6 / ~0.185) — SAM should show as a LEVEL shift with equal-or-better loss; loss-better/acc-equal would echo the smoothing signature (EXP-011/032) and close the axis decisively.
- **ep1 eval** (family ~38 at 97 steps; SAM steps at warmup LR are near-no-ops, expect family-normal).
- total_seconds, startup_seconds (expect ~11s: +1 eager warmup pass), peak_vram_mb (~1650), num_epochs.
