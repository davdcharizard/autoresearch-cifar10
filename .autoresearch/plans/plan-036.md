# Plan EXP-036: Periodic / sparse SAM (Sharpness-Aware Minimization on every 5th step, ρ=0.05)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-036.md

## Milestones

### Milestone 1: SAM step implemented + compile-compatible (smoke-checked)
- [ ] Add constants `SAM_RHO = 0.05`, `SAM_EVERY = 5` near the hyperparameters.
- [ ] Hoist the param list `sam_params = [p for p in model.parameters() if p.requires_grad]` once before the loop.
- [ ] Implement the SAM two-pass step inside the batch loop, gated by `step % SAM_EVERY == 0` (vectorized perturbation via `torch._foreach_*`, perturbation under `torch.no_grad()`); plain single-pass Nesterov-SGD otherwise. LR/logging/Cutout/aug unchanged.
- [ ] Smoke test (≤~60s, small wall-clock cap): run a handful of steps confirming (a) no CUDA-graph / dynamo error from the two compiled forwards per SAM step, (b) loss is finite and decreasing, (c) params restored after each SAM step (no drift). If a cudagraph error appears, apply the fallback ladder (see Code Changes → risks) and re-smoke.
- [ ] Confirm `git diff --name-only` == train.py only; `python -c "import ast; ast.parse(open('train.py').read())"` OK; params unchanged (4,299,866).

### Milestone 2: Full run + throughput (epoch-wall) check
- [ ] Launch `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1` (background).
- [ ] Confirm no NaN/errors; **verify realized epochs ≥ ~75** (the compute-wall fairness gate — if epochs ≪75, the test is compute-confounded and N must be raised).
- [ ] Confirm mean dt ≈ 9–10ms (≈1.2× the 8ms baseline, as predicted for SAM_EVERY=5).

### Milestone 3: Metrics extracted and verified
- [ ] Parse summary; record best_test_acc, num_epochs, num_steps, dt, final_test_loss, params.
- [ ] Run verification protocol vs bar 96.32 / baseline 96.22.

## Code Changes
- **train.py**: Add sparse SAM to the training loop. No architecture/data/eval change.
  1. **Constants** (near CUTOUT_SIZE):
     ```python
     SAM_RHO = 0.05    # SAM neighborhood radius (Foret et al. 2021; standard CIFAR value)
     SAM_EVERY = 5     # apply the SAM ascent on every Nth step (sparse SAM, LookSAM-style) to cap
                       # the 2x-per-step cost at ~1.2x mean → ~76 ep ≈ the ~77-ep convergence point
                       # (dodges the compute/epoch wall; project-insights High). Other steps: plain SGD.
     ```
  2. **Hoist param list** (before the `while` loop, after optimizer creation):
     ```python
     sam_params = [p for p in model.parameters() if p.requires_grad]
     ```
  3. **SAM two-pass step** — replace the current single-pass body:
     ```python
     optimizer.zero_grad()
     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
         outputs = compiled_model(inputs)
         loss = F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING)
     loss.backward()
     optimizer.step()
     ```
     with:
     ```python
     do_sam = (step % SAM_EVERY == 0)
     optimizer.zero_grad()
     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
         outputs = compiled_model(inputs)
         loss = F.cross_entropy(outputs, targets, label_smoothing=LABEL_SMOOTHING)
     loss.backward()  # first pass: clean gradient (also the logged loss)

     if do_sam:
         with torch.no_grad():
             grads = [p.grad for p in sam_params]
             # global L2 norm of the gradient, then ascent step e = rho * g / ||g||
             gnorm = torch.norm(torch.stack([g.norm() for g in grads]))
             scale = SAM_RHO / (gnorm + 1e-12)
             e_ws = torch._foreach_mul(grads, scale)
             torch._foreach_add_(sam_params, e_ws)   # ascend to worst-case neighbor
         optimizer.zero_grad()
         with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
             outputs2 = compiled_model(inputs)
             loss2 = F.cross_entropy(outputs2, targets, label_smoothing=LABEL_SMOOTHING)
         loss2.backward()  # second pass: sharpness-aware gradient at the perturbed point
         with torch.no_grad():
             torch._foreach_sub_(sam_params, e_ws)   # restore original weights
     optimizer.step()  # uses the SAM gradient on SAM steps, the clean gradient otherwise
     ```
     The `train_loss_f = loss.item()` logging line stays (logs the clean first-pass loss). The LR-assignment block is unchanged and still precedes this.
- **Why this tests the hypothesis**: on ~20% of steps the optimizer follows the gradient of the loss at the *worst-case* nearby weights, biasing SGD toward a flatter minimum that generalizes better — the canonical fixed-architecture top-1 generalization lever, on the diagnosed binding constraint (generalization at fixed k=4 capacity).
- **Risks / edge cases**:
  - **torch.compile(reduce-overhead) × two forwards + in-place perturbation**: in-place `_foreach_add_/sub_` on params mirrors what `optimizer.step()` already does between forwards (cudagraphs tolerate value mutation at static addresses), and the first output is consumed by `loss.backward()` before the second forward reuses the static buffer — so this *should* work. Fallback ladder if a cudagraph/dynamo error appears at smoke: (i) add `torch.compiler.cudagraph_mark_step_begin()` before each compiled forward; (ii) if still failing, switch `compiled_model = torch.compile(model)` (default mode) — accept some dt loss; (iii) last resort, eager for SAM. Record the choice in exp-log `## Decisions` (autopilot). Any fallback that drops dt materially is noted as a throughput confound in analysis.
  - **Compute-wall confound**: if epochs fall ≪75, a regression is under-training not a SAM null — gate in Milestone 2; would require raising SAM_EVERY to retest (not done this run).
  - **Grad norm under autocast/bf16**: grads are fp32 (params are fp32; autocast only affects forward compute), so the norm is stable; `+1e-12` guards div-by-zero.

## Configuration Changes
- SAM_RHO: (new) → 0.05 (standard CIFAR SAM radius, Foret et al. 2021).
- SAM_EVERY: (new) → 5 (sparse cadence sized from the project's ~15% epoch slack: mean cost ≈ 1+1/5 = 1.2× → ~76 ep ≈ convergence).
- All else unchanged: k=4, TA+Cutout(16), PEAK_LR 0.2 cosine-to-0, Nesterov SGD m0.9, WD 1e-4, LS 0.1, batch 128, seed 42, torch.compile(reduce-overhead). Params 4,299,866.

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`, background.
- Resources: single NVIDIA H20; ~454 MB + a transient second activation set (SAM steps) — still tiny vs 98 GB.
- Estimated runtime: ~405–430s wall (300s compute budget; SAM steps cost ~2× but only 20% of steps).
- Log output: stdout+stderr → run.log. Step lines use `\r` (extract dt/lr via `tr '\r' '\n'`).
- Tool skill: none (local).

## Abort Criteria
- NaN/inf loss (SAM perturbation destabilizes) → kill, Outcome failed.
- CUDA-graph/dynamo error that survives the full fallback ladder → kill, Outcome failed, log the trace.
- No run.log growth >120s after launch → kill.
- dt blows up (mean ≫12ms → epochs ≪70, severe compute confound) → let it finish only if informative; otherwise note. Total wall approaching 600s → kill.
- Loss diverging (debiased train loss rising for many consecutive evals) → research failure, do not retry.

## Verification Protocol

### Verification Procedure
Baseline = **96.22** (commit 6c417a4), bar = baseline + 0.1 = **96.32**.

1. **Cond 1 — primary metric clears bar** (`best_test_acc >= 96.32`): `grep -aE "^best_test_acc:" run.log` → PASS iff ≥ 96.32.
2. **Cond 2 — clean completion within budget**: `grep -c -a "Traceback\|RuntimeError\|CUDA error\|NaN\|Killed" run.log` == 0; summary block present; `grep -aE "^total_seconds:" run.log` < 600.
3. **Cond 3 — no constraint violations**: `git diff --name-only` == only train.py; `grep -aE "^num_params:" run.log` == 4,299,866; eval-count (`grep -c -a "eval ep" run.log`) == `num_epochs`; core torch only (no new deps); seed 42 unchanged.
4. **Fairness / attribution gate**: realized `num_epochs` ≥ ~75 and mean dt ≈ 9–10ms → the SAM test is near-throughput-fair. If epochs ≪75, flag the result compute-confounded in analysis (don't credit/blame SAM cleanly). Note whether any compile fallback was used (dt impact).

First failed necessary condition ⇒ stop and classify (Cond1 fail on a valid run ⇒ no-improvement; Cond3 fail ⇒ invalid; crash ⇒ crash).

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — the compute-wall fairness check (target ≥75 ep)
- final_test_loss: `grep -aE "^final_test_loss:" run.log` — vs baseline 0.195 (SAM often lowers loss too)
- mean dt: `tr '\r' '\n' < run.log | grep -oE "dt: [0-9]+ms" | tail` — confirm ~1.2× the 8ms baseline
