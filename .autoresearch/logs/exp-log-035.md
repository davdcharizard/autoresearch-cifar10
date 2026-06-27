# Experiment Log EXP-035: Periodic SAM — sharpness-aware ascent-descent every 5th step (ρ=0.05, BN-protected)

## Execution
- **Created**: 2026-06-10
- **Brainstorm**: brainstorm/brainstorm-035.md
- **Plan**: plans/plan-035.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-035
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
All Milestone-1 edits applied on `autoresearch/exp-035` (cut from `autoresearch/dev` @ 1990397), 38 insertions / 0 deletions at exactly the 4 planned sites: (1) constants `SAM_RHO = 0.05`, `SAM_EVERY = 5`; (2) one eager warmup pass through `base_model` appended to the compile-warmup block (cudnn.benchmark algo selection for the eager SAM pass lands in uncharged startup); (3) `bn_modules` list built once before the loop; (4) SAM branch between `loss.backward()` and `optimizer.step()`, active when `step % SAM_EVERY == 0` (pre-increment counter → steps 0, 5, 10, …): global-norm scaled perturbation `e_w = grad × ρ/‖g‖` applied in-place to `p.data`, zero_grad, BN momentum→0, perturbed forward/backward through EAGER `base_model` under bf16 autocast, BN momentum→0.1, in-place restore, then the unchanged `optimizer.step()` consumes the perturbed-point gradients. Non-SAM steps byte-identical to baseline; loss EMA/print still uses the first-pass (unperturbed) loss; eval path untouched.

### Surprises & Discoveries
- None at implementation time. AST clean on first try; the branch slots between backward and step without touching any baseline line (pure insertions).

### Decisions
- Perturbed pass through eager `base_model` instead of compiled `model` (per plan): BN `momentum` attribute toggling inside a compiled graph risks guard recompiles every SAM step; eager is ~1.22× slower per pass but unconditionally safe. Weights are shared, so gradients land on the same parameters either way.
- Perturbation/restore via `p.data.add_/sub_` (same tensor identity) so the compiled first pass sees no parameter-object changes across steps.
- BN running stats frozen with momentum=0 (davda54/sam `disable_running_stats` pattern): normalization still uses batch stats on the perturbed pass, but eval-time constants only ever reflect unperturbed weights (EXP-029 normalization law).

## Run Log

### Run 1
- **Description**: Full run on GPU 0 of the baseline recipe + periodic SAM: every 5th step takes the descent gradient from the worst-case point in a ρ=0.05 ball (explicit flat-minima pressure), other 4/5 steps byte-identical to baseline. Expected: windowed dt ≈ 26.5–28.5ms (4:1 mix of 22.4ms baseline steps and ~48ms SAM steps), ~108–118 epochs, total ~440–470s. Hypothesis: sharpness pressure raises the converged plateau LEVEL by more than the ~26-epoch deficit cost → best ≥ 96.81. Falsified by: windowed dt >31ms ×3 early (cost model wrong, GATE_KILL, no fallback), or a clean converged plateau below the bar (closes the sharpness axis).
- **Job ID**: local background composite, Claude task byx00usbp
- **Log file**: run.log
- **WandB**: N/A
- **Status**: completed (rc=0)
- **Started**: 2026-06-10 (gates passed at poll 1, load 15)
- **Ended**: 2026-06-10 (454s wall incl. composite overhead)
- **Observations**: Fully clean: 215 windows mean 27.7ms, 0 slow >33ms, load 9–11 throughout — the 4:1 mixed dt landed exactly on the cost model (back-solved per-SAM-step cost = 5×27.7 − 4×22.4 = **48.9ms**, predicted ~48). 112 epochs (model predicted 108–118). No instability at peak LR (no NaN, loss trace family-normal), no divergence; startup 11.4s (+1.4 for the eager warmup pass), VRAM 1639.8MB (+27). Trajectory converged smoothly; best 96.25 at ep108; final 5 evals 96.17–96.25 flat; final_test_loss 0.1945 — slightly WORSE than the baseline family's ~0.185, no flatness signature in the loss either.
- **Key Metrics**: best_test_acc 96.25 | final 96.18 | final_test_loss 0.1945 | training_seconds 300.0 | total_seconds 430.6 | startup 11.4s | VRAM 1639.8MB | epochs 112 | steps 10,835 | params 4,286,026 | dt mean 27.7ms | per-SAM-step 48.9ms

## Experimental Adjustments
(none yet)

## Errors & Dead Ends
(none yet)

## Verification Results

### Conditions Checked
1. **best_test_acc ≥ 96.81 (bar = baseline 96.71 + 0.1)** — **FAIL**. `grep "^best_test_acc:" run.log` → 96.25%. −0.56 vs bar; −0.46 vs recorded baseline; −0.32 vs baseline mean ≈ 2σ — a real measured loss.
   - Pre-condition (profile): **PASS** — 215 windows, mean 27.7ms ≤ 31; slow >33ms: 0; num_epochs 112 within 139×22.4/27.7 = 112.4 ±4; num_params 4,286,026 (architecture untouched); training_seconds 300.0; eval_lines 112 = num_epochs. Clean, trustworthy run.
2. **Completes within budget** — not evaluated (first-failure-stop). Informationally: rc=0, total_seconds 430.6 ≤ 600 ✓.
3. **Validation ≤ once/epoch** — not evaluated (first-failure-stop). Informationally: 112 = 112 ✓.

**Informational**: per-SAM-step cost 48.9ms (cost model exact: eager perturbed pass + overhead ≈ baseline step × 2.18); deficit arithmetic: −27 epochs × ~0.014 ≈ −0.38 predicted vs −0.32 observed vs mean — the epoch deficit explains the ENTIRE result within noise, i.e., SAM's retained level gain ≈ +0.06 ≈ ZERO; test_loss 0.1945 vs family ~0.185 — no flatness signature in loss either (contrast EXP-011/032 smoothing which IMPROVED loss); plateau still rising at end (best at ep108 of 112) — the 112-epoch run is slightly tail-truncated, consistent with the max-statistic's plateau-length sensitivity.

## Human Notes
(autopilot — none)
