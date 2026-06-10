# Plan EXP-060: AutoAugment(CIFAR10 learned policy) replacing AugMix

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-060.md

## Closed-Axis Justification (required: augmentation is a High-Importance "closed" axis)
The goal-learnings mark the augmentation axis CLOSED, incl. "policy (EXP-014)" and "do NOT propose further CPU-aug variants" (EXP-059). This plan proceeds anyway, with explicit justification:
1. **The "policy axis closed (EXP-014)" claim was EMPIRICALLY FALSIFIED.** EXP-014 declared the auto-aug policy axis closed because RandAugment ≈ TrivialAugment (96.19 ≈ 96.22). But AugMix (EXP-052/054) — a *different policy family* — subsequently BEAT TrivialAugment by +0.23pp (96.45 vs 96.22). This proves policy FAMILY can move top-1 here by **more than the +0.1pp bar**. The closure was about two near-identical RandAugment/TA policies, not about the policy-family axis being flat.
2. **AutoAugment(CIFAR10) is a genuinely untried, distinct policy family** — and the only CIFAR-SPECIALIZED one (25 sub-policies RL-searched to maximize CIFAR-10 accuracy; canonical AutoAugment+Cutout SOTA recipe). The EXP-059 "CPU-aug closed" entry concerns AugMix's internal knobs (magnitude/width/coverage) and the CPU-vs-GPU delivery question — NOT a new policy family. So this is not a literal retry of any closed sub-lever.
3. **Free w.r.t. the Σdt/epoch budget** (CPU dataloader), one-line change, low-risk failure mode (converged regression at worst).
- **Honest EV caveat**: the TrivialAugment paper's thesis is AA ≈ TA on average, so the modal outcome is AA ≈ TA (~96.2) < AugMix (96.45) → no-improvement. Counter-evidence: in the clean-CIFAR literature AutoAugment's learned policy is tuned for clean top-1 (often ≥ AugMix, whose headline win is corruption robustness). This is a principled, distinct, cheap probe of the one untried policy family — consistent with NEVER-STOP "continue principled long-shots."

## Milestones

### Milestone 1: Code change implemented and smoke-checked
- [ ] Replace `transforms.RandomApply([transforms.AugMix()], p=0.5)` (train.py L171) with `transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)` (full native coverage); update the explanatory comment.
- [ ] Smoke: `uv run python -c "import ast; ast.parse(open('train.py').read())"` parses; `git diff --name-only` shows train.py only.
- [ ] Smoke: instantiate `train_tf` and run one CIFAR-10 PIL sample through it → returns a finite (3,32,32) tensor (confirms AutoAugment receives PIL before ToTensor and composes cleanly).

### Milestone 2: Run launched on an idle GPU + early wall/feasibility gate
- [ ] Pre-launch idle-GPU check via `nvidia-smi` (pick a GPU at ~0 MiB / 0%).
- [ ] Launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` in background.
- [ ] EARLY WALL GATE (~step 1500–3000 / ep4–8): from `ps etimes` (real wall) + log step deltas, project total wall = real ms/step × est-total-steps + ~100s startup/eval. **CPU aug gates on the 600s WALL, not Σdt** (Protocol Finding, EXP-052). AugMix-w3-p0.5 finished tight at 593s with high variance (EXP-054). If projected wall > ~560s, switch AutoAugment to `RandomApply([AutoAugment(CIFAR10)], p=0.5)` (halves coverage → ~halves AA's CPU cost AND moves to the validated ~50%-coverage optimum) and relaunch. This also serves as the over-regularization fallback (see Abort Criteria).
- [ ] Confirm dt steady (~8ms expected; CPU aug shouldn't change the GPU step), no NaN, loss descending.

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary block.
- [ ] Extract metrics, compare to baseline, render conditions.

## Code Changes
- **train.py (L160–171)**: Replace the AugMix `RandomApply` line with `transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10)`. Placed in the same Compose slot (after `RandomHorizontalFlip`, before `ToTensor`) so it receives PIL images — AutoAugment operates on PIL/uint8. Update the comment block to describe the CIFAR10 learned policy and cite the closed-axis justification. This tests the hypothesis that the CIFAR-specialized learned policy supplies a better augmentation-diversity distribution than tuned AugMix, through the same epoch-free CPU delivery. Edge cases: (a) full coverage may over-regularize like full-coverage GPU AugMix (EXP-057) — mitigated by AA's per-op internal probabilities (moderate effective strength) and the p=0.5 fallback; (b) wall risk — gated early (Milestone 2).

## Configuration Changes
- Train augmentation: `RandomApply([AugMix()], p=0.5)` -> `AutoAugment(AutoAugmentPolicy.CIFAR10)` (full coverage). Everything else byte-identical to EXP-054: k=4 WideResNet-20, GPU Cutout(16), cosine peak0.2 / warmup0.05, Nesterov m0.9, WD 1e-4, LS 0.1, batch 128, seed 42, compile(reduce-overhead). No param-count change (augmentation only).
- Conditional fallback (in-flight, only if Milestone-2 gate trips): wrap AutoAugment in `RandomApply([...], p=0.5)`.

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash (run takes >2 min).
- Resources: single idle H20 (must pre-check nvidia-smi; shared node, contention corrupts the dt-budget — idle-GPU gating required per infra-errors).
- Estimated runtime: ~91 epochs, dt ~8ms, Σdt ~300s, wall ~540–595s (the binding risk; gate early).
- Log output: `run.log` in project root (tee'd via redirection); source of truth for metrics + wall projection.
- Tool skill: none (local).

## Abort Criteria
- Projected total wall > ~600s at the early gate AND the p=0.5-coverage fallback still projects > ~580s → abort (wall-infeasible).
- Loss NaN/inf or not descending by ep5.
- Over-regularization / underfit signature at the early gate: epoch-1 test_acc far below the EXP-054 trajectory (EXP-054 ep1 ~55%) and/or projected final epochs ≪ 91 from an inflated dt — if full-coverage AA looks harsh (mirroring EXP-057's full-coverage over-reg), switch to the p=0.5 fallback rather than aborting.
- No output / process hung > 3 min with no new log lines.
- GPU contention mid-run (dt drifts ≫ 8ms, wall/Σdt ≫ 2.5×): kill, relaunch on a clean idle GPU (fair-comparison requirement).

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline` on experiment-indices/improve-cifar10-test-accuracy.tsv); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after the run exits, `grep -aE "^best_test_acc:" run.log`. Parse the float. PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:|^peak_vram_mb:" run.log`; confirm the summary block printed, `total_seconds < 600`, total wall (from `ps`/start-to-end) < 10 min, `num_params == 4,299,866` (unchanged — aug-only), and `grep -ciaE "nan|traceback|error" run.log` shows no fatal errors.
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` lists ONLY train.py; prepare.py / Eval.evaluate() untouched; `evaluate()` called ≤1/epoch (unchanged loop); no new deps (AutoAugment is core torchvision 0.24.1, already present); seed 42 unchanged; ran on an uncontended GPU (steady ~8ms dt, wall/Σdt ≲1.5×).
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep/seed breach; crash if no metrics.
- Timeout: 10 min wall (hard cap per goal). Cleanup: `rm run.log` after recording metrics.

### Informational Metrics (Optional)
- peak_vram_mb: `grep -aE "^peak_vram_mb:" run.log`
- num_epochs / num_steps: `grep -aE "^num_epochs:|^num_steps:" run.log` — epochs vs EXP-054's 91 (confirms aug stayed epoch-free / fair).
- final_test_loss: from run.log tail — compare to EXP-054's 0.1968 (underfit vs over-reg vs converged diagnosis).
