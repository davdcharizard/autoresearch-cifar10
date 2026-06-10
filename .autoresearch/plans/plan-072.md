# Plan EXP-072: AugMix `all_ops=False` — geometric-only AugMix (drop 4 photometric ops)
- **Created**: 2026-06-10
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-072.md

## Closed-axis check
This is a genuinely-UNTESTED cell, not a closed-axis retry. The augmentation lever is the ONLY one that has ever lifted top-1 (EXP-002/012/052/054), and its mapped AugMix sub-levers are chain-count (EXP-052/054/055), magnitude (EXP-053), mix-distribution alpha (EXP-069), and coverage (EXP-054/055/057). The **op-SET composition** (`all_ops` flag — WHICH ops are in the menu) is orthogonal to all four and has never been probed. It does NOT contradict any High-importance insight: it is NOT a scalar/static-knob retune of a bracketed optimum (it changes the op MENU, not a magnitude/count/coverage scalar), NOT an add-a-regularizer move (it REMOVES ops, net-fewer; and it stays CPU-delivered so it's Σdt-free — no convergence/epoch cost), and NOT a logit-scale/effective-LR/graph perturbation (unlike EXP-070/071, it cannot destabilize training — it only changes which augmentation ops the CPU dataloader samples). Honest framing (carried from brainstorm): a real bidirectional probe — the 4 photometric ops (Brightness/Color/Contrast/Sharpness) may distort CIFAR-10 class-relevant color cues (removing them → cleaner diversity → possible gain) OR may add load-bearing diversity (removing them → mild regression). The circumstantial lean toward "color ops unhelpful": TA/RandAugment/AutoAugment all include color ops and all tied at 96.22 < AugMix's 96.45.

## Milestones

### Milestone 1: Code change + smoke
- [ ] train.py L171: `transforms.RandomApply([transforms.AugMix()], p=0.5)` → `transforms.RandomApply([transforms.AugMix(all_ops=False)], p=0.5)`. Single kwarg added; no other change.
- [ ] Smoke: `uv run python -c "import ast; ast.parse(open('train.py').read())"` OK; instantiate `transforms.AugMix(all_ops=False)` and confirm `_augmentation_space(31,(32,32))` returns EXACTLY 9 keys (the geometric/lossless set: ShearX, ShearY, TranslateX, TranslateY, Rotate, Posterize, Solarize, AutoContrast, Equalize) and NONE of {Brightness, Color, Contrast, Sharpness}; apply it to a dummy PIL image to confirm it runs; `git diff --name-only` == `train.py` only.

### Milestone 2: Launch on idle GPU + early gate
- [ ] Pre-launch `nvidia-smi` idle-GPU check; launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Gate (~ep8): dt steady ~8ms (AugMix runs in CPU dataloader workers — OFF the timed step, so dt unchanged from EXP-054; a 9-op vs 13-op menu does not change GPU throughput), no NaN, eval test_acc climbing normally (trajectory should track EXP-054 closely — same recipe, only the AugMix op menu differs). num_params printed 4,299,866.

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary; extract best_test_acc, compare to baseline 96.45 / bar 96.55. Expect ~91 ep, dt 8ms, wall ~585-595s, num_params 4,299,866. Most-likely best_test_acc within ±0.3pp of 96.45; bar-clearing upside iff photometric-op removal helps net.

## Code Changes
- **train.py (L171, single kwarg)**: `transforms.AugMix()` → `transforms.AugMix(all_ops=False)`. Why this tests the hypothesis: torchvision's AugMix defaults to `all_ops=True` (13 ops incl. the 4 magnitude-photometric ops Brightness/Color/Contrast/Sharpness); `all_ops=False` restricts the per-chain op-sampling pool to the original-paper 9 geometric/lossless ops. This isolates AugMix's mixing + geometric-diversity benefit from the photometric distortions, directly probing whether the color/brightness ops help or hurt clean CIFAR-10 top-1. Risks/edge cases: NONE structural — `all_ops` only changes which ops the CPU augmentation samples; it cannot change tensor shapes, params, FLOPs, the compile graph, logit scale, or the effective LR, and cannot destabilize training (unlike EXP-070/071). It stays inside the dataloader workers → Σdt-free and throughput-neutral. The only "risk" is the expected within-noise outcome or a mild regression if the removed ops were net-beneficial diversity.

## Configuration Changes
- AugMix `all_ops`: `True` (torchvision default, used implicitly by EXP-054) → `False`. All other AugMix params unchanged (severity 3, mixture_width 3, chain_depth -1, alpha 1.0), still wrapped in `RandomApply(..., p=0.5)`, still stacked with GPU Cutout16. All else byte-identical to EXP-054 (cosine peak0.2/warmup0.05/Nesterov m0.9/WD1e-4/LS0.1, batch128, seed42, compile reduce-overhead). num_params unchanged (4,299,866) — augmentation is data-side, not a model parameter.

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash.
- Resources: single idle H20 (pre-check `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`; Σdt=300s budget REQUIRES an uncontended GPU — relaunch on contention).
- Estimated runtime: ~91 epochs, dt ~8ms, Σdt ~300s, wall ~585-595s (< 600s; AugMix is CPU-side so no GPU throughput change vs EXP-054's 593s; a smaller op menu is if anything marginally CHEAPER per-op on CPU, but the 8-worker dataloader already hides AugMix latency under the Σdt-gated step, so wall ≈ EXP-054). Monitor the 600s wall (recipe is wall-tight) but this change adds no GPU cost.
- Log output: `run.log` in project root.
- Tool skill: none (local).

## Abort Criteria
- Loss NaN/inf (essentially impossible from an augmentation op-menu change) or eval test_acc not climbing by ~ep5.
- dt ≥13ms sustained (cudagraph break / GPU contention — NOT expected from a CPU-side aug change; indicates contention): kill, relaunch on a clean idle GPU.
- Wall projecting > 600s at the ep~40 mark (if the CPU dataloader somehow throttles — unlikely since fewer ops ≤ current cost): note but let complete if Σdt respected (per EXP-061/065 wall-breach precedent).
- No output / hung > 3 min.

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline`); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after exit, `grep -aE "^best_test_acc:" run.log`; parse the float; PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:|^training_seconds:" run.log`; confirm summary printed, `training_seconds ≈ 300`, `total_seconds < 600`, `num_params == 4,299,866` (UNCHANGED — aug is data-side), and `grep -ciaE "nan|traceback|error" run.log` == 0. (If total_seconds breaches 600 by a small margin while training_seconds≈300 and metric trustworthy → no-improvement per EXP-061/065 precedent.)
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == `train.py` only; prepare.py/eval untouched; evaluate() called once/epoch; no new deps; seed 42 unchanged; ran uncontended (steady ~8ms dt).
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep breach; crash if no metrics / abort.
- Timeout: 12 min wall. Cleanup: `rm run.log` after recording.

### Informational Metrics (Optional)
- `num_epochs`/`num_steps`, `final_test_loss`, `peak_vram_mb`: `grep -aE "^num_epochs:|^num_steps:|^final_test_loss:|^peak_vram_mb:" run.log` — confirm ~91 ep (throughput unchanged) and whether the run tracks EXP-054 (best_test_acc vs 96.45, final_test_loss vs 0.1968). A meaningful deviation in EITHER direction is informative: a top-1 GAIN with the geometric-only menu would newly establish AugMix's photometric ops are counterproductive for clean CIFAR-10; a regression confirms the full 13-op diversity is load-bearing.
