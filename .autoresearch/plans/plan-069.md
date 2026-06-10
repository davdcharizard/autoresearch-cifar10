# Plan EXP-069: AugMix mixing-concentration alpha 1.0 → 2.0
- **Created**: 2026-06-10
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-069.md

## Closed-axis check
The augmentation axis is declared "fully exhausted" in goal-learnings, but the documented closures map specific sub-levers: chain-COUNT/width (EXP-055), per-op MAGNITUDE/severity (EXP-053), COVERAGE (EXP-055 <50%, EXP-057 =100%), GPU delivery (EXP-056/057/059), policy family (EXP-012/014/060), label-mixing (EXP-011/018), cooldown (EXP-033/034/035/063), border/occlusion-pattern (EXP-037/048). The AugMix **internal mixing-weight distribution `alpha`** (concentration of the Dirichlet over chains + Beta clean-mix) is NOT among them — it is orthogonal to width (count) and severity (magnitude) and is genuinely untested. project-insights line 295 (High) states augmentation diversity is the ONLY lever that lifts top-1; line 297 lists internal sub-knobs as untried. This plan does NOT contradict a High-importance insight — it probes the one wall-safe untested cell of the only productive axis. Critically, alpha is **mean-preserving** (does not change average augmentation strength or coverage), so it sidesteps the strength/coverage failure modes (EXP-053/055/057) that the closures established, AND it is **wall-neutral** (no op-count change → no AugMix CPU cost increase → no >600s breach risk, the recurring EXP-061/065/066 failure).

## Milestones

### Milestone 1: Code change + smoke
- [ ] In train.py L171, change `transforms.RandomApply([transforms.AugMix()], p=0.5)` → `transforms.RandomApply([transforms.AugMix(alpha=2.0)], p=0.5)`. Single kwarg; everything else byte-identical to EXP-054.
- [ ] Update the adjacent comment to note alpha=2.0 (concentrate Dirichlet/Beta mixing toward the mean — faithful 3-chain blend, mean-preserving, EXP-069).
- [ ] Smoke: `uv run python -c "import ast; ast.parse(open('train.py').read())"` OK; confirm `transforms.AugMix(alpha=2.0)` constructs without error (signature verified: `AugMix(severity=3, mixture_width=3, chain_depth=-1, alpha=1.0, all_ops=True, ...)`, torchvision 0.24.1); `git diff --name-only` == `train.py` only.

### Milestone 2: Launch on idle GPU + early gate
- [ ] Pre-launch `nvidia-smi` idle-GPU check; launch `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1` (background).
- [ ] Gate (~ep8): dt steady ~8ms (CPU-side aug change must NOT affect GPU step time — single-shape reduce-overhead graph intact), no NaN, eval test_acc climbing normally. Early real-load wall projection: eval-inclusive ms/step × est-total-steps + startup < 600s — alpha is wall-neutral vs EXP-054 (593s), so expect ~590s; if the early projection exceeds ~600s, note it (base-recipe AugMix wall variance, not caused by this compute-free change — cf. EXP-065 602.5s).

### Milestone 3: Completion + verification
- [ ] Run exits 0, prints summary; extract best_test_acc, compare to baseline 96.45 / bar 96.55. Expect ~91 ep, dt 8ms, params 4,299,866 unchanged.

## Code Changes
- **train.py (L171)**: `transforms.AugMix()` → `transforms.AugMix(alpha=2.0)`. Why: alpha is the concentration of the Dirichlet(alpha,…) weights mixing the 3 augmentation chains AND the Beta(alpha,alpha) weight mixing the chain-combination with the clean image. Default alpha=1.0 ⇒ Beta(1,1)=Uniform (clean-mix weight often extreme) and Dirichlet(1,1,1)=uniform-simplex (one chain often dominates → effectively single-chain). alpha=2.0 ⇒ both concentrate toward their means: every augmented image becomes a consistent ~50/50 clean-mix that genuinely blends all three chains — amplifying the "multi-chain Dirichlet mixing + clean convex-mix" structure project-insights line 68 credits for AugMix's win. Mean of both distributions is unchanged (mean-preserving): average augmentation strength and effective 50% coverage are identical to EXP-054 — only the per-image *variance* of the mix drops. Risk/edge case: lower-variance averaging of 3 chains can partially cancel distortions → a slightly softer net image, behaving like a mild strength reduction (potential small regression, within the scalar-knob band). No effect on param count, GPU step, or compile graph (CPU dataloader transform only).

## Configuration Changes
- AugMix `alpha`: 1.0 (torchvision default, EXP-054) → 2.0. Rationale: probes the mixing-weight-distribution sub-axis of the augmentation-diversity lever (the only documented top-1 lever, EXP-012/052/054). 2.0 is a clean 2× concentration — large enough to meaningfully change the mix variance, small enough to stay near the tuned operating point. All else byte-identical to EXP-054: w3 (mixture_width default 3), chain_depth=-1, severity=3, all_ops=True, RandomApply p=0.5, GPU Cutout16, cosine peak0.2/warmup0.05/Nesterov m0.9/WD1e-4/LS0.1, batch128, seed42, compile reduce-overhead. num_params unchanged (4,299,866).

## Execution Environment
- Method: local, `CUDA_VISIBLE_DEVICES=<idle> uv run train.py > run.log 2>&1`, background bash.
- Resources: single idle H20 (pre-check `nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader`; the Σdt=300s budget REQUIRES an uncontended GPU — relaunch on contention).
- Estimated runtime: ~91 epochs, dt ~8ms, Σdt ~300s, wall ~590s (< 600s; alpha is wall-neutral — no op-count change vs EXP-054's 593s). Monitor the 600s wall (recipe is wall-tight, 3 prior breaches EXP-061/065/066) — but this change adds no CPU augmentation cost.
- Log output: `run.log` in project root.
- Tool skill: none (local).

## Abort Criteria
- Loss NaN/inf, or eval test_acc not climbing by ~ep5 (would indicate a broken transform — but a single valid kwarg should not break anything).
- dt elevated to ≥13ms sustained (cudagraph break / contention — NOT expected from a CPU-side aug change; if seen, it is GPU contention): kill, relaunch on a clean idle GPU.
- Dataloader starvation: intra-epoch wall ms/step ≫ dt indicating workers can't keep up (NOT expected — alpha doesn't add ops); if early wall projection ≫ 600s, let it finish only if Σdt=300 stays respected, else note.
- Runtime error constructing `AugMix(alpha=2.0)` (signature mismatch — pre-verified absent): capture traceback, treat as code error (1 fix-retry per execute skill).
- No output / hung > 3 min.

## Verification Protocol

### Verification Procedure
Baseline = **96.45** (from `exp-index.sh baseline`); bar = **96.55**.
1. **Necessary condition 1 — `best_test_acc >= 96.55`**: after exit, `grep -aE "^best_test_acc:" run.log`; parse the float; PASS iff `>= 96.55`. (Stop at first failed necessary condition.)
2. **Necessary condition 2 — clean completion within budget**: `grep -aE "^total_seconds:|^num_epochs:|^num_params:|^training_seconds:" run.log`; confirm summary printed, `training_seconds ≈ 300`, `total_seconds < 600`, `num_params == 4,299,866`, and `grep -ciaE "nan|traceback|error" run.log` == 0. (If total_seconds breaches 600 by a small margin while training_seconds≈300 and the metric is trustworthy → no-improvement per EXP-061/065 precedent, NOT invalid — alpha is compute-free so any breach is base-recipe wall variance.)
3. **Necessary condition 3 — no hard-constraint violation**: `git diff --name-only` == `train.py` only; prepare.py/eval untouched; evaluate() called once/epoch; no new deps (torchvision-native kwarg); seed 42 unchanged; ran uncontended (steady ~8ms dt).
- Verdict: improvement iff all three pass; no-improvement if a necessary condition fails on a valid run; invalid on scope/dep breach; crash if no metrics / cudagraph-break abort.
- Timeout: 12 min wall. Cleanup: `rm run.log` after recording.

### Informational Metrics (Optional)
- `num_epochs`/`num_steps`, `final_test_loss`, `peak_vram_mb`: `grep -aE "^num_epochs:|^num_steps:|^final_test_loss:|^peak_vram_mb:" run.log` — confirm ~91 ep (throughput unchanged; alpha is CPU-side and op-count-neutral). Compare final_test_loss to EXP-054's 0.1968: lower-variance mixing (alpha=2) tends to produce a slightly milder/more-consistent train signal — watch whether loss drops (polish) while top-1 stays flat (the recurring polish-vs-top1 signature) or whether the faithful 3-chain blend genuinely lifts top-1.
