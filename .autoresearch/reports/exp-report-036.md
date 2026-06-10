# Report EXP-036: Periodic / sparse SAM (Sharpness-Aware Minimization, every 5th step, ρ=0.05)

- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-036.md
- **Plan**: plans/plan-036.md
- **Log**: logs/exp-log-036.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) within the fixed 300s budget, editing only `train.py`. Baseline = **96.22%** (EXP-012, commit 6c417a4); pass bar = **96.32%**.

## Idea & Hypothesis
The model is generalization-bound at fixed k=4 capacity with every compute-neutral optimization-polish and regularizer lever closed (~27 axes). SAM (Foret et al. 2021) is the canonical fixed-architecture *generalization* lever — minimize the worst-case loss in an ε-ball → flatter minima → better top-1 — and is mechanistically untouched here (it changes the minimized objective, not gradient conditioning/averaging/schedule). Its blocker is the 2×-per-step cost (compute/epoch wall). Hypothesis: applying SAM (ρ=0.05) **sparsely, every 5th step** (plain SGD otherwise) keeps epochs ≥~75 (≈ the ~77-ep convergence point, using the project's ~15% epoch slack) so the flat-minima benefit can lift best_test_acc above the 96.22 plateau toward 96.32 in a near-throughput-fair test.

## Approach
Three edits to baseline `train.py`, no architecture/data/eval change: constants `SAM_RHO=0.05`, `SAM_EVERY=5`; hoisted `sam_params` list; a two-pass SAM step gated by `step % 5 == 0`. On a SAM step: first forward-backward → clean grad (logged loss); under `no_grad`, global grad-norm → ascent `w += ρ·g/‖g‖` via vectorized `torch._foreach_mul/_add_` (per the EXP-031 throughput lesson); second forward-backward at the perturbed point; restore `w -= e` via `_foreach_sub_`; `optimizer.step()` uses the SAM gradient (clean gradient on the other 4/5 steps). torch.compile(reduce-overhead) kept. Smoke test confirmed the compiled two-pass step runs without CUDA-graph/dynamo errors (no fallback needed), finite grad-norms, params 4,299,866.

## Execution
Single run, exit 0 in 393.3s wall. No NaN/CUDA/cudagraph errors. The reduce-overhead two-pass SAM step worked at scale. Realized **mean dt = 10.16ms (1.27× the 8ms baseline)** → **76 epochs** (matching the plan's ~76-ep prediction; the mid-run 8–9ms readings at printed SAM steps were misleading — the SAM second forward-backward does cost real GPU time). No experimental adjustments.

## Results

- **Primary metric**: best_test_acc = **95.89%** @ ep72 (baseline 96.22, delta **−0.33pp**; −0.43pp vs bar). Bar NOT cleared; below baseline.
- **Converged, not grossly underfit**: final_test_loss **0.1969 ≈ baseline 0.195**, and the tail is dead-flat (ep71–76 all 95.84–95.89). So the 76-ep SAM run reached baseline-quality loss and plateaued — the −0.33pp gap is NOT a severe under-training artifact (contrast EXP-022's underfit loss 0.224). SAM's flat-minima mechanism simply did not yield a top-1 gain here.
- **Disentangling SAM-effect vs epoch-cost**: SAM cost ~16% of epochs (76 vs 91). A plain-SGD run at 76 ep would likely score ~96.0–96.1 (a few hundredths under baseline's 91-ep 96.22). SAM landed 95.89 — at or slightly *below* that hypothetical, i.e. SAM's flat-minima benefit did not even fully offset its own epoch cost, let alone beat the plateau. The flat tail confirms more epochs wouldn't have rescued it much at this config.
- **Analysis / fit to trajectory**: This is the most direct test yet of the diagnosed binding constraint (generalization at fixed capacity), via the single most-cited generalization optimizer — and it failed to move the plateau. Two project patterns explain it: (1) the **compute/epoch-wall** High-Importance insight — SAM's irreducible 2× cost (even sparse → 1.27×) buys fewer epochs, and on this short budget that erases any benefit (cf. k=5/k=6/BlurPool); (2) the **"deep/long-schedule tricks don't transfer to shallow short-budget CIFAR"** Medium insight — SAM's literature gains (+0.3–1.0pp) come at 100–200+ epoch schedules on deeper nets; at ~76 ep on a 9-block net already trained cleanly with BN+warmup+tuned-recipe, the sharpness regularization doesn't bind. Reinforces the firmly-established 96.22 k=4/300s plateau.
- **Key Learning**: Sparse SAM (ρ=0.05, every 5th step) regressed −0.33pp to 95.89 at 76 ep with near-baseline loss (0.197, converged) — even the most on-mechanism *generalization* optimizer can't beat the plateau, because its irreducible ~1.27× cost (→16% fewer epochs) outweighs a flat-minima benefit that doesn't transfer to this shallow net at 300s.

## Verification
- **Conditions**: Cond 1 (≥96.32) **FAILED** — 95.89 < 96.32 (−0.33 vs baseline). Cond 2 (clean, <600s, 0 errors) passed (393.3s). Cond 3 (only train.py; params 4,299,866; eval-count 76 == epochs; core torch; seed 42) passed.
- **Review Notes**: Trustworthy. SAM fired correctly (smoke + scale), no compile fallback used. Fairness gate met (76 ep ≥ ~75; loss ≈ baseline → converged), with the noted caveat of 16% fewer epochs than baseline — a mild compute component, but the flat converged tail shows the result is near its ceiling for this config, not an artifact of stopping early.
- **Verdict**: no-improvement
- **Verdict Basis**: Valid, trustworthy, throughput-fair-ish result; primary condition (clear the bar) failed and the metric is below baseline; no constraint violated.

## Unexplored Avenues
- **Denser / full per-step SAM** — would maximize the flat-minima signal but at ~2× cost → ~45 ep (the EXP-002-style compute wall, cf. compiled-k5 41 ep → 94.21). The High-Importance compute-wall insight predicts a worse compute-confounded regression; NOT worth running.
- **ASAM (adaptive SAM) or a larger ρ** — adaptive radius can help where fixed ρ is mis-scaled, but it shares the same 2× cost / epoch-wall problem and the same shallow-net transfer doubt; LOW confidence.
- **SAM only in the converged tail (last ~20% of budget)** — apply SAM after the model has converged under cheap SGD, to flatten the final minimum at minimal epoch cost. Plausible but mirrors the cooldown/SWA tail-intervention family that has consistently matched-not-beaten the plateau (EXP-019/020/033/034/035); LOW confidence.

## Next Steps
- **Close the SAM / sharpness-aware sub-axis at this budget.** Sparse SAM (the cheapest fair form) regressed; denser SAM hits the compute wall harder. The optimizer-objective axis joins the closed set. Confidence: high that no SAM variant clears +0.1 at 300s/k=4.
- **The plateau is now extremely well-confirmed (~28 axes, including the most direct generalization lever).** Treat 96.22 as the firm k=4/300s ceiling. Confidence: high.
- **Remaining untried levers are all low-confidence and mostly compute-walled**: cheap asymmetric capacity (extra layer3 block at 8×8 — still adds FLOPs → epoch wall risk per the High insight), or the compute-neutral input std-normalization (brainstorm-036 candidate 3 — safe but expected within-noise). If the loop continues, input std-normalization is the lowest-risk next probe (compute-neutral, untried, insight-endorsed) even though its ceiling is low. Confidence: low it clears the bar.

## Exit Action Results
<!-- No exit actions defined for this goal. -->
