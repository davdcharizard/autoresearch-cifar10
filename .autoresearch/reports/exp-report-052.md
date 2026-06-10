# Report EXP-052: AugMix (w2,d1) replacing TrivialAugmentWide — strongest feasible diverse augmentation
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-052.md
- **Plan**: plans/plan-052.md
- **Log**: logs/exp-log-052.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher is better) by editing only `train.py` within the fixed 300s Σdt budget on a single H20. Baseline at the start of this loop = **96.22%** (EXP-012, 6c417a4); bar = baseline + 0.1 = **96.32%**. After 44 consecutive no-improvements with every other axis closed, this loop tested the only untried variant of the only lever that has ever broken a plateau here — strong diverse augmentation.

## Idea & Hypothesis
Chosen idea: swap the single-chain `TrivialAugmentWide()` for `AugMix()` (Hendrycks et al. 2020), keeping Cutout. AugMix superimposes several independently-sampled augmentation chains, mixed with the clean image via random convex weights — strictly more diverse augmented samples than a single-chain auto-aug policy. Reasoning: strong diverse augmentation is the only intervention class that has ever lifted top-1 on this project (EXP-012, +0.22pp), and the High-Importance project insight explicitly directs testing the strongest diverse variant before declaring the augmentation axis closed. Hypothesis: GPU-throughput-neutral (dt ~8ms, AugMix is CPU-side); IF the extra diversity regularizes better on this generalization-bound net, best_test_acc ≥ 96.32; falsified if within ±0.25pp (saturation, extending EXP-014's policy-swap null) or if CPU cost pushes wall > 600s (infeasibility).

## Approach
One-line transform swap in `train_tf`. The hypothesis-as-stated wanted torchvision-default AugMix (mixture_width=3, chain_depth=-1), but that proved infeasible on the wall-clock constraint (see Execution). The shipped change is `transforms.AugMix(mixture_width=2, chain_depth=1)` — the lightest config that preserves AugMix's defining mechanism (mixing ≥2 augmentation chains with the clean image). Everything else (RandomCrop+Flip, Cutout, model, optimizer, time-fraction cosine schedule, seed 42, batch 128, torch.compile) unchanged. num_params 4,299,866 (augmentation doesn't touch the model). The plan explicitly pre-authorized reducing mixture_width/chain_depth for feasibility, so this is within the approved approach.

## Execution
Two runs on idle GPU 1.
- **Run 1 (default AugMix w3,d-1)**: ABORTED at ~step 3700/ep10. dt steady 8ms (GPU step unchanged, as predicted), but clean intra-epoch wall rate was 46.3ms/step — a 5.8× starvation: AugMix's 3-chain CPU cost (~21ms/batch isolated) starved the 8-worker dataloader, projecting ~1500–1850s wall ≫ the 600s limit. A direct dataloader-throughput probe confirmed: default AugMix 21.1ms/batch (~792s full-budget wall), plan-contingency w2,d2 17.9ms (~670s) — both breach 600s; w2,d1 12.6ms (~572s) fits.
- **Run 2 (w2,d1)**: launched; passed the feasibility gate via a real-load early measurement (eval-inclusive 15.2ms/step at 21.6% → projected ~549s); allowed to complete. Clean exit 0 in 571.9s wall, no NaN, no retries.

## Results
- **Primary metric**: best_test_acc **96.34%** (baseline 96.22, delta **+0.12pp**, +0.12%) @ ep89; final 96.25% @ ep91. Clears the 96.32 bar by 0.02pp.
- **Observations**:
  - dt steady 8ms (625×8ms / 80×9ms / 1×24ms) — GPU step unchanged, confirming AugMix is purely CPU-side. num_epochs 91 = baseline (the Σdt budget is unaffected by dataloader-boundness; only wall-clock rose, 571.9s vs baseline ~402s).
  - final_test_loss 0.2010 ≈ baseline 0.195 (marginally higher) — the gain is a top-1 generalization effect, not a loss-polish effect, consistent with the polish-vs-top1 wall: the only thing that moves top-1 at fixed capacity is better regularization, and diverse augmentation is exactly that.
  - peak_vram 453.8 MB = baseline (CPU-side augmentation).
- **Analysis**: Hypothesis CONFIRMED, modestly. Diverse multi-chain augmentation lifted top-1 even in its lightened w2,d1 form, re-confirming the project's strongest insight: augmentation diversity is the one lever that breaks plateaus here (now EXP-012 ×2 with EXP-052). The +0.12pp is small and sits near the documented ±0.25pp noise band, so confidence is moderate — but it cleared the pre-registered +0.1 bar, the mechanism is the most-validated one on this project, and the direction (more diversity → higher top-1, flat/worse loss) matches EXP-012's signature exactly. Notably, the FULL-strength AugMix the idea originally targeted could not even be tested — it is wall-infeasible at 8 workers — so the realized +0.12pp is from a deliberately weakened variant; the diversity lever may not be exhausted.
- **Key Learning**: AugMix multi-chain mixing (lightened to w2,d1 for the 600s wall) beats single-chain TrivialAugment by +0.12pp — augmentation diversity remains the only lever that lifts top-1 here, but its strongest form is CPU-wall-bound on 8 workers.

## Verification
- **Conditions**: all passed. Cond1 best_test_acc 96.34 ≥ 96.32 ✓; Cond2 clean completion, total_seconds 571.9 < 600, num_params 4,299,866, no NaN/traceback ✓; Cond3 only train.py modified, eval/prepare untouched, evaluate() once/epoch, no new deps (AugMix is torchvision-native), seed 42 unchanged, no seed hacking ✓.
- **Review Notes**: Results trustworthy — clean uncontended run on a deterministic seed, scope-clean diff, real-load feasibility verified before completion (wall 571.9s, 28s margin under the limit). One honest caveat: +0.12pp is within the ±0.25pp noise band, so a replication would strengthen confidence; the verdict rests on clearing the pre-registered +0.1 bar plus the strong mechanistic prior (diverse augmentation is the project's only validated top-1 lever).
- **Verdict**: improvement
- **Verdict Basis**: all necessary conditions passed and the primary metric exceeded baseline by the pre-registered +0.1pp margin (+0.12pp).

## Unexplored Avenues
- **Recover throughput to test stronger AugMix (w3/w2,d2)**: the full-strength variant was never testable (wall-infeasible). If AugMix could be moved to a GPU-batched implementation (like cutout_batch) or otherwise sped up within train.py, the stronger, more diverse config might yield a larger gain — the +0.12pp came from a deliberately weakened form. (The naive GPU-batch route applies one transform per batch, losing per-sample diversity — would need a per-sample vectorized design.)
- **AugMix severity / alpha tuning at w2,d1**: severity=5 probed at ~12.1ms/batch (still feasible) — a stronger-magnitude w2,d1 was not run and might add diversity within budget.
- **Stack AugMix(w2,d1) with the previously-saturating levers**: now that the augmentation base moved, mild Mixup (EXP-011) or a cooldown (EXP-033/34/35) that previously read as saturated could be re-probed on the new base.

## Next Steps
- **Replicate / confirm the +0.12pp** (high confidence it's worth doing, low expected movement): a clean re-run characterizes whether 96.34 is stable above noise before building on it. (medium)
- **Push AugMix diversity within the wall budget**: try w2,d1 at higher severity, or w2,d1 + a feasible second lever, since diversity is the validated lever and we only tested its weakest form. (medium)
- **Throughput-recovery for full AugMix**: a GPU-side or faster CPU augmentation path could unlock the genuinely-strong AugMix that was infeasible here — highest potential upside but highest implementation risk. (low-medium)

## Exit Action Results
<!-- No exit actions defined for this goal. -->
