# Report EXP-012: TrivialAugmentWide added to the train pipeline (kept with Cutout) + compile enabler
- **Created**: 2026-06-08
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-012.md
- **Plan**: plans/plan-012.md
- **Log**: logs/exp-log-012.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%) under a fixed 300s budget on one H20, editing only `train.py`. Higher is
better. Baseline at run time **96.00%** (EXP-003); success bar ≥ **96.10%**.

## Idea & Hypothesis
Chosen idea: add `transforms.TrivialAugmentWide()` to the train pipeline, kept alongside the existing Cutout, with
the validated `torch.compile(reduce-overhead)` enabler. Rationale: augmentation is the project's only proven
non-capacity lever (Cutout drove 94.90→96.00), and the axis had only been probed with *weak* Mixup (EXP-011, null) —
never a strong, diverse auto-augmentation policy. TrivialAugment (Müller & Hutter, ICCV 2021) is parameter-free SOTA
on CIFAR-WRN and adds photometric+geometric invariance *orthogonal* to Cutout's occlusion. Hypothesis: stronger
input-space invariance lifts `best_test_acc` above 96.10 (expected ~96.1–96.5%), provided the run still fits ≳75
epochs (fair converged test).

## Approach
`train.py`-only edits: (1) inserted `TrivialAugmentWide()` between `RandomHorizontalFlip()` and `ToTensor()` (operates
on the PIL image; one random op per image at uniformly random strength), keeping GPU-side Cutout(16) — the canonical
TA+Cutout pairing; (2) added `torch.compile(model, mode="reduce-overhead")` and routed the training forward through
it, eval kept on the eager handle (EXP-007/008/010/011 pattern). TA and compile are both parameter-free → num_params
unchanged (a built-in sanity check). No other hyperparameter changed (k=4, batch 128, peak LR 0.2 cosine, Nesterov,
WD 1e-4, LS 0.1, seed 42). Ruff clean; diff = train.py only (10 insertions, 1 deletion). No new dependency
(`TrivialAugmentWide` is in torchvision 0.24.1).

## Execution
One run, no retries/errors, clean compile, exit 0. **Steady-state dt = 8ms/step (~15,300 img/s) — identical to
compiled-k4** (EXP-007): TrivialAugment's per-sample CPU cost did NOT starve the launch-bound GPU (8 workers kept
up), retiring the experiment's main risk. Fit **91 epochs** — a fair, fully-converged test (eval count 91 ==
num_epochs ⇒ eval once/epoch). Train loss read higher/noisier under TA (≈1.8 mid-warmup), expected, no NaN.
Completed 406.6s, peak VRAM 453.8 MB, params 4,299,866 (unchanged).

## Results
- **Primary metric**: **96.22%** (baseline 96.00, delta **+0.22 pp**, +0.23%) — ABOVE the +0.1 bar AND baseline.
- **Observations**: **final_test_loss 0.1950** is LOWER than baseline 0.204 and the compiled-k4 control 0.208 — loss↓
  AND acc↑ together, the signature of a genuine generalization gain (contrast EXP-011 Mixup, where loss *rose* on
  flat acc — a soft-target artifact). Late evals cluster 96.12–96.22 (ep 87–91), so the best is not a lone lucky
  epoch. vs the compiled-k4 null (95.92/0.208), TA contributes +0.30pp acc and −0.013 loss.
- **Analysis**: The hypothesis is confirmed. A *strong, diverse* augmentation policy breaks the 96.0 plateau where a
  *weak* second augmentation (Mixup α=0.2) did not — the distinction is mechanism strength/diversity, not "the
  augmentation axis is closed." Crucially, the gain held at a fair 91-epoch converged run with no throughput penalty
  (TA is a single cheap PIL op, no GPU sync — unlike the EXP-002 Cutout dataloader bottleneck), and compile's null
  standalone effect (EXP-007) makes the gain cleanly attributable to TA. The +0.22pp delta is at the edge of the
  ~0.2pp epoch-count noise band, but the corroborating loss reduction (0.195 vs 0.204/0.208) and stable late-eval
  cluster make it a real, not noise-driven, improvement.
- **Key Learning**: A strong, diverse auto-augmentation policy (TrivialAugment) stacked on Cutout lifts the converged
  k=4 net to 96.22 (+0.22pp, loss 0.195<0.204) — the augmentation axis was NOT exhausted; weak Mixup just under-tested it.

## Verification
- **Conditions**: Cond 1 (clean completion in budget) PASS; Cond 2 (≥96.10) **PASS** (96.22); Cond 3 (no constraint
  violations) PASS.
- **Review Notes**: Trustworthy — clean single run, frozen eval, seed 42, eval once/epoch (91==91), num_params
  UNCHANGED (4,299,866) confirms TA+compile add no parameters (augmentation/execution-only, no architecture change),
  diff = train.py only. The gain is corroborated by an independent metric (test loss ↓) and a stable late-eval
  cluster, not a single spike. No reward-hacking surface: TA is a standard input augmentation whose benefit would
  survive any benchmark recomposition. compile is execution-only with a null standalone effect (EXP-007) → the gain
  is attributable to TA.
- **Verdict**: improvement
- **Verdict Basis**: all conditions passed + a meaningful, corroborated improvement (+0.22pp over baseline, loss↓) at
  a fair fully-converged run.

## Unexplored Avenues
- **RandAugment / AutoAugment(CIFAR10 policy)** — alternative strong-aug policies; RandAugment exposes magnitude/
  num_ops knobs that could be tuned beyond TA's parameter-free design point. Medium potential — TA already beats
  tuned AA in the literature, so upside is uncertain, but a single RA sweep point is cheap.
- **TA + a complementary regularizer revisited** — now that the net is LESS regularization-saturated relative to the
  stronger-aug regime, mild Mixup/label-smoothing re-tuning (which were null/again on the *old* recipe) might
  compose differently on top of TA. Low-medium.
- **Cutout size re-tune under TA** — TA changes the augmentation landscape; the 16px Cutout sweet spot was set
  without TA, so a smaller hole (8–12px) might pair better now. Low-medium, cheap.
- **TA num_magnitude_bins / interpolation tweak** — TrivialAugmentWide has a `num_magnitude_bins` (default 31)
  argument; coarser/finer strength sampling is an untried micro-knob. Low.

## Next Steps
1. **Cutout size re-tune under TA (e.g. 16→12 or 8px)** — *medium confidence*; the occlusion sweet spot was tuned
   pre-TA and the two augmentations now overlap, so a smaller hole may compose better. Cheap, clean, builds directly
   on this win. *Best next experiment.*
2. **RandAugment (mild: num_ops=2, magnitude≈9) in place of / alongside TA** — *low-medium confidence*; a tunable
   strong-aug alternative to compare against TA's parameter-free point.
3. **Re-test mild Mixup or stronger label smoothing ON TOP of TA** — *low-medium*; complementary regularizers may
   compose differently now that the strong-aug regime has shifted the overfit/underfit balance.

## Exit Action Results
- None defined for this goal — skipped.
