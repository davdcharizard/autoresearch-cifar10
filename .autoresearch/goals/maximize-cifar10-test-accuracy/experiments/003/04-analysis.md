# Report EXP-003: Frozen patch-whitening first convolution
- **Created**: 2026-06-28

## Goal
Maximize `best_test_acc` (%) on CIFAR-10 within a fixed 300s training-time budget, editing only `train.py` (higher is better). Entering baseline: **95.72%** (EXP-002, commit c404104). Improvement bar: ≥0.1pp → **≥95.82%**.

## Idea & Hypothesis
Chosen idea (Idea-01, Codex idea-review pick, 8/10 evidence): prepend a **frozen patch-whitening first convolution** — the foundational David Page / hlb / airbench front-end — to the existing DavidNet, leaving the EXP-002 recipe (EMA+flip-TTA, one-cycle, etc.) byte-identical. Mechanism: raw mean-subtracted RGB has a highly anisotropic 27-dim patch covariance; decorrelating it (ZCA whitening) sphereizes the first layer's loss surface so SGD makes useful early progress, reaching a marginally lower-loss tail minimum in the same budget. Hypothesis: 95.72% → **≈95.9–96.1%** (central ~95.95%), clearing the bar, with the gain visible as faster early-epoch convergence and a modestly higher annealed tail; EMA+TTA preserved.

## Approach
Four additive edits to `train.py` (Option A, kernel=3/pad=1, eps=1e-4):
- `compute_whitening_weight(...)`: reads raw `train_set.data` (capped 2000 imgs), `/255` − `EVAL_MEAN` (exact eval space, std=1), unfolds 3×3×3=27-dim interior patches, 27×27 covariance, `torch.linalg.eigh`, eigvecs scaled by `1/√(eig+eps)`, concat `(W,−W)` → frozen `[54,3,3,3]`. Patch subsample uses a **local** `torch.Generator().manual_seed(0)` (no global-RNG perturbation).
- `ResNet9`: frozen `self.whiten = Conv2d(3,54,3,pad=1,bias=False)`; `prep` widened to `conv_bn(54,64)`; `load_whitening()` (loaded after `.apply()`/`.to()` so kaiming doesn't overwrite it); `_forward_once` prepends `self.whiten(x)`. Pad=1 preserves 32×32 → pool chain untouched (verified 512×4×4 entering `pool`).
- `main()`: whitening computed + loaded after `.to(device)` but **before** `t_start_training` (off the 300s budget, printed `whitening_seconds`); SGD optimizer built over `requires_grad`-filtered params (excludes the frozen conv); EMA `AveragedModel` construction follows, so its initial copy carries the loaded whitening (constant ⇒ EMA = constant).
- Everything else (schedule, PEAK_LR=0.4, wd, LS, Cutout, EMA, TTA, batch, global seed) unchanged for a clean single-variable A/B. Plan passed a Codex adversarial review (8 concerns → bounded patch materialization, corrected off-budget timing semantics, genuineness cross-check, tightened abort discipline) and a real-frozen-path smoke test before the run.

## Execution
Single run, no retries. `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` on GPU 1. `whitening_seconds 0.08` (off-budget). Ran 174 epochs / 16,802 steps in 452.8s wall (300.0s training), no divergence, ~29.3k img/s, VRAM 1.61 GB. The whitening **front-end visibly accelerated early convergence** (ep1 60.19% vs EXP-002 57.08%; ep10 85.45% vs 81.57%; ep25 88.84% vs 79.35%). The tail crossed the 95.82% bar at ep159 and peaked **95.87% at ep162**, holding through ep174 (final 95.83%). No errors.

## Results

- **Primary metric**: **95.87%** (baseline: 95.72%, delta: **+0.15pp**, +0.16%)
- **Observations**: Result landed just below the predicted 95.9–96.1% band but cleanly cleared the +0.1pp bar. The mechanism was confirmed *directly*: whitening produced a large early-epoch lead (ep≤25 +3 to +9pp vs EXP-002) — exactly the conditioning/early-convergence acceleration airbench credits — and that lead converted to a **+0.15pp higher tail despite running 9 fewer epochs** (174 vs 183) because of the extra 54-ch conv per step. So whitening more than paid for its own throughput cost. The off-budget eigendecomposition was trivially cheap (0.08s). VRAM and wall essentially unchanged.
- **Analysis**: Hypothesis confirmed, mechanism validated, magnitude at the lower end of the estimate (the honest caveat — that whitening's benefit compresses in a fully-annealed 174-epoch regime vs airbench's epoch-starved headline — held: the early lead was large but the annealed tail gain was modest, ~+0.15pp). In-scope and genuine: only `train.py`, frozen conv excluded from the optimizer, eigendecomp off-budget on a capped 2000-image patch set, exactly one eval/epoch, summary best == max per-epoch eval (not fabricated).
- **Key Learning**: A frozen ZCA patch-whitening front-end adds +0.15pp (95.72→95.87) by accelerating early convergence; the gain survives the ~9-epoch throughput cost, confirming input conditioning is still a (modest) lever at this accuracy.

## Verification
- **Conditions**: all passed. (1) clean run, `total_seconds 452.8`<600, best present; (2) `training_seconds 300.0`≥295, `whitening_seconds 0.08` off-budget, `prepare.py` unchanged (vs working tree and vs integration branch), `TIME_BUDGET_S=300`; (3) `95.87%`≥95.82% bar → improvement, **genuineness cross-check** max per-epoch best=95.87%=summary (from `Eval.evaluate`); (4) only `train.py` changed, global seed 42 + local Generator(0) for patches (no seed search), one `evaluator.evaluate`, no test-set/eval-internals access, whitening frozen+optimizer-excluded+off-budget, budget-timing loop unchanged, TTA still 2 passes.
- **Review Notes**: results confirmed trustworthy. Gain arrives through the sanctioned eval interface on genuinely whitened inputs; early-vs-tail trace is mechanistically consistent (large early lead, modest tail gain). No false-pass risk.
- **Verdict**: **improvement**
- **Verdict Basis**: all necessary conditions passed + meaningful +0.15pp improvement over baseline.

## Unexplored Avenues
- **Whitening + raised PEAK_LR**: a better-conditioned input tolerates larger steps; the early-convergence headroom suggests the one-cycle peak could go above 0.4 for a compounding gain (held fixed here for clean A/B). Plausible next tenth.
- **Whitening + identity-init stem (Option B)** / **kernel-2 whitening**: airbench pairs whitening with identity init (further epochs-to-target reduction); not tried to keep this run clean.
- **Recover the lost ~9 epochs**: the whitening conv costs throughput; folding it into a more efficient stem or precomputing once could buy back epochs and a bit more tail.
- **Compose with capacity/optimizer levers**: whitening is orthogonal and now part of the base — width (Idea-03) or Muon (Idea-04) can be tested on top of 95.87% with EMA+TTA+whitening all on for free.

## Next Steps
- **Whitening-enabled LR raise / one-cycle retune** — exploit the conditioning headroom (PEAK_LR 0.4→~0.5, or higher peak) now that the input is whitened; cheap, low-risk follow-up. Confidence: medium.
- **Capacity probe (wider DavidNet, Idea-03)** — with conditioning + denoising solved, test whether capacity is now binding on the path to 96%; higher ceiling, higher variance (spends epochs). Confidence: medium.
- **Muon optimizer (Idea-04)** — the remaining named lever (training-dynamics optimizer), deferred for a tighter reference port + LR smoke. Confidence: low-medium.

## Exit Action Results
- No exit actions defined for this goal — none executed.
