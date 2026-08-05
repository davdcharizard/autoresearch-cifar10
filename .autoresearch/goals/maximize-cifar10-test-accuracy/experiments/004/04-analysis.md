# Report EXP-004: Identity-initialized (ReZero) layer2 residual block — capacity probe
- **Created**: 2026-06-28

## Goal
Maximize `best_test_acc` (%) on CIFAR-10 within a fixed 300s training-time budget, editing only `train.py` (higher is better). Entering baseline: **95.87%** (EXP-003, commit 6e25456). Improvement bar: ≥0.1pp → **≥95.97%**.

## Idea & Hypothesis
Chosen idea (Codex idea-review pick, evidence 7/impact 8): probe the one major lever untouched across EXP-001→003 — **representational capacity**. The airbench lineage (arXiv:2404.00498) documents the 95→96 step as a bigger net (added conv + residual per block). Add a `Residual(256)` block to `layer2` (the only stage without one), bringing the net from 8 to 10 learnable convs at 8×8 resolution. To realize the chosen "identity-init" cleanly, the residual is **ReZero-gated** (Bachlechner et al. 2020): a learnable scalar `α=0` at init makes the block exact identity, so the deeper net starts bit-equivalent to the proven net and earns capacity gradually as `α` ramps off zero — keeping a live gradient path (∂L/∂α=⟨grad_out, branch_out⟩≠0). This removes the depth+LR confound (no LR retune needed: `PEAK_LR` held at 0.4) and defuses the under-annealing failure mode. Whitening + EMA + flip-TTA stay byte-identical. Hypothesis: 95.87% → **~95.95–96.1%** (central ~96.0%), clearing the bar, *provided* the deeper net at ~150 epochs reaches a lower annealed loss floor than the 8-conv net at 174 epochs; falsifiable on the trajectory (early epochs match EXP-003 within noise due to identity start; tail settles higher).

## Approach
Two additive edits to `train.py`:
- Added a `GatedResidual(nn.Module)` class (`c1=conv_bn(c,c)`, `c2=conv_bn(c,c)`, `self.alpha=nn.Parameter(torch.zeros(1))`, `forward: x + alpha*c2(c1(x))`).
- Appended `GatedResidual(256)` to `self.layer2` (channel- and spatial-preserving, 256→256 @ 8×8, so layer3/pool/fc/whiten untouched).
- Everything else (PEAK_LR=0.4, schedule, wd, LS, Cutout, EMA, TTA gate, batch, seed 42, whitening) unchanged for a clean single-variable A/B.

**Critical design correction (from the plan-phase Codex review):** the originally-planned identity-init via zeroing the new block's final BatchNorm γ is **fatally broken** in this codebase — `conv_bn` ends in ReLU, so a zeroed final BN gives `ReLU(0)=0` with derivative 0, meaning **no gradient ever reaches the block** and it stays identity *forever* (testing "same net, fewer epochs", not capacity). ReZero (a learnable scalar gate) achieves the same identity-at-init property while keeping the gradient path live. The Milestone 1 in-process smoke explicitly verified the fix: `α.grad = 0.0179 ≠ 0` after one backward (the block is trainable), alongside identity-at-init (`allclose(block(h),h)`), correct shapes, the 512×4×4 pool input, learnable params 7,783,169 (exact), and the frozen whiten conv. The plan also hardened scope/leakage verification per the review.

## Execution
Single run, no retries. `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` on GPU 1. Ran 142 epochs / 13,704 steps in 300.0s training (445.2s wall), no divergence, ~26.3k img/s steady, VRAM 1635 MB, whitening 0.46s off-budget. The early trajectory tracked EXP-003 closely (identity-gate start working), the capacity advantage emerged mid-training, and `best_test_acc` reached 96.00% at ep119, holding to ep142. No errors.

## Results

- **Primary metric**: **96.00%** (baseline: 95.87%, delta: **+0.13pp**, +0.14%)
- **Observations**: Landed right at the predicted central estimate (~96.0%) and cleanly cleared the +0.1pp bar. The mechanism was confirmed *on the trajectory*: (1) **identity-gate start preserved early convergence** — ep1 58.70% / ep10 85.19% are within noise of EXP-003's ep1 60.19% / ep10 85.45%, i.e. the ReZero block did not disrupt the early phase (it starts as exact identity), exactly as designed; (2) **the capacity advantage emerged as α ramped off zero** — by ep25 the deeper net was at **92.63% vs EXP-003's 88.84% (+3.79pp)**, ep50 94.00%, and the tail floor settled higher (96.00% vs 95.87%); (3) this held **despite running 32 fewer epochs** (142 vs 174) from the extra 8×8 block's ~11% per-step throughput cost — the capacity gain more than paid for the lost annealing budget. VRAM essentially flat (1635 vs 1614 MB).
- **Analysis**: Hypothesis confirmed, mechanism validated, magnitude at the central estimate. The chosen ReZero realization was decisive — the plan-review correctly identified that the naive BN-zero identity-init would have produced a dead block and a false negative; the gated version delivered both the clean identity start (matching early trajectory) and genuine capacity uptake (mid-training lead). In-scope and genuine: only `train.py` changed (diff limited to the new class + one layer2 line; forward/eval/loop untouched), seeds unchanged (no seed search), exactly one eval/epoch, summary best == max per-epoch eval (from `Eval.evaluate`).
- **Key Learning**: Representational capacity was still a binding lever at 95.87% — one identity-initialized (ReZero) residual block in layer2 adds +0.13pp (95.87→96.00) and the capacity gain outruns the 32-epoch throughput cost, crossing the 96% line documented by airbench.

## Verification
- **Conditions**: all passed. (1) clean run, `total_seconds 445.2`<600, best present; (2) `training_seconds 300.0`≥295, `prepare.py` byte-unchanged (vs worktree and integration branch), diff-content limited to the `GatedResidual` class + the one `layer2` line (forward/eval/loop/HPs untouched); (3) `96.00%`≥95.97% bar → improvement, genuineness cross-check (max per-epoch best=96.00%=summary, from `Eval.evaluate`), one eval/epoch, seeds unchanged (no hacking), no test-set leakage.
- **Review Notes**: results confirmed trustworthy. The gain arrives through the sanctioned eval interface; the early-matches-EXP-003 / mid-training-lead trajectory is mechanistically consistent with an identity-init block that ramps capacity in. No false-pass risk (the dead-block false-negative risk was retired by the gradient smoke).
- **Verdict**: **improvement**
- **Verdict Basis**: all necessary conditions passed + meaningful +0.13pp improvement over baseline.

## Unexplored Avenues
- **Recover the lost 32 epochs**: the new block costs ~11% throughput. A more efficient stem, or folding the gain into a width bump instead of depth (one extra conv at the cheap 8×8 stage), could buy back annealing budget and more tail.
- **More capacity, same recipe**: with capacity now proven binding, a second gated block (e.g. layer1 already has one; add depth to layer3 or widen block2 channels toward airbench96's 384) is the natural next probe — ReZero init makes added depth safe to stack without LR retune.
- **Compose multi-crop TTA (EXP-004 idea-02, deferred)**: airbench96 reaches 96.05% with `tta_level=2` (mirror×{center,±1px}); it is orthogonal eval-side and now stacks on top of the 96.00% capacity result for a plausible next tenth.
- **Whitening-enabled schedule retune, correctly**: EXP-004 idea-03 was dropped for a math error (its knobs raised tail LR). A *correct* shorter-warmup-at-fixed-peak retune remains untried and cheap.
- **Identity-init α schedule / per-channel gate**: a vector α (LayerScale) instead of a scalar could let the block ramp channels independently — a richer capacity-uptake variant.

## Next Steps
- **Stack more capacity (depth or width) with ReZero init** — capacity is confirmed binding; add a second gated block or widen block2 toward airbench96, keeping LR fixed via the gate. Confidence: medium-high (direct extension of a just-validated lever, but each add spends more epochs).
- **Multi-crop TTA on top of 96.00%** — cheap, orthogonal eval-side lever (airbench's remaining TTA gain beyond flip); compose with the new architecture. Confidence: medium.
- **Correct schedule retune (shorter warmup at fixed peak)** — the cheap probe of remaining schedule headroom, fixing idea-03's mechanism error. Confidence: low-medium.

## Exit Action Results
- No exit actions defined for this goal — none executed.
