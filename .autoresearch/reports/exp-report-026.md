# Report EXP-026: Bag-of-Tricks free convergence bundle (zero-init residual γ + no-bias-decay)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-026.md
- **Plan**: plans/plan-026.md
- **Log**: logs/exp-log-026.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%), higher-is-better, editing only `train.py` within a fixed 300s budget on one H20. Baseline = **96.22%** (EXP-012, commit 6c417a4); bar = **96.32%** (+0.1pp).

## Idea & Hypothesis
Chosen idea: the two FREE (compute-neutral, param-neutral) Bag-of-Tricks levers (He et al. 2019) — (a) zero-init residual γ: zero the last BN's γ in each BasicBlock so each residual branch outputs 0 at init and the block starts as identity (cleaner early gradient flow); (b) no-bias-decay: apply weight decay only to conv/linear weights, not to BN γ/β or biases. Selected because, after EXP-025 closed batch size, every scalar knob is bracketed and every compute-adding or regularizer-adding change has failed — leaving compute-NEUTRAL convergence-quality levers as the only class with a defensible positive mechanism and zero epoch-wall risk. Hypothesis: the bundle improves early optimization / effective regularization of the convergence-bound recipe and lifts best_test_acc above 96.32 at unchanged ~91 epochs; falsifiable as a within-±0.2pp null if the tricks are too marginal on this shallow net.

## Approach
Two edits in `train.py`, both compute/param-neutral. (1) In `ResNet.__init__` after `self.apply(self._weights_init)`, a loop sets `init.zeros_(m.bn2.weight)` for every `BasicBlock`. (2) The optimizer was split into two SGD param groups — `weight_decay=1e-4` for ndim≥2 (conv/linear weights, 23 tensors), `0.0` for ndim≤1 (BN γ/β + fc bias, 45 tensors). Smoke test confirmed params 4,299,866 unchanged, all 9 bn2.γ zeroed, both groups covering all params. The training loop's LR update/readout are transparent to the two-group optimizer. Scope = train.py only; no new deps. No deviations from plan.

## Execution
Single run, no retries. Clean startup, clean compile, no NaN, no Traceback. Completed **93 epochs** / 35,913 steps at dt ~8ms in 300s compute, total 404.8s, peak VRAM 453.8MB. The dt/epoch/VRAM all match baseline — compute-neutrality confirmed.

## Results
- **Primary metric**: best_test_acc = **96.18%** (baseline: 96.22, delta: **−0.04pp**, −0.04%).
- **Observations**: A clean WITHIN-NOISE NULL. The −0.04pp delta is far inside the project's ~0.2pp noise floor (goal-learnings High), and the run is a fair throughput-neutral test (dt 8ms, 93 ep ≈ baseline 91, params/VRAM unchanged) — no epoch-wall or update-collapse confound, unlike the compute-adding (EXP-004/009/015/024) and batch (EXP-025) failures. Notably **final_test_loss fell 0.195→0.1899** — a small but real loss/calibration improvement that did NOT convert to top-1 accuracy.
- **Analysis**: Both free tricks are mechanistically real but marginal here. Zero-init residual γ's documented benefit is depth-driven (it eases signal propagation in deep nets); ResNet-20 has only 9 residual blocks and already trains stably (warmup + BN + projection shortcuts), so identity-init buys little. No-bias-decay's effect is bounded by the WD magnitude, which is already tiny (1e-4), so removing it from the few BN/bias params barely changes regularization. The slight loss drop without a top-1 gain echoes the SWA signature (EXP-019/020: loss↓/flatness↑ but top-1 flat) — the recipe's top-1 is already at its optimum for this capacity, and convergence-quality polish moves loss/calibration, not accuracy. This is now the 4th distinct compute-neutral "polish" lever (after EMA, SWA-floor sweep, LS-down) to land at/below the cosine-to-0 baseline on top-1.
- **Key Learning**: The two free Bag-of-Tricks levers (zero-γ + no-bias-decay) are a clean within-noise NULL on this shallow ResNet-20 (96.18, −0.04pp; loss 0.195→0.190) — zero-γ needs depth and WD is too small for no-bias-decay to bite; convergence-quality polish moves loss, not top-1.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) FAILED (96.18); Conds 2–3 skipped per protocol (would pass — clean 404.8s run, train.py-only, params 4,299,866, eval-count 93 == epochs, no new deps, seed intact).
- **Review Notes**: Trustworthy — clean compute-neutral run, scope/params intact, intervention within the allowed class (init + optimizer config are explicitly fair game), no integrity issue. A genuine within-noise null, NOT a confound.
- **Verdict**: no-improvement
- **Verdict Basis**: necessary condition failure (primary metric below bar). Valid result, no constraint violation.

## Unexplored Avenues
- **Ablate the two tricks separately**: with a bundled null there's no value in ablating — neither moved top-1 and the bundle is already at baseline. Low value.
- **Zero-init γ on a DEEPER net**: zero-γ's benefit scales with depth, but depth/width past k=4 / ResNet-20 hits the epoch wall here (EXP-004/009), so it can't be fairly combined at this budget. Closed by the same wall.
- **Other Bag-of-Tricks levers**: the remaining ones are compute-ADDING (ResNet-B/C/D downsample tweaks ≈ BlurPool-class) or already-failed (mixup, large-batch) — all closed.

## Next Steps
- **Per-channel input std-normalization** (std=(1,1,1)→true CIFAR std) — the last untried compute-neutral scalar probe; expected BN-absorbed null, cleanly closes the input-normalization axis with zero confound risk. (medium confidence clean null; low confidence gain.)
- After the std-norm probe, the search is genuinely exhausted: ~19 axes closed — all scalar knobs bracketed (LR, Cutout, LS, WD, batch), every compute-adding change hits the epoch wall, every batch increase is compute-bound, every added regularizer under-fits, and now every compute-neutral convergence-polish lever (EMA, SWA, LS-down, Bag-of-Tricks) lands at/below baseline on top-1 (moves loss, not accuracy). The honest scientific conclusion is the **96.22 plateau is the ceiling for k=4 ResNet-20 at 300s/H20** (capacity- and convergence-bound at this budget). (high confidence the plateau is real.)
- Any further gain requires relaxing a fixed constraint (larger time budget, or a fundamentally more compute-efficient architecture that stays launch-bound while improving generalization) — outside the goal's hard constraints.

## Exit Action Results
<!-- No exit actions defined for this goal. -->
