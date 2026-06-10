# Report EXP-028: SiLU/Swish activation (ReLU → SiLU everywhere)
- **Created**: 2026-06-09
- **Goal**: goals/improve-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-028.md
- **Plan**: plans/plan-028.md
- **Log**: logs/exp-log-028.md

## Goal
Maximize CIFAR-10 `best_test_acc` (%, higher-is-better) within a fixed 300s training-compute budget on a single H20, editing only `train.py`. Baseline = **96.22** (EXP-012); pass bar = **96.32** (+0.1pp). EXP-028 tests whether a smooth activation lifts top-1.

## Idea & Hypothesis
Chosen idea: **SiLU/Swish activation** (Ramachandran et al. 2017) — replace ReLU with SiLU = `x·σ(x)` at all three activation sites. Selected as the single largest untried orthogonal lever: the activation function had never been changed (always `F.relu`), and it was explicitly flagged in the EXP-009 goal-learning as an orthogonal axis. Unlike the closed axes, a smooth activation is NOT a regularizer (so not blocked by the convergence-bound recipe), NOT a capacity change (so not the epoch wall, *provided SiLU fuses*), and NOT convergence-polish (it changes the representation, which can move top-1). Hypothesis: smoothing the optimization landscape and removing the dead-ReLU zero region lifts `best_test_acc` above 96.32 at unchanged ~91 epochs / dt ~8ms.

## Approach
Three one-token edits in `BasicBlock`/stem: `F.relu` → `F.silu` at train.py L89 (pre-residual), L92 (post-residual), L127 (stem). No config or init changes (kaiming/ReLU-gain left as-is — BN after every conv absorbs the slight mismatch). Smoke test confirmed 0 relu / 3 silu, params unchanged at 4,299,866 (activation is param-free), forward (2,3,32,32)→(2,10), train.py-only diff. No deviations from plan.

## Execution
Single run, no retries. `CUDA_VISIBLE_DEVICES=0 uv run train.py` on GPU 0, exited 0 in 396.6s. Clean compile, no NaN/Traceback, loss decreased normally. A Monitor watch surfaced the summary on completion. Notable early signal: dt printed steadily at **9ms** (vs baseline 8ms) from step 1 — SiLU did not fully fuse away.

## Results
- **Primary metric**: 95.98% (baseline: 96.22, delta: **−0.24pp**, −0.25%)
- **Observations**: num_epochs 88, num_steps 34254 (baseline ~91 / ~35500), mean dt ≈ 9ms (681/685 sampled lines), peak_vram 511.8 MB (slightly above baseline ~467 — SiLU's σ(x) intermediate). **final_test_loss 0.1960 ≈ baseline 0.195 — flat.**
- **Analysis**: Hypothesis not supported. SiLU neither improved top-1 (−0.24pp) nor loss (flat at 0.196). The −0.24pp is within the ~0.2pp noise floor plus a small (~3-epoch) penalty: SiLU cost ~1ms/step (~12%), dropping epochs 91→88. This is a MILDLY confounded but roughly fair test (88 ≫ 85 confound floor) — even crediting the lost epochs, there is no positive signal, and the loss being flat (not lower) means a smooth activation did not even help convergence quality here. The result fits the broader plateau picture: on an already well-tuned shallow ResNet-20-style net, the optimization landscape is not ReLU-limited; the dead-ReLU concern (which motivates smooth activations on deep/hard-to-train nets) does not bind here because BN + warmup + the tuned recipe already train cleanly. Throughput-wise it adds a small but real cost — a secondary reason not to keep it.
- **Key Learning**: Swapping ReLU→SiLU does not help this well-tuned shallow CIFAR net (flat loss, −0.24pp top-1) and costs ~1ms/step; the recipe is not activation-limited, so smooth-activation gains (a deep/hard-to-optimize phenomenon) don't transfer.

## Verification
- **Conditions**: Cond 1 (best_test_acc ≥ 96.32) FAILED — 95.98 < 96.32; Cond 2/3 recorded informationally and PASSED (clean completion 396.6s < 600, train.py-only, params 4,299,866 unchanged, 88 evals for 88 epochs, no new deps, seed 42).
- **Review Notes**: Results trustworthy. Mild throughput confound (88 vs 91 ep) noted but above the ~85 floor → roughly fair; the flat loss independently confirms no convergence benefit, so the negative is not merely an epoch artifact. No integrity concerns (single-forward eval unchanged, allowed intervention class).
- **Verdict**: no-improvement
- **Verdict Basis**: verification condition failure (primary metric below bar); valid, roughly-fair negative.

## Unexplored Avenues
- **Mish** (`x·tanh(softplus(x))`, EXP-029 candidate): marginally stronger CIFAR literature than SiLU, BUT ~2× the pointwise cost — SiLU already cost ~1ms/step and didn't fuse, so Mish would likely drop MORE epochs (worse confound) for the same null mechanism. LOW confidence; the SiLU result (flat loss, not landscape-limited) predicts Mish nulls too. Only worth it if a cheaper/fused activation path existed.
- **GELU / hard-swish**: hard-swish is cheaper (piecewise-linear, fuses better) and approximates SiLU — could test the smooth-activation hypothesis at lower dt cost, but the flat-loss SiLU result suggests the mechanism itself doesn't help here, so low value.
- The activation axis is effectively **closed**: the representative smooth activation (SiLU) is null on both top-1 and loss and carries a throughput cost.

## Next Steps
- **Per-channel input std-normalization** (low confidence): the last untouched cheap input-pipeline scalar; assessed as expected null/mild-regression (train/test std mismatch, eval frozen at std=(1,1,1)). A quick closer to formally bracket the normalization axis.
- **Accept the plateau / honest ceiling** (high confidence): ~21 axes now closed (scalar knobs, aug family, regularizers, capacity/epoch-wall, batch, weight-averaging/polish, downsampling both sides, and now activation). 96.22 is at/near the k=4 / 300s ceiling. Per NEVER-STOP, continue probing, but expectations should be calibrated to closing remaining minor axes rather than clearing +0.1pp.
- **Radical re-architecture at fixed compute** (low confidence): e.g., a fundamentally different stem or a depth/width re-balance that *reduces* dt to buy epochs — but width/depth is largely bracketed and the budget is compute-tight. Only pursue if a throughput-POSITIVE structural idea emerges (since dt is the binding constraint).

## Exit Action Results
- None defined.
