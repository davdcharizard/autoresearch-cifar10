# Brainstorm EXP-002
**Created**: 2026-06-28

<!-- Goal/metric/constraints in goals/maximize-cifar10-test-accuracy/01-definition.md; baseline (95.22%, after EXP-001) in 04-results.tsv. -->

## Web Search & Literature Review

- **Fast-CIFAR knowledge note** (`knowledge/references/fast-cifar10-recipes.md`): DavidNet→hlb→airbench lineage; whitening front-end, GELU, flip-TTA, one-cycle. Reused from loop 1.
- **SWA/EMA weight averaging** (Izmailov et al., arXiv:1803.05407; PyTorch `torch.optim.swa_utils.AveragedModel` + `get_ema_multi_avg_fn`, `use_buffers=True`): averaging the SGD trajectory lands in a flatter, better-generalizing basin — near-free under one-cycle. In-torch, no new dep.
- **airbench whitening conv** (`init_whiten`, arXiv:2404.00498) + **flip-TTA** (`infer_mirror`): the documented accelerators toward 95–96%.
- **WideResNet** (Zagoruyko & Komodakis, arXiv:1605.07146): width > depth for CIFAR residual nets — widening converts compute to accuracy efficiently.
- **Bag of Tricks** (He et al., arXiv:1812.01187): label smoothing 0.1, TTA as standard recipe components.

Full developed proposals: `proposals/idea-01.md` (whitening), `idea-02.md` (EMA+TTA), `idea-03.md` (wider net).

## Experimental History Review

- **BASE (1f69af5): 91.57%** — CIFAR ResNet-20 (unmodified).
- **EXP-001 (26fdc83): 95.22%** [improvement, +3.65pp] — DavidNet/ResNet-9 + time-based one-cycle, bf16+channels_last, Cutout + label-smoothing 0.2, batch 512. Validated base recipe. Key learning: schedule-completion + wide-shallow net is worth +3.65pp; gains arrive in the low-LR tail; VRAM 1.6/98 GB (capacity-spare); fit ~192 epochs.
- **Untried levers** (from EXP-001 § Next Steps / learnings): whitening front-end, flip-TTA, weight EMA, wider/deeper net, recipe micro-tuning.
- No failed approaches recorded yet.

## Diagnose What Limits the Objective

We have a working DavidNet at 95.22% (airbench-95 territory) with documented headroom to ~96%. The remaining gap is **not** under-annealing (fixed) — it is some mix of: (a) **generalization/robustness** of the evaluated iterate (the raw one-cycle endpoint is a noisy sample; predictions are single-view), (b) **model capacity** (only 1.6/98 GB VRAM, ~192 epochs of a 24-epoch-designed net — likely under-capacity for the compute), and (c) absence of airbench's **whitening** convergence accelerator. Each is a distinct, evidence-backed angle. Since architectural/eval changes plausibly beat tuning exposed params, this is **not** a sweep.

## Collected Ideas

- (eval booster) Flip-TTA inside `forward` (eval-gated) — free ~+0.2–0.4pp.
- (optimization booster) Weight EMA (Polyak/SWA) evaluated each epoch — ~+0.2–0.5pp.
- (architecture front-end) Frozen whitening initial conv (airbench) on DavidNet.
- (activation) ReLU→GELU.
- (capacity) Widen DavidNet 1.25–1.75× (WideResNet); trade excess epochs for capacity.
- (capacity) Add the missing residual block to stage 2 (light depth).
- (recipe) label smoothing 0.2→0.1; PCT_START / peak-LR micro-tune.
- (moonshot) Full airbench94/96 ConvGroup rewrite (whitening + GELU + residual ConvGroups + TTA).

## Combinations

- **EMA + flip-TTA** (idea-02): orthogonal — EMA picks better *weights*, TTA reads them better; near-additive, both cheap. The strongest cheap combo.
- **Whitening + GELU** (idea-01): airbench's two front-end accelerators together.
- **Wider net + (later) EMA/TTA**: capacity bump now; boosters compose on top in a future loop.

## Candidate Ideas

### 1. Weight EMA + flip-TTA on DavidNet
**Summary**: Keep EXP-001's net + recipe byte-for-byte; add two orthogonal, well-evidenced boosters that don't touch training dynamics. (a) **Weight EMA** via `torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.998), use_buffers=True)` — updated each step after 15% warmup, evaluated each epoch in place of the raw iterate (BN buffers EMA-averaged inline, no `update_bn` pass). (b) **Flip-TTA** inside `forward`, gated on `not self.training`: average logits of `x` and `x.flip(-1)`. Both reachable through the frozen eval (which calls `model(inputs)`); exactly one `evaluator.evaluate`/epoch preserved.

**What it targets**: The generalization/robustness gap of the evaluated iterate — EMA variance-reduces the noisy one-cycle endpoint into a flatter basin; TTA ensembles two correlated views. Cite EXP-001 §Next-Steps (named these as cheap levers).

**Reasoning**: SWA/EMA (arXiv:1803.05407) + mirror-TTA (airbench) are standard, near-additive fast-CIFAR boosters; `AveragedModel` is in torch 2.9 (no new dep) and verified compatible with the frozen `evaluate(model, device)` interface (`.eval()` propagates to `self.module`).

**Sources**: `proposals/idea-02.md`; arXiv:1803.05407; PyTorch swa_utils docs; airbench `infer_mirror`; EXP-001 `04-analysis.md`.

**Estimated Effort**: low (~20 lines, no architecture/schedule/dep change).

**Risk Assessment**: Main risk — EMA decay mistuned for the one-cycle tail could make EMA-eval lag the raw 95.22% endpoint and cap best_acc below baseline (we eval EMA, not raw, after warmup). Mitigation: decay 0.998 (~5-epoch horizon) sized to the anneal; TTA-only fallback (cannot regress weights). Secondary: doubled-TTA eval wall-clock (EXP-001 had ~150s headroom under the 600s cap; monitor `wall:`). Expected **~95.5–95.9%** (central ~95.7%).

### 2. Frozen whitening initial conv (+ optional GELU) on DavidNet
**Summary**: Prepend a fixed (frozen) whitening conv (eigendecomposition of ~5000 normalized 2×2 training patches via `torch.linalg.eigh`, 3→24 ch, `cat(scaled,−scaled)`) as the first layer; feed into the existing prep (3→24 input); use `AdaptiveMaxPool2d(1)` to absorb the 32→31 spatial shift; optionally ReLU→GELU. Everything else unchanged. Init AFTER model construction (so kaiming doesn't clobber it) and exclude the frozen weight from the optimizer. Whitening patches computed in the exact frozen-eval normalization space.

**What it targets**: Convergence speed — whitening removes low-level decorrelation work so the same budget does more high-level learning. airbench's documented load-bearing accelerator.

**Reasoning**: Whitening is the key ingredient behind airbench's 94–96%; EXP-001 reached 95.22% on the same net class *without* it. Honest caveat (from the proposal): the whitening speedup is largest in the *epoch-starved* regime; at ~192 epochs the base may be partly saturated, so the gain may be only a few tenths.

**Sources**: `proposals/idea-01.md`; arXiv:2404.00498; hlb-CIFAR10; EXP-001 `proposals/idea-03.md` (verbatim init_whiten); knowledge note.

**Estimated Effort**: low–medium (additive front-end; dimension/freeze/init-order bookkeeping).

**Risk Assessment**: Spatial pool break (mitigated by AdaptiveMaxPool2d), init-overwrite, normalization desync (silent accuracy cap — patches must be in eval space), and the honest soft spot: benefit may wash out at 192 epochs (real chance of a noise-tie). Expected **~95.3–95.6%** (+0.1–0.4pp), with non-trivial chance of landing within noise.

### 3. Widen the DavidNet 1.5× under the same recipe
**Summary**: Scale channel widths 1.5× (prep 3→96; stages 96→192(+Res)→384→768(+Res); Linear 768→10), ~14.8M params, recipe/schedule unchanged. The time-based one-cycle auto-adapts to the lower epoch count (~85 epochs est. at ~2.25× FLOPs/step). Width ladder (1.25×/1.5×/1.75×) as fallback if a width under-converges.

**What it targets**: Model capacity — EXP-001 spent the budget on ~192 epochs of a small net (deep in diminishing returns) using only 1.6/98 GB; redirect spare FLOPs into parameters that raise the accuracy ceiling (WideResNet: width>depth for CIFAR).

**Reasoning**: VRAM/compute headroom is measured and large; width converts compute to accuracy efficiently; no LR retune needed (time-based schedule). Honest caveat: 95.22% is already airbench-95 territory, so pure width without whitening/TTA may yield only a few tenths, and too-aggressive width risks under-convergence (<24-epoch-equiv).

**Sources**: `proposals/idea-03.md`; WideResNet arXiv:1605.07146; airbench96; EXP-001 `04-analysis.md` (VRAM headroom, next steps).

**Estimated Effort**: low (~8-line width-parameterized `__init__`).

**Risk Assessment**: Too-big → too-few-epochs → one-cycle under-converges below baseline (mitigated by ladder + best-across-epochs + time-based anneal completion). Capacity may not be the binding limiter at 95% (modest gain, clean read either way). Expected **~95.4–95.9%** (central ~95.6%).

## Review

Cross-model adversarial review by **Codex** (full text: `01-idea-review.md`). Scored verdict: **Idea-02 EMA+TTA** evidence 8.5/10, impact 8/10 (winner); Idea-01 whitening 7/6 (gain likely washes out at 192 epochs, narrow margin); Idea-03 wider 6.5/7.5 (capacity-bottleneck claim less proven, pure width may wash/under-converge). **Pick: Idea-02** — "best expected value: directly improves the evaluated iterate and prediction robustness, works through the frozen model(inputs) interface, leaves the 95.22% training recipe mostly intact." Codex independently verified locally that torch 2.9.1 `AveragedModel.forward` delegates to `self.module(...)`, so the wrapped TTA forward is reachable and `.eval()` propagates.

Top concerns and resolutions folded into the chosen idea:
1. **Wall-clock with doubled TTA eval is tighter than stated** (EXP-001 had only ~153s headroom under the 600s cap). → **Gate flip-TTA to the final tail** (`progress >= 0.8`, where the best accuracy is produced) via a model flag, so only ~20% of epochs do a double-forward. Still exactly one `evaluator.evaluate`/epoch.
2. **EMA can regress below baseline if decay lags the low-LR tail** (we eval EMA, not raw, after warmup). → Short-horizon decay **0.998** (~5-epoch horizon; 0.997 if lag appears), seed EMA at 15% warmup, keep `LABEL_SMOOTHING=0.2` unchanged. TTA-only is the floor-preserving fallback if EMA underperforms.
3. **Verify frozen-eval compatibility** → add a smoke check that `evaluator.evaluate(ema_model, device)` runs and `ema_model.module.training` becomes `False`.

## Idea Evaluation

Adopting the reviewer's pick (Idea-02) without override. At 95.22% — already airbench-95 territory near the ~96% ceiling — the highest-EV move is the one that adds reliable gain without risking the hard-won training trajectory: EMA + flip-TTA are orthogonal, near-additive, cheap (~20 lines, no new dep, no schedule change), and verified compatible with the frozen eval. Idea-01 (whitening) and Idea-03 (wider) have higher ceilings in principle but the reviewer and both proposals concede their gains likely wash toward noise at the current epoch-rich/near-ceiling regime; they remain strong *future* loops and, importantly, **compose on top of** the EMA+TTA boosters once validated. Scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: Weight EMA + flip-TTA on DavidNet (Idea-02, `proposals/idea-02.md`), refined per review.

**Why this idea**:
Best expected value with the lowest risk of regressing the 95.22% base. It improves *which* weights are evaluated (EMA → flatter basin) and *how* they read each image (flip-TTA → 2-view ensemble) without touching the proven optimization, so the raw trajectory is preserved as a near-floor. Both boosters are standard, well-evidenced, in-torch (no new dep), and verified reachable through the frozen `evaluate(model, device)` interface. The reviewer's wall-clock and decay risks are engineered away (tail-gated TTA, short-horizon EMA, label-smoothing held fixed). These boosters are also a foundation: once validated they stack with whitening (Idea-01) and wider nets (Idea-03) in later loops.

**Hypothesis**:
Evaluating an EMA of the DavidNet weights (decay 0.998, BN buffers EMA-averaged) each epoch and averaging logits over the image and its horizontal mirror (TTA, gated to the final 20% of training) will raise `best_test_acc` from 95.22% to **~95.5–95.9%** (central ~95.7%), clearing the +0.1pp bar (≥95.32%) with margin, while keeping total wall under the 600s cap and the training recipe byte-identical. Falsified if best_test_acc ≤ 95.32%, the run crashes, or wall exceeds 600s.
