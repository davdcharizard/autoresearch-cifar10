# Proposal idea-02: Multi-crop TTA (extend flip-TTA with reflect-pad shifted crops)

## One-line
Replace the current 2-pass flip-only TTA inside `ResNet9.forward` with airbench's `infer_mirror_translate` scheme — mirror averaged over {center, +1px, −1px} reflect-pad crops (6 forward passes) — an eval-side-only change that averages out the model's residual spatial sensitivity. Targeting 95.87% → ~95.95–96.05%, honestly a coin-flip on clearing the +0.1pp bar.

## Mechanism (tie to the diagnosis: tail accuracy is variance-limited, not capacity-limited)

The EXP-001/002/003 trace shows the recipe is fully annealed (LR→0) and the remaining gains are coming from *eval-side variance reduction*, not from training-dynamics or capacity: EMA (denoise weights) gave +0.50pp and the **flip half** of that win was an attributable +0.25pp tail step-up at the TTA gate (EXP-002 analysis: 95.28→95.49→95.55 right at progress≥0.8). Flip-TTA works because the trained model is not perfectly mirror-invariant, so averaging logits over x and x.flip(-1) cancels the left/right component of its prediction variance.

The same argument extends to *translation*: a conv net with MaxPool downsampling and a global-pool head is not perfectly shift-invariant (the pooling grid aliases sub-pixel position). Averaging logits over a small set of ±1px translated crops cancels the spatial-position component of prediction variance that flip alone leaves on the table. airbench reports this is a real, separable gain: `tta_level=2` (mirror+translate) is what takes airbench96 from ~95.6% (no TTA) to its 96.05% headline, and the paper singles out multi-crop TTA as the one feature whose benefit does NOT reduce to epochs-to-94% (i.e. it is a genuinely orthogonal eval-side lever, not a training speedup). The causal chain to the metric: more averaged views → lower per-example logit variance → fewer borderline examples flip to the wrong class → higher `correct` count in the unchanged `Eval.evaluate` loop.

Crucially this composes with everything already on (EMA-denoised weights, frozen whitening) because it operates purely on the input batch inside `forward()`, never touching training.

## Concrete change (files/functions I actually read)

All edits are in `train.py`. The eval harness `prepare.py:Eval.evaluate` (lines 32–47) calls `model(inputs)` once per batch — so all TTA must live inside `forward()`; this is unchanged from the current design.

**1. `ResNet9._forward_once` (train.py:150–157) — leave as-is.** It is the single clean-pass primitive (whiten→prep→layer1→layer2→layer3→pool→fc×scale_out). The TTA views call it.

**2. `ResNet9.forward` (train.py:159–164) — replace the flip-only block.** Currently:
```python
def forward(self, x):
    if self.training or not self.tta:
        return self._forward_once(x)
    return 0.5 * (self._forward_once(x) + self._forward_once(x.flip(-1)))
```
Change to airbench's `infer_mirror_translate` structure, implemented with `_forward_once` as the net and `F.pad(..., mode='reflect')` for the crops (F is already imported, train.py:7):
```python
def _mirror(self, x):
    return 0.5 * (self._forward_once(x) + self._forward_once(x.flip(-1)))

def forward(self, x):
    if self.training or not self.tta:
        return self._forward_once(x)
    logits = self._mirror(x)
    pad = 1
    xp = F.pad(x, (pad, pad, pad, pad), mode='reflect')   # [N,C,34,34]
    crops = [xp[:, :, 0:32, 0:32], xp[:, :, 2:34, 2:34]]   # ∓1px diagonal shifts
    logits_t = torch.stack([self._mirror(c) for c in crops]).mean(0)
    return 0.5 * logits + 0.5 * logits_t
```
This is exactly airbench's `tta_level=2` weighting: 0.5·mirror(center) + 0.5·mean(mirror(shift+), mirror(shift−)). Note the geometry: with `pad=1` the padded tensor is 34×34; `[0:32,0:32]` and `[2:34,2:34]` are the two ±1px diagonal translations, and the *un-shifted* center view corresponds to the original `x` already used in the first `_mirror(x)` term — so the three crop positions are {0, +1px-diag, −1px-diag}, mirrored each → 6 `_forward_once` calls per batch.

**3. Keep `self.tta` flag (train.py:133) and the gating in `main()` (train.py:319–326) unchanged.** `use_tta = eval_progress >= TTA_START_FRAC` (0.8) still gates the heavy path to the final ~20% of epochs, which is what bounds the eval wall-clock (see Risks). No constant or schedule change is required for the primary variant.

No other edits. Optimizer, schedule, EMA, whitening, seed all byte-identical → clean single-variable A/B vs the 95.87% flip-only run.

## Eval wall-clock budget (the binding practical constraint)

TTA forward passes are OFF the 300s *training* budget (the loop times only `total_training_time`, train.py:264/291–292; eval is outside that accumulator). But the **600s total-wall `timeout` kill is hard** and current total wall is ~452.8s (EXP-003). I must not blow it.

Cost accounting: flip-TTA already does 2 `_forward_once` per eval batch on the TTA-gated epochs. Multi-crop does 6 → +4 passes per batch, but only on the ~20% of epochs past the gate. The test set is 10000 imgs at batch 256 = 40 batches/eval. EXP-003 ran 174 epochs in 300s training; the gated epochs are roughly the last 35 (20%). Extra cost ≈ 35 evals × 40 batches × 4 extra passes. A `_forward_once` on a 256-batch is sub-millisecond on this GPU (~30k img/s training throughput includes backward; pure inference is faster), so the test forward of 10000 imgs is on the order of a few hundred ms. The added 4 passes × ~35 gated epochs is an estimated **~30–70s of extra wall**, landing total wall around ~480–520s — under 600s with margin but NOT comfortable. Mitigations if it runs hot:
- The gate already restricts heavy TTA to the tail; do NOT lower `TTA_START_FRAC` below 0.8 for the primary variant.
- If a smoke test shows wall >560s, drop to the 4-pass variant (center+one shift, i.e. crops=`[xp[:,:,0:32,0:32]]` only) — still a translate gain, half the extra cost.

The reflect-pad crop op itself is negligible (one `F.pad` + two slices per batch).

## Evidence

- **airbench `legacy/airbench96.py` `infer()`** (fetched source): `infer_mirror_translate` does exactly mirror over 3 reflect-pad-1 crops {[0:32,0:32], center, [2:34,2:34]} with 0.5/0.5 weighting = 6 passes. This is the documented `tta_level=2` that produces the 96.05% headline; `tta_level=1` (flip only) is the weaker setting.
- **airbench paper (arXiv:2404.00498) §3.5**: multi-crop TTA is called out as the one feature whose gain is NOT explained by epochs-to-94% — i.e. a real orthogonal eval-side lever, not a training speedup. Reported flip-only ≈ +0.2–0.3pp, full multi-crop reaching 96.05%.
- **EXP-002 analysis** (experiments/002/04-analysis.md): the flip half of TTA was an *attributable* +0.25pp step-up at the gate on THIS exact model — direct in-codebase evidence that this model has exploitable eval-time prediction variance and that averaging-over-views converts it to accuracy here, not just in airbench's net.
- **Codebase fact**: `Eval.evaluate` (prepare.py:42) calls `model(inputs)` and only counts `evaluate()` calls, not forward passes (the ≤1-eval/epoch constraint is on `evaluator.evaluate`, train.py:327, called once per epoch). Multi-crop stays inside one `forward()` → fully in-scope, exactly as flip-TTA already is.

## Strongest risk

**The marginal gain over flip-only may be <0.1pp and miss the bar.** The airbench 96.05% headline compares multi-crop TTA to NO TTA. We already bank the flip component (the larger half). The translate-only increment in airbench's own ablation is the *smaller* residual, plausibly ~+0.1–0.2pp, and our net differs (DavidNet MaxPool(2)×3 + MaxPool(4) global vs airbench's architecture; trained on random-translate=4 augmentation here via `RandomCrop(32, padding=4)`, train.py:186, which already builds in substantial shift-invariance and may shrink the TTA translate headroom). If the model is already near-shift-invariant from training-time crop aug, the ±1px eval crops add little and the result could land 95.88–95.95% — a real, mechanistically-sound gain that nonetheless fails the +0.1pp bar. This is the honest central worry: **eval-side variance reduction has diminishing returns, and we are stacking on top of the flip win that already captured most of it.**

Secondary risks: (a) reflect-pad introduces a 1px mirrored border that is mildly off-distribution vs the test images — low risk at pad=1 (airbench uses exactly this), but a reason not to push to larger shifts; (b) wall-clock creep toward 600s if the per-pass cost is higher than estimated — mitigated by the gate and the 4-pass fallback; (c) bf16 autocast is NOT used in eval (`Eval.evaluate` runs the model in default precision under `inference_mode`), so the TTA passes are fp32 — slightly slower than a bf16 estimate would suggest, reinforcing the wall-clock caution.

## Quantitative estimate

Central estimate: **95.87% → ~95.95%** (≈+0.08pp), with a plausible band of **+0.0 to +0.18pp** (95.87–96.05%). I put **~45–55% probability on clearing the ≥95.97% bar**. This is deliberately less optimistic than the airbench headline because (i) we already hold the flip component and (ii) training-time translate aug pre-reduces the translate headroom. The upside case (lands ~96.0%) materializes if this model retains meaningful shift-sensitivity despite crop aug — EXP-002's clean flip step-up suggests it might.

## Effort

**Low.** One localized rewrite of `forward()` plus a small `_mirror` helper; no new constants, no schedule/optimizer/training change, no new deps (`F.pad`, `torch.stack` already available). The only real care item is a pre-run smoke check of total wall-clock (must stay <600s) and a per-epoch eval-time sanity print, then a single official run. Well within one experiment loop.

## Sources
- [cifar10-airbench legacy/airbench96.py infer()](https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench96.py)
- [94% on CIFAR-10 in 3.29 Seconds (arXiv:2404.00498)](https://arxiv.org/abs/2404.00498)
- [airbench performance benchmarks (DeepWiki)](https://deepwiki.com/KellerJordan/cifar10-airbench/1.2-performance-benchmarks)
