# Proposal idea-02 (EXP-005): Multi-crop TTA — extend flip-TTA with reflect-pad shifted crops

## One-line
Replace the current 2-pass flip-only TTA inside `ResNet9.forward` with airbench's `infer_mirror_translate` scheme — mirror averaged over {center, +1px-diag, −1px-diag} reflect-pad crops (6 forward passes) — an **eval-side-only** change that averages out the model's residual spatial sensitivity. Re-developed for the new 96.00% baseline (EXP-004) against the harder **96.10%** bar. Honest read: lowest-risk lever (no training change, cannot destabilize) but plausibly **lower-ceiling** — central estimate ~96.04%, roughly a **coin-flip-to-slightly-against** clearing +0.1pp.

## Mechanism (tie to the named limiter: tail accuracy is eval-side-variance-limited)

The EXP-001→004 trace shows the recipe is fully annealed (LR→0 by budget end, `03-experiment-learnings.md` Medium: "Most accuracy gain arrives in the low-LR tail") and the cheap remaining gains have been coming from **eval-side variance reduction**, not training dynamics. The single most direct in-codebase evidence: in EXP-002, the **flip** half of TTA produced an *attributable* tail step-up of ~+0.25pp **on this exact model** right at the `progress≥0.8` gate — `95.28% (ep144) → 95.49% (ep145) → 95.55% (ep146)` (`experiments/002/04-analysis.md` § Execution/Observations). Flip-TTA works because the trained net is not perfectly mirror-invariant, so averaging logits over `x` and `x.flip(-1)` cancels the left/right component of prediction variance.

The same argument extends to **translation**: a conv net with three `MaxPool2d(2)` downsamples plus a global `MaxPool2d(4)` head (train.py:149–152) is not perfectly shift-invariant — the pooling grid aliases sub-pixel position, so the argmax for borderline examples depends on the exact crop offset. Averaging logits over a small set of ±1px translated crops cancels the spatial-position component of prediction variance that flip alone leaves on the table. airbench reports this is a real, *separable* gain: `tta_level=2` (mirror+translate) is the setting that yields its **96.05%** headline at this model scale, while `tta_level=1` (flip only) is weaker. The causal chain to the metric is concrete: more averaged views → lower per-example logit variance → fewer borderline test examples flip to the wrong class → higher `correct` count in the unchanged `Eval.evaluate` loop (prepare.py:42–45).

Crucially this **composes** with everything already on (EMA-denoised weights from EXP-002, frozen whitening from EXP-003, the ReZero capacity block from EXP-004) because it operates purely on the input batch inside `forward()` and never touches training — so it stacks cleanly on top of the 96.00% result rather than trading against it.

## Concrete change (files/functions I actually read)

All edits are in `train.py`. The frozen eval harness `prepare.py:Eval.evaluate` (prepare.py:32–47) calls `model(inputs)` once per batch under `@torch.inference_mode()` in **fp32** (no autocast in eval) — so all TTA must live inside `forward()`, exactly as the current flip-TTA already does. The ≤1-`evaluate()`-per-epoch constraint is on `evaluator.evaluate` calls (train.py:348, one per epoch), NOT on forward passes — so multi-pass TTA inside one `forward()` is in-scope, the same way the existing 2-pass flip-TTA is.

**1. `ResNet9._forward_once` (train.py:171–178) — leave byte-identical.** It is the single clean-pass primitive (`whiten → prep → layer1 → layer2 → layer3 → pool.flatten → fc × scale_out`). All TTA views call it.

**2. `ResNet9.forward` (train.py:180–185) — replace the flip-only block.** Current code:
```python
def forward(self, x):
    if self.training or not self.tta:
        return self._forward_once(x)
    return 0.5 * (self._forward_once(x) + self._forward_once(x.flip(-1)))
```
Replace with airbench's `infer_mirror_translate` structure, using `_forward_once` as the net and `F.pad(..., mode='reflect')` for the crops (`F` is already imported, train.py:7):
```python
def _mirror(self, x):
    return 0.5 * (self._forward_once(x) + self._forward_once(x.flip(-1)))

def forward(self, x):
    if self.training or not self.tta:
        return self._forward_once(x)
    logits = self._mirror(x)                               # center, mirrored
    pad = 1
    xp = F.pad(x, (pad, pad, pad, pad), mode="reflect")    # [N,C,34,34]
    crops = [xp[:, :, 0:32, 0:32], xp[:, :, 2:34, 2:34]]   # ∓1px diagonal shifts
    logits_t = torch.stack([self._mirror(c) for c in crops]).mean(0)
    return 0.5 * logits + 0.5 * logits_t
```
This is **exactly** airbench's verified `tta_level=2` weighting (confirmed verbatim against `legacy/airbench96.py`: `pad=1`, `F.pad(inputs,(pad,)*4,'reflect')`, crops `[:, :, 0:32, 0:32]` and `[:, :, 2:34, 2:34]`, `0.5*net(x)+0.5*net(x.flip(-1))` per view, `0.5*logits + 0.5*logits_translate`). Geometry: with `pad=1` the padded tensor is 34×34; `[0:32,0:32]` and `[2:34,2:34]` are the two ±1px *diagonal* translations, and the center (un-shifted) view is the first `_mirror(x)` term on the original `x`. Three crop positions × mirror each = **6 `_forward_once` calls per batch** on the TTA-gated epochs (vs 2 today). `torch.stack` and `torch.mul`/`add` are already available; no new deps.

**3. Keep `self.tta` flag (train.py:154) and the eval gating in `main()` (train.py:340–347) unchanged.** `use_tta = eval_progress >= TTA_START_FRAC` (`TTA_START_FRAC = 0.8`, train.py:30) still gates the heavy path to the final ~20% of epochs, which is what bounds the eval wall-clock (see below). The gated module is the EMA model after warmup (`ema_model.module.tta = use_tta`, train.py:343) — so multi-crop runs on the EMA weights, consistent with EXP-002/004.

No other edits. Optimizer, schedule, EMA, whitening, ReZero block, seed all **byte-identical** → a clean single-variable, eval-side-only A/B against the 96.00% flip-only run. Because training is untouched, the run is reproducible bit-for-bit through training and the *only* thing that can move is the gated-epoch eval accuracy — this lever cannot destabilize or regress training.

## Eval wall-clock budget (the binding practical constraint)

TTA forward passes are **OFF** the 300s *training* budget — the loop accumulates only `total_training_time` (train.py:269, 313) inside the data loop, and `evaluator.evaluate` runs after the inner loop (train.py:348), outside that accumulator. But the **600s total-wall `timeout` kill is HARD**, and the current baseline total wall is **445.2s** (EXP-004 § Execution). I must not blow it.

Cost accounting from real measurements:
- The test set is 10,000 imgs at batch 256 = **40 batches/eval** (prepare.py:24–30).
- EXP-004 ran **142 epochs** in 300s training; the TTA-gated epochs are the final ~20% past `progress≥0.8`, i.e. roughly the **last ~28 epochs**.
- Flip-TTA already does **2** `_forward_once` per eval batch on those gated epochs; multi-crop does **6** → **+4 passes/batch** on ~28 epochs only. Non-gated epochs (~114) are byte-identical 1-pass evals and unaffected.
- **Empirical anchor**: EXP-002 (2-pass flip-TTA, 37 gated epochs of its 183) had wall 442.7s; EXP-004 (same TTA, 28 gated of 142) had wall 445.2s. The total non-training wall (eval + startup + dataset/loader setup) is ~145s across ~142 evals → per-epoch eval averages ~1s, of which the model forward on 10k fp32 imgs is a fraction (the rest is loader iteration / `.item()` syncs). A single extra `_forward_once` of 10k imgs at batch 256 in fp32 inference_mode on this H20-class GPU is on the order of **~0.2–0.4s** (training does ~26–30k img/s *including backward*; pure fp32 inference of 10k imgs ≈ a few hundred ms).

**Estimate**: +4 extra passes × ~28 gated epochs × ~0.25–0.35s/pass ≈ **+28–40s** of extra wall, landing total wall around **~475–490s** — under 600s with a ~110–125s margin, but NOT so comfortable that I'd skip a check. **A pre-run smoke timing is mandatory** (the EXP-004 plan-review explicitly flagged the fp32 eval cost as needing a real measurement): run one `model.tta = True` eval through `Eval.evaluate` and print the wall delta vs a `tta = False` eval before the official run. Mitigations if it runs hot:
- The gate already restricts heavy TTA to the tail; keep `TTA_START_FRAC = 0.8` for the primary variant (do NOT widen it — widening adds wall for the lowest-value early-tail epochs).
- If the smoke shows projected total wall **> 560s**, drop to a **4-pass** variant (center + ONE shift: `crops = [xp[:, :, 0:32, 0:32]]`) — still a translate gain at half the extra cost.

The reflect-pad crop op itself (one `F.pad` + two slice views per batch) is negligible.

## Evidence

- **airbench `legacy/airbench96.py` `infer_mirror_translate`** (fetched & verified verbatim): mirror over 3 reflect-pad-1 crops `{[0:32,0:32], center, [2:34,2:34]}` with 0.5/0.5 weighting = 6 passes. This is the documented `tta_level=2` producing the **96.05%** headline; `tta_level=1` (flip only) is the weaker setting. Our implementation above is byte-for-byte this scheme.
- **airbench paper (arXiv:2404.00498)**: multi-crop TTA is the eval-side lever that lifts the 96-tier net to its 96.05% headline; the translate component is documented as a real, separable gain on top of flip.
- **EXP-002 analysis** (`experiments/002/04-analysis.md` § Observations): the flip half of TTA was an *attributable* +0.25pp tail step-up at the gate on THIS exact model — direct in-codebase proof that this net has exploitable eval-time prediction variance and that averaging-over-views converts it to accuracy *here*, not just in airbench's net.
- **Codebase facts**: `Eval.evaluate` (prepare.py:42) calls `model(inputs)` once/batch; the ≤1-eval/epoch constraint is on `evaluator.evaluate` (train.py:348, one call/epoch). Multi-crop stays inside one `forward()` → fully in-scope, exactly as the current 2-pass flip-TTA already is (train.py:185). Training is keyed on `total_training_time` only (train.py:285), so eval passes do not consume the 300s budget.

## Strongest risk

**The marginal gain of translate over flip-only is likely <0.1pp, and the bar is now HARDER.** This is the honest central worry and it is *worse* than in the EXP-004 framing:

1. **We already bank the flip component** (the larger half of TTA's value, attributed at +0.25pp in EXP-002). airbench's 96.05% headline compares full multi-crop to **no** TTA; the translate-only *increment* on top of flip is the smaller residual, plausibly ~+0.05–0.15pp.
2. **Training already builds shift-invariance.** `train_tf` uses `RandomCrop(32, padding=4)` (train.py:207) — the model is trained on ±4px random translations every step, which directly reduces the eval-time translate headroom that ±1px crops would otherwise recover. If the net is already near-shift-invariant from this aug, the eval crops add little.
3. **The bar moved up onto a higher base.** We are at 96.00% (EXP-004) *without* this lever; airbench reports **96.05% WITH** it at this scale. The headroom from "96.00 without translate-TTA" to ">96.10 with it" requires the translate increment to be **>0.10pp on top of an already-flip-TTA'd, already-crop-augmented net** — i.e. it must roughly *match or exceed* the entire flip increment, which is not what the diminishing-returns picture predicts. There is a real chance the result lands **96.00–96.08%**: a mechanistically-sound, genuine gain that nonetheless **fails the +0.1pp bar** and scores `no-improvement`.

**Secondary risks**: (a) reflect-pad introduces a 1px mirrored border mildly off-distribution vs the clean test images — low at pad=1 (airbench uses exactly this), but a reason NOT to push to ±2px or 4-neighbor shifts (richer sets add wall-clock and border artifacts for diminishing variance reduction; I deliberately keep the airbench-exact 6-pass set). (b) Wall-clock creep toward 600s if per-pass cost exceeds the estimate — mitigated by the gate + mandatory smoke + the 4-pass fallback. (c) fp32 eval (no autocast in `Eval.evaluate`) makes each pass slightly slower than a bf16 estimate would suggest — already accounted for in the ~0.25–0.35s/pass figure above.

The assumption that most needs to hold: **the EMA-weighted, crop-augmented net retains enough residual sub-pixel shift sensitivity at the annealed tail that ±1px averaging cancels >0.10pp of misclassifications.** EXP-002's clean flip step-up is weak supporting evidence that this net *does* carry exploitable view-variance; whether the *translate* slice of it clears 0.1pp is the coin-flip.

## Quantitative estimate

Central estimate: **96.00% → ~96.04%** (≈ +0.04pp), with a plausible band of **+0.00 to +0.12pp** (96.00–96.12%). I put **~35–45% probability on clearing the ≥96.10% bar** — deliberately *below* even-odds because (i) we already hold the flip component, (ii) `RandomCrop(pad=4)` pre-reduces the translate headroom, and (iii) the bar now sits at 96.10 while airbench's *full*-TTA result at this scale is only 96.05. The upside case (lands ≥96.10) materializes only if this net retains meaningfully more shift-sensitivity than airbench's despite the crop aug — possible (our pooling chain differs: `MaxPool2d(2)×3 + MaxPool2d(4)` global, train.py:149–152) but not the base rate.

This is **less optimistic than the EXP-004 version of this idea** (which estimated ~45–55% at the easier base), because the bar moved onto a higher base and the lever's headroom did not.

## Effort

**Low.** One localized rewrite of `forward()` plus a small `_mirror` helper; no new constants, no schedule/optimizer/training/EMA/whitening change, no new deps (`F.pad`, `torch.stack` already imported, train.py:4–7). The only real care items: (1) a **mandatory pre-run smoke** of one `tta=True` vs `tta=False` eval to measure the real per-pass wall delta and confirm projected total wall < 600s, and (2) a single official run under `timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py'` on GPU 1. Well within one experiment loop.

## Honest bottom line for the selector

This is the **lowest-risk** candidate (eval-side only, byte-identical training, cannot regress or destabilize) but plausibly the **lowest-ceiling** one against the now-harder 96.10 bar. It is a strong pick **if** the loop wants a safe, fast probe that banks any positive translate-TTA increment and de-risks the eval path for future runs; it is a **weak** pick if the loop needs a high-probability bar-clear, since the most likely outcome is a real-but-sub-0.1pp gain. If selected, the 4-pass fallback and the smoke-timing gate should be treated as part of the plan, not optional.

## Sources
- [cifar10-airbench legacy/airbench96.py infer()](https://github.com/KellerJordan/cifar10-airbench/blob/master/legacy/airbench96.py) — verified `infer_mirror_translate` verbatim (pad=1 reflect, crops [0:32,0:32]/[2:34,2:34], 0.5/0.5 weighting, 6 passes).
- [94% on CIFAR-10 in 3.29 Seconds on a Single GPU (arXiv:2404.00498)](https://arxiv.org/abs/2404.00498)
- `experiments/002/04-analysis.md` — flip-TTA +0.25pp attributable step-up on this model.
- `experiments/004/04-analysis.md` — 96.00% baseline, 142 epochs, 445.2s wall; lists multi-crop TTA as a deferred composable next step.
