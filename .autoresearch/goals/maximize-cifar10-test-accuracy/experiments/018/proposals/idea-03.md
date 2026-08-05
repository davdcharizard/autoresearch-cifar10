# Proposal idea-03: AdaptiveConcatPool head — richer global pooling (avg ⊕ max)

## Core change (train.py only)
Replace the final `nn.MaxPool2d(4)` + `flatten` + `fc(512→10)` head with **adaptive concat pooling**: pool the final 4×4×512 feature map with BOTH global-average-pool AND global-max-pool, concatenate the two 512-vectors into a 1024-vector, and feed `fc(1024→10)`. Classic fastai "AdaptiveConcatPool2d" head.

```python
# in __init__: self.fc = nn.Linear(1024, num_classes, bias=False)
def _forward_once(self, x):
    x = self.whiten(x); x = self.prep(x)
    x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)   # [N,512,4,4]
    a = x.mean((2, 3)); m = x.amax((2, 3))                       # GAP, GMP -> [N,512]
    x = torch.cat([a, m], dim=1)                                 # [N,1024]
    return self.fc(x) * self.scale_out
```
Env `HEAD_POOL` (`max` = baseline, `avgmax` = concat, `avg` = global-avg-only) to sweep. The only shape change is `fc`: 512→10 becomes 1024→10 (+5,120 params, negligible). SCALE_OUT may need a minor retune (concat changes the logit-input scale) — expose as a rider but default unchanged first.

## Mechanism — why this is a DIFFERENT (representational) lever
The current head discards information: `MaxPool2d(4)` keeps only the per-channel **maximum** over the 4×4 map, throwing away the spatial-average signal. Average pooling and max pooling capture **complementary statistics** — max responds to the most salient activation (presence of a feature), average integrates over the whole map (prevalence/extent). Concatenating both gives the linear classifier a strictly richer, lower-variance representation than either alone. This changes *what the classifier sees* — a representation change, distinct from the saturated capacity/optimizer/regularization/aug axes, and it is genuinely **throughput-free** (two cheap reductions replacing one, fc unchanged in cost).

## Why it targets the limiter
The limiter is the generalization ceiling (project-insights High, EXP-014). This idea attacks it at the **readout**: the net's penultimate representation may carry signal that max-only pooling discards. Because it is throughput-free (no epoch cost — the #1 failure mode is sidestepped entirely, like the EXP-008 aug win), it is a clean, low-risk probe of whether the head pooling is leaving accuracy on the table. The DavidNet lineage uses max-pool by convention; the airbench/fastai lineages use avg or concat — this tests which the readout actually wants here.

## Throughput
Strictly throughput-neutral: replaces one `MaxPool2d(4)` reduction with two reductions (mean + amax) over a tiny 4×4 map, and the fc matmul grows from 512×10 to 1024×10 (both trivial). Expect num_epochs unchanged (~149). No fused-kernel risk. This is the safest of the three on the under-anneal axis.

## Design — SAME-SESSION multi-cell
- c0: unchanged baseline (`MaxPool2d(4)` head) — full-speed same-session anchor.
- cA: avg⊕max concat head, fc 1024→10 — PRIMARY.
- cB: global-avg-only head (fc 512→10) OR concat + SCALE_OUT retune — second operating point (isolates whether avg alone, or the concat richness, drives any gain).

## Correctness / EMA / eval
- `fc` re-sized to 1024 input; kaiming-init via existing `_weights_init`. Optimizer picks it up.
- Eval/EMA: deterministic reductions, train≡eval; `AveragedModel(use_buffers=True)` averages the resized fc weight normally. flip-TTA valid (GAP and GMP are both flip-invariant over spatial dims).
- bf16/channels_last: `mean`/`amax` over spatial dims preserve memory format.
- Smoke: (i) head output shape [N,10] unchanged; (ii) `HEAD_POOL=max` path is bit-identical to the current baseline (regression guard — must reproduce 96.38-class behavior); (iii) one-step backward finite grads; (iv) num_params rises by exactly 512×10 = 5,120.

## Verification
- Best head cell ≥ **96.48** AND > same-session c0 by >0.1pp, replicated with a confirmation re-run on any apparent win (low-c0-draw lesson, EXP-016/017).
- num_epochs ≈ 149 (must stay full — this lever's whole premise is throughput-free); ep25 within ~0.5pp of c0; fully annealed.
- Integrity: train.py-only; prepare.py byte-unchanged; ≤1 eval/epoch; seed 42; `HEAD_POOL=max` ≡ baseline smoke.
- ON A WIN: bake the winning head as default.

## Hypothesis
Giving the linear classifier both the average AND max pooled statistics (vs max only) supplies complementary global features and lifts best_test_acc ≥96.48 over the same-session control, throughput-free at ~149 epochs. If it ties at full epochs, the max-pooled readout already captures the linearly-separable signal and the head is not the limiter — narrowing the search to the feature extractor's inductive bias (idea-01 BlurPool) or the stem.

## Effort: low. Risk: (1) smallest-upside of the three — a linear readout change may only shuffle ~0.1pp (honest prior: the net is deep enough that GAP vs GMP rarely matters much by the final layer); (2) SCALE_OUT may need retuning for the concat magnitude (mitigated: cB retune rider); (3) likely the cheapest to combine as a free rider on idea-01/02 rather than a standalone win.
## Sources: fastai AdaptiveConcatPool2d (docs.fast.ai, layers); Lin et al. 2014 "Network in Network" (global avg pooling, arXiv:1312.4400); project-insights High (generalization ceiling EXP-014); train.py:152-178 (the pool/fc head).
