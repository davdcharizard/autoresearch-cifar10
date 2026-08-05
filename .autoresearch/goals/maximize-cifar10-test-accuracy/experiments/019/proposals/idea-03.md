# Proposal idea-03: AdaptiveConcatPool head — richer global pooling (avg ⊕ max)

## Core change (train.py only)
Replace the final `nn.MaxPool2d(4)` + flatten + `fc(512→10)` head with **adaptive concat pooling**: pool the final 4×4×512 map with BOTH global-average AND global-max, concatenate into a 1024-vector, feed `fc(1024→10)`.

```python
# __init__: self.fc = nn.Linear(1024, num_classes, bias=False)
def _forward_once(self, x):
    x = self.whiten(x); x = self.prep(x)
    x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)   # [N,512,4,4]
    a = x.mean((2, 3)); m = x.amax((2, 3))                       # GAP, GMP
    return self.fc(torch.cat([a, m], 1)) * self.scale_out
```
Env `HEAD_POOL` (`max`=baseline, `avgmax`=concat, `avg`=global-avg-only). Only `fc` input grows 512→1024 (+5,120 params).

## Mechanism — why this is a DIFFERENT (representational) lever
The current head keeps only the per-channel **max** over the 4×4 map, discarding the spatial-average. Avg and max pooling capture **complementary statistics** (prevalence/extent vs salient presence). Concatenating both gives the linear classifier a strictly richer, lower-variance readout. This changes *what the classifier sees* — a representation change distinct from the saturated capacity/optimizer/regularization/aug/downsampling axes — and is genuinely **throughput-free**.

## Why it targets the limiter
The limiter is the generalization ceiling (project-insights High, EXP-014). This attacks it at the **readout**: max-only pooling may discard linearly-useful signal in the penultimate representation. Throughput-free (two cheap reductions replacing one; fc cost unchanged) → sidesteps the #1 failure mode (under-anneal) entirely, like the EXP-008 aug win. The DavidNet lineage's max-only head is a convention, not a tuned choice; airbench/fastai use avg/concat.

## Throughput
Strictly neutral: replaces one `MaxPool2d(4)` reduction with two reductions (mean+amax) over a tiny 4×4 map; fc matmul 512×10→1024×10 (trivial). num_epochs ~150 (verify). No fused-kernel risk — the safest finalist on the under-anneal axis.

## Design — SAME-SESSION multi-cell
- c0: unchanged baseline (`MaxPool2d(4)` head) — full-speed anchor.
- cA: avg⊕max concat head, fc 1024→10 — PRIMARY.
- cB: global-avg-only head (fc 512→10) — isolates whether avg alone or the concat richness drives any gain.

## Correctness / EMA / eval
- `fc` re-sized to 1024 input, kaiming-init; optimizer picks it up.
- Eval/EMA: deterministic reductions, train≡eval; `AveragedModel(use_buffers=True)` averages the resized fc weight; flip-TTA valid (GAP/GMP flip-invariant over spatial dims).
- bf16/channels_last: `mean`/`amax` over spatial dims preserve memory format.
- Smokes: (i) head output [N,10] unchanged; (ii) `HEAD_POOL=max` bit-identical to baseline (regression guard); (iii) finite backward; (iv) num_params rises by exactly 5,120; (v) eval native-fp32 + flip coverage.

## Verification
- Best head cell ≥ **96.48** AND > same-session c0 by >0.1pp, replicated with a mandatory confirmation re-run on any apparent win.
- num_epochs ≈ 150 (throughput-free premise); ep25 within ~0.5pp of c0; fully annealed.
- Integrity: train.py-only; prepare.py byte-unchanged; ≤1 eval/epoch; seed 42; summary best == per-epoch max; `HEAD_POOL=max` ≡ baseline smoke.
- ON A WIN: bake the winning head as default.

## Hypothesis
Giving the linear classifier both avg AND max pooled statistics (vs max only) supplies complementary global features and lifts best_test_acc ≥96.48 over the same-session control, throughput-free at ~150 epochs. If it ties, the max-pooled readout already captures the linearly-separable signal — narrowing the search to the feature extractor (idea-01 SE) or schedule (idea-02).

## Effort: low. Risk: (1) smallest-upside finalist — a linear readout change may only shuffle ~0.1pp (the deep net's final features may already be max-separable; EXP-018 reviewer scored this idea 4.5/10 as "too shallow for the stated limiter"); (2) SCALE_OUT may need a minor retune for the concat magnitude (expose as a rider, default unchanged first); (3) likely best as a cheap rider on a stronger change.
## Sources: EXP-018 proposals/idea-03.md + 01-idea-review.md (4.5/10); fastai AdaptiveConcatPool2d; Lin et al. 2014 NiN (arXiv:1312.4400); train.py:152-178.
