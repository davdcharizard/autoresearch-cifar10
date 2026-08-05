# Proposal (EXP-011, idea-01): CutMix data-mixing regularization

## One-line summary
Add **CutMix** (Yun et al., ICCV 2019, arXiv:1905.04899) to the training step as a GPU-side, throughput-free regularizer: with per-batch probability `p`, sample `λ~Beta(α,α)`, paste a random box of area `(1-λ)` from a batch-permutation into each image, and train on the area-corrected mixed loss `λ·CE(out,y) + (1-λ)·CE(out,y_perm)`. This is the canonical *strong* CIFAR mixing augmentation, it is UNTRIED here, and (unlike plain mixup) it pays off at short schedules — exactly the regime this net sits in.

## Mechanism (tied to the named limiter)

The diagnosis (`03-experiment-learnings.md`, Patterns + Failed Approaches) is now strongly converged: this net is **regularization-bound with a ~4× epoch surplus** (fits ~150 epochs in 300s for 96.38% vs airbench96's ~37). Three axes are confirmed exhausted or harmful:
- **Capacity** under-anneals (EXP-005 142→131, EXP-007 150→94, both lost epochs → still-climbing tails).
- **Optimizer** is a flat tie (EXP-009/010: Muon == SGD at ~96.35±noise → "regularization-bound, not optimizer-bound").
- **Eval-side TTA** is near-exhausted (EXP-006 −0.07pp, increment hides under the noise floor).

The single most productive lever in this project's history is **throughput-free regularization**: EXP-008 (Cutout 8→12 + light RandomErasing) gave **+0.38pp (96.00→96.38)** — the largest gain since the recipe install — by converting wasted epochs into a higher annealed ceiling *without cutting epochs* (augmentation runs on CPU workers, GPU step untouched).

CutMix is the same lever class, but acting through a **different and complementary mechanism** than Cutout/RandomErasing:
1. **Cutout/RandomErasing are single-image occlusion** — they delete information (zero a box) and keep the original hard label. They regularize by forcing the net to use distributed cues.
2. **CutMix is two-image region mixing with soft labels** — it pastes a real patch from another class and splits the label by area. This (a) provides a *richer training signal in the deleted region* (real content instead of zeros), which the CutMix paper argues is why it beats Cutout, and (b) imposes a **label-mixing / soft-target regularization** that biases the net toward calibrated, localized evidence accumulation. Mechanistically distinct from occlusion, so it can stack on top of the EXP-008 occlusion augmentation rather than being redundant with it.

The causal chain to the metric: CutMix slows convergence (consumes more of the epoch surplus) and shifts the *fully-annealed* minimum toward a flatter, better-generalizing solution. Because most accuracy here lands in the low-LR tail (EXP-001 Pattern), and the tail is budget-protected (the GPU step cost of CutMix is ~zero), the gain shows up where the project consistently captures it.

**Why CutMix and not mixup (the short-schedule argument, must reflect in design):** plain input-space mixup typically needs very long schedules (800–2000 epochs) to pay off — Manifold/mixup papers train PreAct-ResNet18 for 2000 epochs on CIFAR; "100 epochs is enough *without* it" (Mixup Without Hesitation, arXiv:2101.04342). CutMix converges much faster and gives its best *relative* advantage early. Published ResNet-18/CIFAR-10 @200-epoch numbers: baseline ~95.28, mixup ~95.55, **CutMix ~96.22** (TransformMix Table; OpenMixup CutMix-protocol benchmarks). At our ~150-epoch effective budget, CutMix is the right member of the mixing family.

## Concrete change (this codebase)

All edits are inside `main()` in `train.py`; no architecture, no `prepare.py`, no new deps (pure torch). I read the actual training step (`train.py:279-304`) and the loss/criterion construction (`train.py:251`).

### 1. New hyperparameters (near the existing block, `train.py:19-31`)
```python
CUTMIX_ALPHA = 1.0       # Beta(α,α); α=1 → uniform λ, the CutMix-paper default
CUTMIX_P = 0.5           # per-batch probability of applying CutMix
CUTMIX_OFF_TAIL_FRAC = 0.85   # disable CutMix once progress ≥ this (clean low-LR tail); see §4
```

### 2. A CutMix helper (module level, pure torch, GPU-side)
```python
def cutmix_batch(inputs, alpha, generator=None):
    """Region-mix a batch with a permutation of itself. Returns
    (mixed_inputs, perm, lam) where lam is the AREA-CORRECTED label weight
    on the original targets. Operates in-place-safe on a clone."""
    n, _, h, w = inputs.shape
    perm = torch.randperm(n, device=inputs.device)
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    # box of area (1-lam): side ratio sqrt(1-lam)
    r = (1.0 - lam) ** 0.5
    cut_h, cut_w = int(h * r), int(w * r)
    cy = int(torch.randint(h, (1,), device=inputs.device).item())
    cx = int(torch.randint(w, (1,), device=inputs.device).item())
    y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
    mixed = inputs.clone()
    mixed[:, :, y1:y2, x1:x2] = inputs[perm, :, y1:y2, x1:x2]
    # CORRECT lam to the EXACT pasted-area fraction (clamped box may differ)
    lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    return mixed, perm, lam
```
Notes: `torch.randint`/`torch.randperm` consume the **global** CUDA RNG (seeded `torch.cuda.manual_seed(42)`), so the run stays deterministic-as-seeded — no seed hacking, no new generator needed. The box-clamp + λ-recompute is the standard CutMix `rand_bbox` correction; it guarantees the loss weights match the true pasted area even when the box runs off the image edge.

### 3. Integrate into the training step (replace `train.py:299-303`)
Current:
```python
optimizer.zero_grad(set_to_none=True)
with torch.autocast("cuda", dtype=torch.bfloat16):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
loss.backward()
```
New (CutMix applied AFTER the `.to(device)` move at `train.py:294-297`, BEFORE autocast forward):
```python
optimizer.zero_grad(set_to_none=True)
use_cutmix = (progress < CUTMIX_OFF_TAIL_FRAC) and \
             (CUTMIX_P > 0) and (float(torch.rand(1).item()) < CUTMIX_P)
if use_cutmix:
    mixed, perm, lam = cutmix_batch(inputs, CUTMIX_ALPHA)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(mixed)
        loss = lam * criterion(outputs, targets) \
             + (1.0 - lam) * criterion(outputs, targets[perm])
else:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
loss.backward()
```
`criterion` stays `nn.CrossEntropyLoss(label_smoothing=...)` — see §3 on the LS value. The two-term form calls the SAME criterion twice on ONE `outputs`, so there is exactly **one forward and one backward** per step (no extra forward). `loss.item()` logging (`train.py:317`) is unchanged. EMA update, LR schedule, eval gating all unchanged — CutMix touches only the loss computation.

A 1-line smoke check worth adding to the plan: assert that with `CUTMIX_P=1.0`, `outputs` shape == `[BATCH_SIZE,10]` and `0 ≤ lam ≤ 1`, and that `mixed` differs from `inputs` in exactly the pasted box.

## Hyperparameters + defaults
- **`CUTMIX_ALPHA = 1.0`** — the CutMix-paper default (`Beta(1,1)` = uniform λ). The paper found α=1 optimal across architectures; no reason to deviate on a first test. Lower α (e.g. 0.2) makes λ bimodal near 0/1 (mostly tiny or huge boxes) — that is the *mixup* regime, not what we want.
- **`CUTMIX_P = 0.5`** — apply CutMix on ~half of batches, train on clean images the other half. This is the standard "p=0.5" setting and is deliberately *conservative* given we ALREADY run strong occlusion augmentation (Cutout 12 + RandomErasing). p=0.5 leaves half the batches as clean-image (still Cutout/RE-augmented) updates, hedging against the over-augmentation/under-fit risk (§Risks). If the ep25 trajectory shows headroom (not depressed), p=1.0 is the natural follow-up; if depressed, p=0.25.
- **`CUTMIX_OFF_TAIL_FRAC = 0.85`** — disable CutMix for the final 15% of the budget (§4).

This is a **single conceptual change** (add CutMix) with conservative defaults; I recommend NOT co-tuning anything else this experiment, to keep clean attribution against the 96.38 baseline.

## Interaction with LABEL_SMOOTHING=0.2 (explicit design risk)

CutMix and label smoothing are **both** soft-target regularizers, and stacking the full strength of both risks over-softening the targets (under-confidence → under-fit). But the composition is mathematically well-defined and not double-counting in a harmful way:

- `criterion = CrossEntropyLoss(label_smoothing=s)` applied to a hard target `y` produces the loss against the smoothed distribution `(1-s)·onehot(y) + s/K`. The CutMix two-term loss `λ·CE_s(out,y) + (1-λ)·CE_s(out,y_perm)` is therefore CE against the *mixed-and-then-smoothed* target `λ·[(1-s)e_y + s/K] + (1-λ)·[(1-s)e_{y'} + s/K]` = `(1-s)·[λ e_y + (1-λ) e_{y'}] + s/K`. So LS and CutMix compose linearly and correctly — LS smooths the already-area-mixed two-hot target. There is no bug in stacking them; the question is purely whether the *total* softening is too much.

**Recommendation: hold LABEL_SMOOTHING=0.2 for this experiment** (single-variable attribution against 96.38), but flag LS reduction as the FIRST follow-up if the run reads as under-fit:
- The conservative `p=0.5` already means half the batches see no CutMix softening (they keep LS=0.2 alone, the proven setting). Only the CutMix half stacks both. This bounds the over-softening exposure.
- Reducing LS to 0.1 (or 0.0) *only when CutMix is active* is the cleaner long-run design (the CutMix paper itself does not combine with heavy LS), but introducing it now would confound two changes. Defer it.
- **Falsification hook:** if ep25 is depressed >~0.6pp below EXP-008's 92.31 AND the final lands flat/below 96.38 with normal epochs, the leading hypothesis is over-softening; the pre-registered next move is `LABEL_SMOOTHING 0.2→0.1` (or LS-off-on-CutMix-batches), NOT abandoning CutMix.

This is the single most important interaction to watch and is the proposal's main genuine uncertainty.

## Interaction with EMA + the time-based schedule (tail-disable curriculum)

CutMix injects gradient noise (mixed labels). The validated project pattern is "most accuracy lands in the low-LR tail" (EXP-001) and EMA averages the tail iterates. Training the low-LR tail on *clean* images is a known-good curriculum (airbench and many recipes turn mixing OFF near the end).

**Recommendation: disable CutMix for the final 15% (`progress ≥ 0.85`)** via `CUTMIX_OFF_TAIL_FRAC`. Rationale:
- The EMA window (decay 0.998, warmup from 15%) and flip-TTA gate (final 20%) both concentrate on the tail; feeding them clean-image low-LR updates makes the averaged iterate converge to the *clean* data optimum, not a mixed-label compromise.
- 0.85 (not 0.80) keeps CutMix on for most of the EMA window so the regularization still shapes the bulk of training, while giving a clean final anneal. This is a deliberate middle ground; 0.80 (align with TTA gate) is a reasonable alternative if the tail still looks noisy.
- Cost is zero — it is a scalar comparison on the existing `progress` variable already computed at `train.py:286`.

I weighed leaving CutMix on for the full schedule: it is simpler and is what the CutMix paper does (constant p throughout). But this project has *specific, validated* evidence that the tail is where accuracy is set and that EMA denoises it — so the asymmetric bet (tail-disable) is better motivated here than the paper's constant-p default. If tail-disable underperforms, full-schedule constant-p is the fallback.

## Throughput impact: genuinely free (the key advantage)

- CutMix is a handful of GPU tensor ops: one `randperm`, one `Beta.sample()` (CPU scalar), one slice-copy `mixed[...] = inputs[perm,...]`, and a second `criterion` call on the SAME logits. **No extra forward pass, no extra backward.** The dominant cost (the ResNet-9 forward/backward) is unchanged.
- The slice-copy and clone are on a `[512,3,32,32]` bf16/fp32 tensor — microseconds, negligible vs the ~20ms GPU step. The second `criterion` call is a 512×10 softmax-CE — trivial.
- Unlike CPU-worker augmentation (EXP-008), CutMix is *on-GPU and on-budget*, but its compute is so small it will not measurably cut `num_epochs`. **Pre-registered check: `num_epochs` stays in the ~142–150 band.** If it drops below ~135, something is wrong (e.g. the `.clone()` or a sync) and the attribution is confounded — investigate before trusting the metric.
- Minor honest cost: the `Beta(alpha,alpha).sample()` constructs a distribution object per step; if profiling shows any overhead, replace with a precomputed `torch._sample_dirichlet` or a numpy draw (numpy is an allowed dep). Expected to be irrelevant.

## Expected effect (pp estimate + reasoning)

- **Published headroom is large in absolute terms** (CutMix +0.9pp over baseline, +0.7pp over mixup on ResNet-18/CIFAR-10 @200ep), BUT those baselines (~95.28) had *no* other strong augmentation. Our 96.38 base ALREADY has Cutout 12 + RandomErasing + LS 0.2 + EMA + whitening — much of the "occlusion + soft-target" regularization CutMix provides is partially captured. So the *marginal* gain from adding CutMix on top is smaller than the paper's isolated number.
- **Honest estimate: +0.10 to +0.30pp**, with meaningful probability mass on "sub-noise / flat" because of the saturation + redundancy with existing occlusion aug. The mechanism (real-content region mixing + label mixing) is *complementary* enough to occlusion that I expect a positive central tendency, but I will not claim the full +0.38pp of EXP-008 — that lever was the *first* strong augmentation; CutMix is the *second*, on a higher base, with diminishing returns.
- **Asymmetry argument (why it is still worth a run):** like EXP-008 and unlike the capacity ideas, CutMix is throughput-free, so the worst realistic case is "flat, normal epochs" — it cannot lose by under-annealing. Combined with a credible +0.1–0.3pp central estimate that overlaps the bar, the EV is favorable.

## Pre-registered success / failure read

- **Throughput guard (attribution):** `num_epochs ∈ ~[142,155]`, `total_seconds ≈ 440–450s`, `img/s ≈ 25k`. If `num_epochs < 135`, the result is confounded — fix the CutMix compute cost before judging.
- **ep25 trajectory (under-fit detector):** EXP-008 baseline was ep25 92.31. With `p=0.5` + tail-disable I expect ep25 modestly *below* that (harder-but-not-broken), roughly **91.5–92.3**. 
  - ep25 in [91.5, 92.3] and a tail that overtakes → healthy, mechanism working.
  - ep25 < ~91.5 AND flat/below-baseline final → **over-augmentation/over-softening** signature → next move is reduce `p` to 0.25 and/or `LABEL_SMOOTHING 0.2→0.1` (per §3).
- **Success:** `best_test_acc ≥ 96.48%` (+0.10pp) AND clearly above the ~0.1pp noise floor, with `num_epochs` in the normal band. Per the noise-floor protocol (`03-experiment-learnings.md` High Importance), a single-run +0.05–0.09pp "win" does NOT pass.
- **Null:** `best_test_acc ∈ [96.30, 96.45]` with normal epochs → CutMix is redundant with existing occlusion aug on this saturated base; record as no-improvement, keep the LS-retune + p-tuning follow-ups for a future bundle.
- **Regression / under-fit:** `best_test_acc < 96.25` with normal epochs and a depressed ep25 → total augmentation tipped into under-fit; the de-risking path (lower p, reduce LS, tail-disable earlier) is pre-registered above.

## Strongest risk

**CutMix is redundant with the occlusion augmentation EXP-008 already installed, so its marginal gain is sub-noise on this saturated base** — OR the combined Cutout+RE+CutMix+LS-0.2 tips the net into under-fit (the "total augmentation too aggressive" risk the diagnosis explicitly names). The assumption that most needs to hold: that real-content region mixing + label mixing is a *sufficiently complementary* regularizer to single-image occlusion that it shifts the fully-annealed optimum by >0.1pp, rather than just re-occluding regions the net is already robust to. The conservative `p=0.5`, the tail-disable, and the held LS-0.2 are all chosen to keep the downside to "flat," and the ep25 detector + pre-registered de-risking (lower p, reduce LS) give a clear read on which failure mode (redundant vs over-aggressive) occurred.

## Effort

**Low–medium.** One module-level helper (~12 lines), three new constants, and a ~10-line edit to the training step (`train.py:299-303`); no architecture/schedule/dependency change. One 300s run plus a smoke check. Slightly more than EXP-008's 2-line edit because of the helper + loss-branch, but well within one experiment loop. The main *intellectual* effort is reading the ep25/tail signature correctly to pick the right follow-up (p vs LS), which is pre-registered.

## Evidence pointers
- **CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features**, Yun et al., ICCV 2019, arXiv:1905.04899 — the `rand_bbox` area-correction, the two-term area-weighted loss, α=1.0 default, p~0.5; the core claim that pasting *real content* beats Cutout's zeros.
- **Short-schedule benchmark evidence** (the load-bearing "CutMix not mixup" argument): TransformMix (arXiv:2403.12429) Table — ResNet-18/CIFAR-10: baseline 95.28, mixup 95.55, **CutMix 96.22**; OpenMixup CIFAR CutMix-protocol benchmarks (200/400/800/1200 ep). Mixup Without Hesitation (arXiv:2101.04342): plain mixup needs very long schedules to pay off; CutMix converges fast — fits our ~150-epoch budget.
- **EXP-008** (`experiments/008/04-analysis.md`): throughput-free augmentation = +0.38pp, the largest lever; establishes the regularization-bound + epoch-surplus operating point and the "watch ep25/tail" protocol this proposal reuses. The Unexplored-Avenues there explicitly names adding a *second complementary augmentation* as the high-value next step.
- **EXP-005/007 under-anneal entry** (`03-experiment-learnings.md` Medium): motivates choosing a throughput-FREE lever (CutMix's GPU cost ~0) over capacity adds.
- **EXP-009/010 Muon tie** (`03-experiment-learnings.md` Low): "regularization-bound, not optimizer-bound" — directs effort to the regularization axis, which CutMix is on.
- **EXP-001 tail Pattern + noise-floor Protocol Finding** (`03-experiment-learnings.md`): motivates the tail-disable curriculum and sets the ≥0.10pp-above-noise success bar used in the read.

## Out-of-scope confirmation
Edits only `train.py`; `prepare.py` untouched; no new dependencies (pure torch; numpy only as an optional micro-optimization, already allowed); seeds `torch.manual_seed(42)`/`torch.cuda.manual_seed(42)` unchanged (CutMix RNG draws from the seeded global CUDA generator); ≤1 `evaluator.evaluate` per epoch unchanged; runs on GPU 1 (`CUDA_VISIBLE_DEVICES=1`).
