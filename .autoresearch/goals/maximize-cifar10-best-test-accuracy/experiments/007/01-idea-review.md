I have complete context. Emitting the review.

---

# Adversarial Review — EXP-007 (CIFAR-10, best_test_acc)

**Baseline:** 92.30% @ `11f8469` · improvement bar **≥ 92.40%** (+0.10) · fixed **300 s** counted budget, ≤600 s wall · one H20 · `train.py`-only · no seed hacking.

**Finalists reviewed:** Width-2 ResNet-20 (`idea-01`), Plateau CutMix + weak tail (`idea-02`), Isolated weight decay 5e-4 (`idea-04`). Batch-256 (`idea-03`) read as context — it is a self-declared preflight no-go (23,006 projected updates = 60% of EXP-004; fails its own update/sample gates) and is correctly excluded.

I verified the load-bearing numbers before judging: `TIME_BUDGET_S=300` (`prepare.py:7`); EXP-004 strong-switch 84.60% → first weak 91.43% → **peak 92.30% at epoch 98** (2nd-to-last), 38,358 steps, 25 evals (`004/04-analysis.md:23`); EXP-006 Cutout 83.15%/90.47%/91.63% (`006/04-analysis.md:23`); once-per-tail-epoch eval cadence (`train.py:264`); and idea-01's `1,073,962` param count, which I recomputed component-by-component and confirms **exactly** — a good signal of that proposal's rigor.

---

## Prioritized feedback (most important first)

### 1. [Cross-cutting, affects the whole diagnosis] The "capacity is the leading limiter" premise is under-evidenced and partly contradicted by this goal's own history.
The brainstorm diagnosis (`01-brainstorm.md:20`) and idea-01's core motivation rest on "capacity is the leading untested limiter." But EXP-001's learning records **train loss reaching 0.0215** — near-zero — meaning the 0.27M model has ample capacity to *memorize* CIFAR-10 (`03-experiment-learnings.md:72`). That points to a **generalization/regularization** limiter, not a raw-capacity one. No in-repo evidence shows the model underfitting *clean* data. This does not kill idea-01 (train loss under N1/M7 is unmeasured, and the 84.60% augmented checkpoint could reflect capacity strain), but it means the diagnosis is a hypothesis, not an established fact — and it directly reweights the field toward the regularization lever (idea-04).
- **Fix:** Before planning idea-01, read EXP-004's train-loss EMA under N1/M7 from its log/report. If train loss under strong aug is still high, the capacity story holds; if the clean limiter is generalization, width alone is unlikely to move it. Elevate the underfit-vs-overfit train-loss read from a decision-rule footnote (`idea-01.md:281`) to a pre-registered primary diagnostic.

### 2. [idea-02, near-fatal] Always-on α=1 CutMix *stacked on* N1/M7 re-runs the failure mode EXP-006 diagnosed, and the proposal rebuts only half of it.
EXP-006's analysis gives two mechanisms for the −0.67 loss: (a) 25% information deletion, **and** (b) the plateau augmentation being "too aggressive as an every-view replacement," leaving the weak tail a weaker representation it could not recover in ~20 epochs (`006/04-analysis.md:24`). idea-02 answers only (a) — pixels are retained — but its intervention is **strictly more augmentation than EXP-004** (RandAugment *then* batch-level CutMix on every strong batch), which pushes hard on mechanism (b). Beta(1,1) averages ~50% of each image replaced. On a 0.27M model over ~95 epochs this is a real over-regularization / underfit risk, and EXP-006 already showed a ~20-epoch weak tail cannot close a representation deficit of this kind. The proposal's own failure-mode #1 concedes this but treats the 60 s tail as sufficient mitigation on no evidence.
- **Fix:** If pursued, do not run always-on α=1. Predeclare a lower α (e.g. 0.2–0.5) or a per-batch application probability p<1 so the plateau isn't maximally distorted every step — but note that adds an unvalidated strength knob, which is exactly why this idea is a poor fit for a single-run slot right now. Cleaner: defer CutMix until after an isolated capacity or regularization result narrows the diagnosis.

### 3. [idea-01, key design choice] Width-2 sacrifices ~31% of updates **and** ~30% of the tail epochs in the exact region where EXP-004's peak appeared — the case for width-2 over width-1.5 is asserted, not closed.
EXP-004 peaked at **epoch 98 of 99** (`004/04-analysis.md:23`) — the very end of a ~20-epoch weak tail. Width-2 projects ~13–14 tail epochs (`idea-01.md:161`). Because eval is once-per-tail-epoch (`train.py:264`), fewer tail epochs mean both fewer refinement steps *and* fewer best-metric sampling opportunities near where the peak historically lands — double jeopardy the proposal under-weights (it lists "fewer eval opportunities" only as a minor confound, `idea-01.md:210`). Width-1.5 retains ~31.2k calibrated steps (81% exposure, `idea-01.md:178`) and still 2.24× params. The proposal's defense of width-2 (`idea-01.md:180`) is reasonable — it tests the capacity hypothesis more decisively and 3.98× params cost only 1.44× step-time — but it doesn't quantify the tail-epoch/peak-sampling risk.
- **Fix:** Either (a) commit to width-2 but pre-register that the analysis reads the *tail trajectory shape* (is best still rising at the final epoch? → under-optimized, route to width-1.5) rather than only the scalar best; or (b) reconsider width-1.5 as the lower-variance first bet given the peak-at-end pattern. The proposal already routes a clean optimization-lag failure to width-1.5 (`idea-01.md:283`) — good, but the peak-sampling risk should be named explicitly as a distinct mechanism from raw update loss.

### 4. [idea-01, thin margin] The calibrated 26,563-step projection clears the self-imposed 26,000 floor by only ~2%.
`38,358 × 7.515/10.852 ≈ 26,563` (`idea-01.md:157`) vs a 26,000 gate (`idea-01.md:168`). GPU-clock drift, allocator state, or a slightly higher real per-step overhead than the synthetic microbenchmark could push the real run under the floor. Mitigating factor in idea-01's favor: at width-2 the run is **GPU-bound** (GPU ~92 b/s vs EXP-004 strong loader 165–175 b/s, `idea-01.md:170`), so the synthetic step-time benchmark is representative here — more so than for idea-02, whose timing depends on worker-side batch cloning.
- **Fix:** Re-run the fresh-process microbenchmark at plan time and predeclare the decision if real steps land in 26,000–26,500 (proposal says compare against both projections — make the borderline action explicit, not just "report").

### 5. [idea-01 ↔ idea-04 entanglement] Holding weight decay at 1e-4 for a 4× model may confound the width test.
idea-01 keeps `1e-4` for attribution (`idea-01.md:207`) but lists "unchanged decay may under-regularize" as a failure mode. If the true optimum is "more capacity + more decay," width-2-alone can fail while width-2+5e-4 succeeds — meaning a width-2 no-improvement would *not* cleanly falsify the capacity hypothesis. The proposal handles this correctly by routing an overfit-signature failure to a separate decay review (`idea-01.md:284`) rather than bundling — good isolation discipline — but planning should treat idea-01 and idea-04 as a **two-step sequence**, not independent bets.
- **Fix:** Pre-register the branch: width-2 no-improvement *with overfit signature* → width-2 + 5e-4 as the immediate follow-up; *with underfit/optimization-lag* → width-1.5.

### 6. [idea-04, ceiling] Cleanest attribution, but the upside is capped near the decision threshold and the negative mechanism is as plausible as the positive one.
The optimistic band is 92.40–92.60% (`idea-04.md:97`) — success is only +0.10 to +0.30, and the proposal itself rates a 0.2–0.5 regression "plausible" from underfit. Applying 5e-4 to **BN affine + bias** with no param groups (`idea-04.md:54`), across the long high-LR plateau, is the specific under-fit risk on a 0.27M model. This is the sound, low-ceiling, compute-neutral bet — genuinely valuable if the diagnosis is "generalization-limited" (see #1), but it cannot be the high-upside play.
- **Fix:** None required for correctness — it's the most controlled idea in the batch. If selected, pre-register reading the strong-switch checkpoint vs EXP-004's 84.60% as the underfit tell (proposal already does, `idea-04.md:100`).

### 7. [idea-02, unproven assumption — but gated] The `fork_rng(devices=[])` RNG-isolation argument is sound *only if* v2.CutMix draws exclusively from the CPU torch global RNG.
The reasoning (`idea-02.md:159`) is actually correct: per-sample crop/flip/RandAugment run before `collate_fn`, so CutMix samples from a batch-dependent state and restoring it keeps the RandAugment stream identical to EXP-004 while still varying CutMix params. That's a well-designed control. The residual risk is that if torchvision's CutMix touches any RNG source *not* captured by `fork_rng(devices=[])` (a separate `Generator`, numpy), isolation silently breaks. The proposal correctly gates this with a mandatory before/after RNG-state-equality preflight (`idea-02.md:265`) — so it's a handled risk, not an open one. Credit where due: this is the most careful confound control in the three proposals.

### 8. [Minor, all three] Single-run threshold sensitivity is acknowledged uniformly and correctly.
All three note +0.10 = ten test images and forbid rerolls (`idea-01.md:222`, `idea-02.md:408`, `idea-04.md:114`). No action; noted so it isn't mistaken for an omission.

---

## Scored verdict

| Idea | Evidence & reasoning | Potential impact |
|---|---|---|
| **Width-2 ResNet-20** (`idea-01`) | **4/5** — rigorous timing preflight, exact param count verified, honest confounds, WRN literature; docked because the capacity diagnosis is under-evidenced (feedback #1) and the width-2-vs-1.5 tail-epoch risk is under-weighted (#3). | **4.5/5** — clearly the highest ceiling; directly attacks the diagnosed bottleneck, 3.98× capacity for 1.44× step-time, plausible +0.2–0.7 with a path toward published wide-CIFAR territory. |
| **Weight decay 5e-4** (`idea-04`) | **4/5** — maximally attributable one-literal change, compute-neutral, canonical value; but transfer is directional (WRN used it with wider models + 2× epochs) and the underfit mechanism is equally plausible. | **2/5** — low ceiling capped near the threshold (+0.1–0.3), realistic chance of a small regression. |
| **Plateau CutMix** (`idea-02`) | **3/5** — excellent RNG/lifecycle engineering and throughput gating, but the core bet rebuts only half of EXP-006's failure and under-weights the "too aggressive every-view" mechanism (#2). | **2.5/5** — modest self-estimated ceiling (+0.1–0.4) paired with the highest implementation and over-regularization risk; poor risk/reward for a single run. |

### Winner: **Width-2 ResNet-20 (`idea-01`)**

It wins on the two criteria that matter: it has the **highest ceiling backed by the strongest mechanistic + literature argument** (WRN width→CIFAR gains, quadratic capacity for near-linear H20 cost), and its feasibility is **measured, not assumed** — the exact param count checks out, the run is GPU-bound so the microbenchmark is representative, and the exposure trade is quantified and gated rather than hand-waved. Per the judging rule I decline to risk-discount it into irrelevance for being exploratory: it attacks the diagnosed limiter head-on with a real upside, where idea-04 is a low-ceiling safe bet and idea-02 is a high-risk/low-reward re-approach of a pattern this goal's history already penalized.

It wins **conditionally** — the capacity diagnosis (#1) is the load-bearing assumption and is only partially supported. If the plan-time train-loss read shows the model is generalization-limited rather than capacity-limited, the correct pick flips to idea-04.

### Required refinements before planning (in order)
1. **Test the premise first.** Read EXP-004's train-loss-under-N1/M7 trajectory; only proceed with width-2 if it indicates capacity strain rather than easy fit (feedback #1). If it indicates generalization-limited, switch to idea-04.
2. **Pre-register the failure routing** as a two-step sequence: overfit-signature no-improvement → width-2 + 5e-4; optimization-lag / best-still-rising-at-final-epoch → width-1.5 (feedback #3, #5).
3. **Elevate the tail-shape / peak-sampling analysis** to a primary readout, not a scalar-best comparison, given EXP-004 peaked at the penultimate epoch and width-2 loses ~30% of tail epochs (#3).
4. **Re-run the fresh-process microbenchmark at plan time** and predeclare the action for a borderline 26,000–26,500-step landing (#4).
5. Keep the isolation discipline as written — `1e-4` decay unchanged, single width-only diff, no dropout/decay bundling — so the result is attributable.
