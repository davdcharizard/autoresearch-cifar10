# Brainstorm EXP-041
**Created**: 2026-06-10
**Goal**: goals/maximize-cifar10-test-accuracy.md

## Web Search & Literature Review

- No new external searches. Source for the lead candidate is already in the knowledge base: **airbench** (knowledge README row) — Keller Jordan's CIFAR-10 speedrun lineage introduced *alternating flip*: instead of iid RandomHorizontalFlip, each image is shown flipped/unflipped deterministically in alternating epochs, guaranteeing exactly balanced orientation coverage. Published as a measurable improvement over iid flip in the 10–40-epoch regime, where flip-SAMPLING variance is a real error term.
- Coverage arithmetic for OUR regime (139 epochs): under iid flip each image's flipped fraction has σ = √(0.25/139) ≈ 4.2% around 50% — small but nonzero; alternation sets it to exactly 0. The candidate's effect is the value of removing that residual — honestly small, but it is the last unmeasured mechanism CLASS (data order/coverage) and the implementation is zero-cost in every closed currency.
- Note: TrivialAugmentWide's op set contains no mirror op, so horizontal orientation coverage is governed solely by the flip stage — the alternation actually controls what it claims to control.

## Experimental History Review

- 42 experiments, 6 improvements (last EXP-006); baseline 96.71 @ 1990397 (mean ≈96.57, σ ≈0.16); bar 96.81; 35 consecutive non-improvements.
- **EXP-040 closed capacity at the hardware level** (kernel cliff above 256 channels; 4× sits at the cliff edge). Per its Next Steps, remaining brainstorm space = data composition/order and objective shaping, with the instruction that candidates there must be ZERO-COST so expected nulls are free.
- **Data order/coverage is the only mechanism class with no measurement at all.** Everything else: recipe constants audit-complete (explicit EXP-036, implicit EXP-038/039 two-sided); regularization/augmentation dose-response peaked both sides; schedule/optimizer/noise bracketed; init both directions; architecture closed in every currency; eval-constants closed; weight averaging closed; throughput at numerics floor.
- Prior screenings of this candidate (brainstorm-039/040) rejected it on implementation risk: the naive virtual-100k-dataset design halves eval count and breaks every signature check, and per-epoch state doesn't propagate to persistent DataLoader workers. NEW: a shared-memory epoch tensor (`torch.zeros(1).share_memory_()`, written by the main process, read by forked workers) removes both objections — epoch semantics stay byte-identical (139 epochs, 139 evals, same loader shape), making the experiment signature-comparable to family.
- Recognized-and-rejected hack (recorded for the screen): deliberately raising EVAL-draw variance to harvest the best-over-checkpoints max (EXP-039 showed 3× spread with flat mean) does not improve the model and would not survive a benchmark-composition change — reward hacking; never propose.
- Screens binding: zero-cost requirement (this candidate: zero dt — flip is a PIL transpose already paid by RandomHorizontalFlip ~50% of the time; zero heat/noise/numerics change — same op, deterministic schedule), absorption law (mechanism is coverage variance, not a regularizer — orthogonal to augmentation strength), σ-arithmetic (honest expected ≈ 0–0.1).

## Candidate Ideas

### 1. Derandomized alternating horizontal flip via shared-memory epoch tensor (zero-cost, preserves epoch semantics)
**Summary**: Remove `RandomHorizontalFlip` from the transform stack; subclass `datasets.CIFAR10` with `__getitem__(i)` that flips the PIL image iff `(epoch + i) % 2 == 0`, where `epoch` is read from a shared-memory int64 tensor the main loop updates at each epoch top. Every image is seen in both orientations exactly equally (alternating per epoch); everything else — crop, TA, RE, loader shape, epoch/eval count — byte-identical.

**Reasoning**: The last unmeasured mechanism class, with published in-lineage evidence (airbench) and a mechanism that is genuinely orthogonal to every closed axis: it changes neither augmentation STRENGTH (same flip marginal, 50%) nor gradient noise statistics (per-batch flip composition variance is unchanged at batch 512: alternation fixes per-IMAGE coverage across epochs, not within-batch mixture) — it removes only the per-image orientation-coverage variance accumulated over the run (σ≈4.2% per image under iid). The honest effect estimate is +0.0–0.1 (the mechanism's value shrinks ~√epochs vs the 10–40-epoch anchors); the bar-pass branch requires residual coverage imbalance to matter more than coverage arithmetic suggests. The null is free (zero cost in all currencies) and closes the data-order class measured, completing the program's coverage of mechanism classes. Implementation risks priced: persistent-workers state propagation solved by `share_memory_()` (forked workers share the tensor's storage); iterator-recreation per epoch means the increment lands before any fetch of the new epoch; `from PIL import Image` (or `transforms.functional.hflip`) adds no dependency (PIL ships with torchvision).

**Sources**: knowledge README airbench row; coverage arithmetic above; EXP-027 σ-calibration (reports/exp-report-027.md).

**Estimated Effort**: low-medium — ~15-line Dataset subclass + 2-line loop change; CPU sanity must verify the flip schedule (same image at consecutive epochs differs by mirror) and worker propagation (a smoke iteration with num_workers>0).

**Risk Assessment**: Zero closed-currency cost; signature checks unchanged (dt 22.3–22.4, 139 ep, params 4,286,026). Failure modes: (a) shared-tensor read in workers silently stale → flip schedule frozen at epoch 0 → degenerates to a FIXED half/half assignment (still balanced within-epoch, coverage per-image broken) — detectable in CPU sanity before launch; (b) effect ≈ 0 → free null, class closed.

### 2. Per-channel input std normalization (std (1,1,1) → (0.2470, 0.2435, 0.2616))
**Summary**: The one input-pipeline constant never dosed: the recipe subtracts the per-band mean but divides by 1 (original-paper fidelity, per the README note); modern recipes divide by per-channel std.

**Reasoning**: Predicted-null by an in-regime MEASURED mechanism: EXP-019 showed bn1 immediately renormalizes the stem (whitening washed out at −0.26 from its dt/init tolls; pure rescaling has no toll but also nothing to contribute — input scale is absorbed into bn1's statistics within ~10 batches at m=0.1). A ~3× input rescale could microscopically shift conv1's effective LR/WD balance, but that's sub-noise by the EXP-015 ±0.1 dead-zone. Free but information-poor: the null is already written by EXP-019's mechanism.

**Sources**: train.py L130–133 + README note; exp-report-019.md (stem renormalization).

**Estimated Effort**: trivial.

**Risk Assessment**: Safe; expected 0.0 with a measured mechanism already predicting it — fails the "genuinely unmeasured" bar that Candidate 1 passes.

### 3. Sigmoid/BCE objective replacing CE + label smoothing
**Summary**: Multi-label BCE-with-logits head (big-vision style), zero-dt.

**Reasoning**: Objective shaping is a named open area, but the evidence is ImageNet-scale, fixed-epoch, lighter-aug — the absorption law's exact kill profile (0-for-14 external transfer; EXP-036 measured the loss-target axis FLAT under TA+RE). Also changes the loss SCALE seen by the LS-calibrated heat constants — not provably zero-cost in heat.

**Sources**: exp-report-036.md; project-insights absorption entry.

**Estimated Effort**: low.

**Risk Assessment**: Fails the in-regime screen AND the strictly-zero-cost requirement (heat coupling); predicted-null with a confound — worst of the three.

## Idea Evaluation

Evidence strength: Candidate 1 has in-lineage published evidence (airbench's alternating flip is a documented win in its regime) and — decisively — is the only candidate testing an UNMEASURED mechanism class; Candidate 2's null is already predicted by a measured in-regime mechanism (EXP-019), and Candidate 3 carries the absorption law's full 0-for-14 discount plus a heat confound. Mechanism clarity: Candidate 1's is exact and quantitative (remove per-image orientation-coverage variance, σ≈4.2% → 0, all else equal); the others are vague-or-predicted-zero. Expected impact: all three are honestly sub-bar in expectation; Candidate 1 alone buys class-closure information with its null. Risk: Candidate 1's single real risk (worker state staleness) is detectable in CPU sanity before any GPU time; its run is signature-identical to family so every standing integrity check applies unchanged. Feasibility: ~15 lines with a pre-launch propagation test. Candidate 1 selected.

## Chosen Idea
**Selected**: Derandomized alternating horizontal flip via shared-memory epoch tensor (Candidate 1)

**Why this idea**:
It is the last mechanism class with no measurement, the only candidate whose evidence comes from the speedrun lineage measuring THIS exact intervention (rather than an analogous one), and the shared-memory design removes the implementation objections that screened it out in the last two brainstorms — at exactly zero cost in every closed currency, so the expected null is free and closes the class.

**Hypothesis**:
Replacing iid flip with deterministic per-epoch alternation removes per-image orientation-coverage variance (σ≈4.2% → 0) at byte-identical execution signatures (dt 22.3–22.4ms, 139 epochs/139 evals, params 4,286,026), raising the converged plateau if residual coverage imbalance is a real error term at 139 epochs — bar-pass (≥96.81) only in that branch. Falsified by a plateau in the baseline band (96.4–96.7) → coverage variance is immaterial at this epoch count, the data-order class closes measured, and the program's mechanism-class coverage is complete. Diagnostics: ep5/10/20 (early coverage benefit should appear FIRST in the low-epoch regime where airbench operates), last-15 plateau mean/spread, final_test_loss.
