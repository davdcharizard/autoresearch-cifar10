# Brainstorm EXP-003
**Created**: 2026-06-08
**Goal**: goals/improve-cifar10-test-accuracy.md

## Web Search & Literature Review
No new external sources needed — this loop is grounded in our own EXP-002 finding plus standard practice:

- **EXP-002 (current best, reports/exp-report-002.md)**: Cutout(16) improved acc to 95.42% but a per-sample
  `torch.randint().item()` in the CPU dataloader workers throttled throughput (79→54 epochs, ~9.8→14 ms/step).
- **General practice**: data augmentation that is cheap/vectorized (or applied on-GPU per batch) avoids
  dataloader CPU bottlenecks; `.item()` calls force host-device syncs and per-sample Python overhead. Batched
  tensor masking on the GPU is near-free for a 32×32 image and a tiny model.
- **Project-insights (Med)**: bf16 + channels_last default; (High) the binding budget is the 300s wall-clock —
  so recovering lost throughput directly buys more epochs.

## Experimental History Review
Source: experiment-indices/improve-cifar10-test-accuracy.tsv, goal-learnings, project-insights, exp-reports.

- **Current best / baseline**: **95.42%** (EXP-002, commit edf15d3).
- **Trajectory**: 91.73 (BASE) → 92.06 (EXP-000 recipe) → 94.90 (EXP-001 widen k=4) → 95.42 (EXP-002 Cutout).
- **Goal-learnings**: (High) widening is the dominant lever and nearly free on H20; (High) Cutout(16)
  regularizes the wide model (+0.52, overfit loss 0.25→0.22); (Low) **per-sample `torch.randint().item()` in
  CPU dataloader transforms throttles throughput ~30%** — vectorize/GPU-side to recover epochs & compound gains.
- **Project-insights**: VRAM essentially free (490 MB / 98 GB); bf16+channels_last default; 300s wall-clock binds.
- **Untried gaps**: efficient/GPU Cutout, Cutout strength tuning, WD 5e-4, more width (k=6), mixup, deeper-wider mix.
- **No failed approaches** — all three experiments improved.

## Candidate Ideas
First principles: EXP-002 reached 95.42% with only 54 epochs because its Cutout implementation became the
dataloader bottleneck. The single clearest opportunity is to **remove that artifact** — the same regularized
model trained for the ~79 epochs that the budget actually allows should score higher. This also permanently
speeds every future experiment. All ideas edit only train.py and keep the k=4 model + recipe + Cutout(16).

### 1. Vectorized GPU Cutout (recover throughput)
**Summary**: Remove Cutout from the CPU `train_tf` pipeline and instead apply it as a **batched GPU operation**
inside the training loop, right after `inputs` are moved to device (before autocast forward). For a batch of B
images, sample B random centers with a single `torch.randint` on-GPU, build a `(B,1,H,W)` mask via broadcasted
`arange` comparisons, and multiply (zero the 16×16 window per image). No per-sample Python loop, no `.item()`
host-sync. Semantics identical to EXP-002 (one random 16×16 hole per image per step); only the *where/how* changes.

**Reasoning**: EXP-002 proved the regularization helps even at 54 epochs; the throughput loss was a pure
implementation artifact (goal-learnings Low entry). A vectorized GPU mask is near-zero cost on a 32×32 tensor,
so the run should recover to ~79 epochs (EXP-001 throughput) → more training of the regularized model → higher
acc. Bonus: the speedup carries into all later experiments.

**Sources**: reports/exp-report-002.md § Execution/Analysis; goal-learnings (Cutout throughput entry); project-insights (300s binds).

**Estimated Effort**: low–medium (remove the transform; add a ~6-line batched-mask op in the loop).

**Risk Assessment**: Low. Risk is a masking bug (wrong shape/broadcast) — caught by a quick sanity check and the
abort criteria (NaN / no improvement). Worst case: graceful no-improvement. Eval untouched (Cutout train-only).

### 2. Add weight decay 5e-4 (WRN standard)
**Summary**: Raise WEIGHT_DECAY 1e-4 → 5e-4 on the current k=4+Cutout model.

**Reasoning**: WRN-standard decay; another regularizer for the high-capacity model.

**Sources**: WRN (1605.07146); goal-learnings.

**Estimated Effort**: low (one constant).

**Risk Assessment**: Low–medium. At only 54 epochs (current throughput) extra regularization risks
under-fitting; better attempted *after* throughput is restored (Idea 1), and ideally as its own isolated test.

### 3. Push width to k=6
**Summary**: WIDTH_MULT 4→6 ({96,192,384}, ~9.7M params) with Cutout + recipe.

**Reasoning**: More capacity; VRAM still free.

**Sources**: WRN; reports/exp-report-001.md.

**Estimated Effort**: low.

**Risk Assessment**: Medium. k=6 grows FLOPs ~2.25× → even fewer epochs (the Cutout bottleneck still present),
likely under-trained. Much more attractive *after* Idea 1 restores throughput.

## Idea Evaluation
All respect hard constraints (train.py only, no deps, single GPU/300s, eval once/epoch, no seed hacking).

- **Evidence strength**: Idea 1 is backed by our own measured result — the regularization already works; only
  throughput was lost, for an identified, fixable reason. Strongest, most direct evidence.
- **Mechanism clarity**: Idea 1 — restore ~25 epochs of training of an already-better model → higher acc; crisp
  and low-risk. Ideas 2/3 add more regularization/capacity but are handicapped by the *current* throttled
  throughput, so they'd be tested under worse conditions than necessary.
- **Expected impact**: Idea 1 highest and also unblocks 2/3 (which become more attractive once epochs are back).
- **Risk profile**: Idea 1 fails gracefully; 2/3 risk under-fit at the throttled budget.
- **Feasibility**: all low-effort; Idea 1's combination of high EV + unblocking-future-work dominates.

Idea 1 (vectorized GPU Cutout) wins decisively: it directly converts a known wasted ~30% of compute back into
training, compounding the proven Cutout gain, and permanently speeds the workstream. WD tuning (2) and more
width (3) are the natural next loops once throughput is restored.

## Chosen Idea
**Selected**: Idea 1 — Vectorized GPU Cutout (move Cutout to a batched on-device op in the training loop).

**Why this idea**:
EXP-002 established that Cutout regularization helps (95.42% at just 54 epochs) but its per-sample CPU
implementation throttled throughput by ~30%. Replacing it with a near-zero-cost batched GPU mask should restore
the ~79 epochs the 300s budget actually allows, giving the already-better regularized model more training — a
high-confidence gain that also accelerates every subsequent experiment. Edits only train.py; eval untouched.

**Hypothesis**:
Applying Cutout as a vectorized GPU batch operation (instead of a per-sample CPU transform) will restore
throughput to ~79 epochs in the 300s budget and raise best_test_acc above the 95.42% baseline (expected
~+0.2–0.6 pp into the ~95.6–96% range), with identical Cutout semantics, completing cleanly in budget. If
the extra epochs don't help (regularized model already saturated at 54 epochs), the result lands flat
(no-improvement) but still permanently speeds future runs.
