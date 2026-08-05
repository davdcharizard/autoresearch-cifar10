# Brainstorm EXP-003
**Created**: 2026-06-28

## Web Search & Literature Review

- **airbench / "94% on CIFAR-10 in 3.29 Seconds" (Keller Jordan 2024, arXiv:2404.00498)** (https://arxiv.org/abs/2404.00498; repo https://github.com/KellerJordan/cifar10-airbench): The fast-CIFAR lineage (David Page 2019 → tysam-code hlb-CIFAR10 → airbench). Key levers, mostly **additive**: (1) **frozen patch-whitening first conv** — the foundational first layer in every variant, "biggest convergence accelerator"; (2) **identity initialization** pairs with whitening (reduces epochs-to-94%); (3) **alternating-flip augmentation** — deterministic per-image flip alternating each epoch, "equivalent to 0–25% training speedup", the final ~10% of airbench's speedup; (4) **multi-crop TTA** — the one non-additive trick; (5) current record uses the **Muon optimizer** (`airbench94_muon.py`, 94% in 2.59s) and data filtering for 96%. Whitening construction: `eigh` of patch covariance, scale eigvecs by `1/sqrt(eigval+eps)`, concat `(V,−V)`. Caveat: airbench wins are measured in epoch-STARVED regimes (≤10–40 epochs); our 300s budget fits ~183 epochs, compressing the marginal return of speed-oriented tricks.
- **hlb-CIFAR10 (tysam-code)** (https://github.com/tysam-code/hlb-CIFAR10): reproduction of Page's recipe; whitening "maps input to a nicely distributed sphere"; ~95.79% in ~110s — the regime closest to ours.
- **WideResNet (Zagoruyko & Komodakis 2016)**: for CIFAR, *widening* a shallow residual net is a more compute-efficient capacity lever than deepening — supports the width-multiplier idea.
- Distilled reference already in-repo: `knowledge/references/fast-cifar10-recipes.md`.

## Experimental History Review

Sources: `04-results.tsv`, `project-notes/project-insights.md`, `03-experiment-learnings.md`.
- **BASE** (1f69af5, 91.57%): ResNet-20 + MultiStepLR — under-annealed (schedule never reached 2nd LR drop in budget).
- **EXP-001** (26fdc83, **95.22%**, improvement): swapped to DavidNet/ResNet-9 + time-based completing one-cycle, bf16+channels_last, Cutout+label-smoothing, batch 512. +3.65pp. Learning: most accuracy arrives in the **low-LR tail of a completing one-cycle**.
- **EXP-002** (c404104, **95.72%**, improvement): weight EMA(0.998, use_buffers) evaluated in place of raw iterate + flip-TTA gated to final 20%. +0.50pp, ~free, orthogonal/eval-side. Learning: eval-side denoising (tail-noise/robustness limiter) now largely **spent**.
- **Current best / baseline**: **95.72%**; bar **≥95.82%**.
- **What worked**: DavidNet + completing one-cycle (architecture+schedule); EMA+TTA (eval-side). Tricks compose **additively** on the DavidNet base.
- **Untried gaps** (flagged by EXP-001/002 Next Steps + idea-review item 8): **whitening front-end**, **capacity (wider/deeper)**, optimizer change. VRAM 1.6/98 GB and ~30k img/s throughput leave huge headroom (project-insight High).
- **Diagnosis (what limits the objective now):** At 95.72% with a fully-annealing one-cycle and eval-side denoising already in place, the remaining limiters are (a) **optimization conditioning / convergence efficiency per unit time** (whitening, Muon attack this), and (b) **representational capacity** (width attacks this). Augmentation-coverage levers (alternating flip) are largely averaged out at 183 epochs. This is NOT a profilable-bottleneck goal; the diagnosis is reasoned from history + the fast-CIFAR literature, not a cost decomposition.

## Collected Ideas

- Frozen patch-whitening first conv (decorrelate input → better-conditioned stem). [outside technique]
- Identity-init learnable stem paired with whitening (airbench). [outside technique]
- Alternating-flip augmentation replacing random flip. [outside technique — DROPPED, see below]
- Wider DavidNet (1.25–1.5× channels) to add capacity. [capacity/orthogonal lever]
- Deeper DavidNet (add the missing layer2 residual block). [algorithm/representation]
- Muon optimizer (orthogonalized momentum) for conv/Linear weights. [moonshot / outside technique]
- Raise PEAK_LR now that EMA+TTA stabilize the tail. [simplification/sweep-adjacent]
- Multi-crop TTA (extend eval-side TTA beyond flip). [orthogonal lever — eval-side, mostly spent direction]
- EMA-decay / TTA-gate micro-sweep on the new 95.72% base. [exploitation/sweep]

## Combinations

- **Whitening + identity-init stem**: airbench pairs them; whitening conditions the input, identity init keeps the learnable stem from destroying that conditioning early — additive epochs-to-target reduction.
- **Whitening + raised PEAK_LR**: a better-conditioned input tolerates larger steps, so whitening could unlock a higher peak LR for a compounding gain (deferred — adds a tuning knob, kept out of the clean A/B).
- **Wider net + Muon**: Muon's per-direction conditioning matters more as matrices get larger — but stacking two unproven levers confounds attribution; tested separately first.

## Candidate Ideas

### 1. Frozen patch-whitening first convolution
**Summary**: Prepend a fixed, non-learnable 3×3/pad-1 ZCA-whitening conv (eigendecomposition of CIFAR-10 training-patch covariance, computed once at startup off the 300s timer, in the exact eval normalization space) in front of the existing `prep` conv, feeding the learnable stem decorrelated 54-channel input (concat of ±eigenvectors). Kernel 3/pad 1 preserves 32×32 so the pooling chain is untouched. The whitening conv is frozen (`requires_grad=False`) and **excluded from the SGD param list** so weight decay can't corrupt it; loaded after `.apply()` so kaiming-init doesn't overwrite it.

**What it targets**: Optimization conditioning of the learnable stem. Raw mean-subtracted RGB has a highly anisotropic 27-dim patch covariance (R/G/B + spatial-DC correlation); decorrelating it sphereizes the first layer's loss surface so SGD makes useful progress earlier, reaching a marginally better tail minimum within the same budget. Orthogonal to EXP-002's eval-side EMA+TTA (kept ON).

**Reasoning**: The foundational airbench/hlb/Page front-end, named the single biggest convergence accelerator; additive with the rest of the recipe; the existing `prep=conv_bn(3,64)` is exactly the layer airbench replaces, and `prep`'s BN self-calibrates the whitened scale (no hand-tuning). See `proposals/idea-01.md`.

**Sources**: `proposals/idea-01.md`; arXiv:2404.00498; hlb-CIFAR10; `knowledge/references/fast-cifar10-recipes.md` (lines 8,15).

**Estimated Effort**: Low (~30–40 lines, one builder + ~5 lines in ResNet9 + ~6 in main; no schedule/budget change).

**Risk Assessment**: Correctness hazards all enumerated and mitigated (spatial-dim via pad-1; optimizer param-filter; post-`apply` load; normalization-space match). The real risk is **scientific, not a bug**: at 183 fully-annealed epochs the marginal gain may be a few tenths and could land near/below the +0.1pp bar (honest estimate +0.1–0.4pp). Clean cheap negative if it fails.

### 2. Increase model capacity (width multiplier)
**Summary**: Widen DavidNet channel counts by a uniform multiplier (primary **1.25×** → 80/160/320/640, ~10.3M params), keeping the EXP-001+002 recipe byte-identical, on the bet that capacity is now the binding limiter on the path from 95% toward 96%. Round widths to multiples of 8 for channels_last efficiency. Gate the choice on a ~30s throughput smoke probe (read live img/s from the first ~2 epochs, then kill) to confirm ≥~100 epochs survive before committing the official run.

**What it targets**: Representational capacity — a richer per-layer feature basis → lower achievable loss floor, *conditional* on enough low-LR tail steps surviving the throughput hit.

**Reasoning**: WideResNet shows width is the compute-efficient CIFAR capacity lever for wide-shallow nets; airbench scales its net up for the 96% target vs 94%; our VRAM/throughput headroom is large (project-insight High); EMA+TTA scaffolding is inherited free. See `proposals/idea-03.md`.

**Sources**: `proposals/idea-03.md`; Zagoruyko & Komodakis 2016; arXiv:2404.00498; project-insights High (EXP-001).

**Estimated Effort**: Low (~6 lines in ResNet9 + a width constant; one throughput smoke + one official run).

**Risk Assessment**: Two-sided. 1.25× costs ~1.56× FLOPs → ~115 epochs (from 183); the dominant risk is **under-annealing washing out the capacity gain** (EXP-002 idea-review item 8). Secondary: PEAK_LR/wd were tuned at 6.57M params — a sub-bar result could be a tuning artifact, not a true capacity ceiling. Honest estimate: central ~95.85% (just at bar), high variance, but the **highest ceiling** of the three (clean path toward 96%).

### 3. Muon optimizer for conv/Linear weights (moonshot)
**Summary**: Replace stock Nesterov-SGD on the 2D conv weights with a pure-torch **Muon** optimizer — orthogonalize each weight's momentum matrix via a 5-step Newton–Schulz quintic iteration (so every singular direction gets a unit-scale update), keeping BN (1D) and the tiny `fc` head (10×512, rank-degenerate) on SGD. Schedule both off the same time-based one-cycle shape with separate peaks (SGD 0.4 for BN+head; Muon ~0.05 for conv). EMA/TTA/wd/LS/architecture unchanged.

**What it targets**: Optimization efficiency per step/epoch — a better-conditioned update spectrum reaches a lower-loss/flatter basin in the same wall-clock, lifting the tail accuracy EMA+TTA then denoises.

**Reasoning**: Muon set airbench's current single-GPU record; the verbatim algorithm (coeffs, bf16, Frobenius pre-scale, transpose-on-tall) is pinned from the repo. ResNet9 is fully bias-free so the Muon/SGD partition is clean (only BN + head off Muon). Orthogonal to EXP-002 eval-side wins. See `proposals/idea-04.md`.

**Sources**: `proposals/idea-04.md`; arXiv:2404.00498 + `airbench94_muon.py`; Muon writeup (Keller Jordan).

**Estimated Effort**: High (custom optimizer; LR/scaling tuning; divergence risk).

**Risk Assessment**: Dominant risk = **Muon peak LR is genuinely unknown for this 183-epoch full-WD recipe** — too high → bf16-NS divergence/NaN; too low → under-steps and NS overhead costs epochs → net-negative. Honest estimate: ~50% fails to clear bar, ~25% upside +0.1–0.4pp, ~25% divergence/under-step. Highest variance + effort, but uniquely answers whether the optimizer is still a lever; clean fallback ladder (halve LR / raise LR / ns_steps=3 / revert).

## Review

Cross-model adversarial review (Codex, idea-critic role) in `01-idea-review.md`. Verdict: **pick Idea-01 (whitening)** — 8/10 evidence, 5.5/10 impact; cleanest, preserves proven 300s dynamics, keeps EMA/TTA, no hard-constraint issue, attacks conditioning without spending the low-LR tail. Width (03): 6/10 ev, 8/10 impact — highest ceiling but knowingly trades away the strongest prior lever (update count) with no observed capacity bottleneck. Muon (04): 5/10 ev, 7/10 impact — too many coupled unknowns (unpinned LR/scaling + a WD-removal confound) for a one-shot EXP-003.

Top concerns + resolutions for the chosen idea (Idea-01):
- **#5 RNG confound (valid, will fix):** the proposed patch-subsample used global `torch.randperm` after `manual_seed(42)`, advancing global RNG before dataloader/augmentation and muddying attribution. **Resolution:** use a local `torch.Generator().manual_seed(...)` for the patch subsample (or a deterministic stride / all patches) so global RNG state is untouched — carry into the plan.
- **#4 upside compressed at 183 epochs (valid, scientific not bug):** whitening's airbench gain is an epochs-to-94% accelerator; we already fully anneal. **Resolution:** keep the first run a clean A/B; **log early-epoch (e.g. ep≤10) and tail eval deltas vs EXP-002's trajectory** so a null is interpretable (did conditioning improve early convergence even if final acc didn't move?).
- **#6 "WD corrupts frozen params" overstated (partly valid):** SGD skips `grad is None` params, so `requires_grad=False` usually already prevents WD updates; the optimizer param-filter is still correct defensive practice (kept), just not the main risk.
- Concerns #1/#2/#3/#7 target the unpicked ideas (Muon WD confound, width spending the tail, width's smoke-gate) — not adopted, so resolved by non-selection.

## Idea Evaluation

Adopt the reviewer's pick: **Idea-01, frozen patch-whitening front-end**. It is the highest expected-value *low-risk* move for EXP-003 — it composes additively on the kept EMA+TTA base, leaves the validated training dynamics (one-cycle, update count, low-LR tail) fully intact, and cleanly answers whether input conditioning is still a lever. Width (highest ceiling) and Muon (optimizer lever) are explicitly deferred to later loops: width once/if capacity is shown binding, Muon with a tighter reference port. No override of the verdict.

## Chosen Idea
**Selected**: Frozen patch-whitening first convolution (Idea-01)

**Why this idea**:
It is the canonical, untried fast-CIFAR front-end (Page/hlb/airbench) and the cleanest fit to our constraints: a frozen 3×3/pad-1 ZCA-whitening conv computed once off-budget, slotted before the existing `prep`, with EMA+TTA and the entire 300s one-cycle recipe untouched for clean attribution. Unlike width it does not spend the low-LR-tail update count that produced prior gains, and unlike Muon it has no unpinned LR/scaling. Refined per review: local-RNG patch sampling (no global-RNG confound) and early-vs-tail delta logging for an interpretable result. Honest expected magnitude +0.1–0.4pp (straddles the +0.1pp bar; a clean sub-bar null is genuinely possible and would itself be informative).

**Hypothesis**:
Replacing the raw-RGB input to the learnable stem with a frozen, data-whitened 54-channel representation will improve `best_test_acc` from 95.72% to **≈95.9–96.1%** (central ~95.95%), clearing the ≥95.82% bar, by better-conditioning the first-layer optimization so the completing one-cycle reaches a marginally lower-loss tail minimum within the same 300s budget — with the gain visible as faster early-epoch convergence and a modestly higher annealed tail vs the EXP-002 trajectory, and EMA+TTA preserved.
