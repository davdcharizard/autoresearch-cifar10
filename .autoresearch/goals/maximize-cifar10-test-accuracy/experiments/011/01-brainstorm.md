# Brainstorm EXP-011
**Created**: 2026-06-29

<!-- Goal/metric/constraints: goals/maximize-cifar10-test-accuracy/01-definition.md. Baseline 96.38 (EXP-008), bar ≥96.48, in 04-results.tsv. -->

## Web Search & Literature Review

- **CutMix (Yun et al., ICCV 2019, arXiv:1905.04899)** + short-schedule benchmark evidence (OpenMixup CIFAR benchmarks; TransformMix arXiv:2403.12429; Mixup-Without-Hesitation arXiv:2101.04342): **CutMix helps even at SHORT schedules** — ResNet-18/CIFAR-10 CutMix ≈96.1–96.2% @200ep vs vanilla ~94.9 and mixup ~95.6; CutMix converges fast and gives its best *relative* advantage early, while plain mixup needs 800–2000 epochs to pay off. ⇒ for our ~150-epoch budget, **CutMix (region mixing + soft labels), not mixup**, is the right member of the mixing family. Region-mixing is mechanistically distinct from single-image occlusion (Cutout/RandomErasing), so it can stack on the EXP-008 aug rather than being redundant.
- **Bag of Tricks (He et al., CVPR 2019, arXiv:1812.01187)**: exclude BN γ/β and biases from weight decay ("no bias decay"); label smoothing. BUT the main effect of WD in a BN net is on the EFFECTIVE learning rate (BN scale-invariance makes conv-weight norm not affect block output), and isolated returns are **mixed/diminishing** (He 2019; some ResNet-18 studies found targeted norm/bias decay worth ~0.8pp, others ~0). ⇒ WD-shaping is a modest-confidence tunable, not a guaranteed win.
- **SAM (Foret et al., ICLR 2021, arXiv:2010.01412)**: min-max over an ε-ball seeks FLAT minima → better generalization (+0.3–1.0pp on CIFAR); ρ≈0.05 for CIFAR. Cost: two forward/backward per step (~1.85× wall) → under the fixed 300s budget this roughly halves epochs unless gated.
- Standing refs: `knowledge/references/fast-cifar10-recipes.md` (DavidNet/airbench lineage), `muon-optimizer.md`, `rezero-identity-init.md`.

## Experimental History Review

Current best **96.38% (EXP-008)**: DavidNet/ResNet-9 + frozen ZCA whitening conv + ReZero GatedResidual(256)@layer2 + EMA(0.998) + flip-TTA, SGD-Nesterov lr 0.4 / wd 5e-4 / LS 0.2, time-based triangular one-cycle (~150 epochs/300s), aug = RandomCrop+Flip + Cutout(12) + light RandomErasing.

- **What worked**: architecture install (EXP-001, +3.65pp), EMA+flip-TTA (EXP-002, +0.50), whitening (EXP-003, +0.15), ReZero capacity@layer2 (EXP-004, +0.13), **stronger augmentation (EXP-008, +0.38 — largest lever since EXP-001)**.
- **What failed (approach-specific)**: capacity adds under-anneal — deepen 4×4 (EXP-005, −0.10), widen 256→384 (EXP-007, −0.15, cut epochs 150→94); TTA variants swamped by noise (EXP-006, −0.07); Muon optimizer diverged at lr 0.24 (EXP-009) and, once LR-tuned, only TIED SGD (EXP-010, 96.33 vs 96.38).
- **Converged diagnosis**: the net is **regularization-bound with a ~4× epoch surplus** (fits ~150 ep for 96.38 vs airbench96's ~37). Optimizer axis exhausted (EXP-009/010); capacity adds under-anneal; eval-side TTA exhausted. The productive lever class is **throughput-free regularization** (EXP-008 existence proof). ~0.1pp run-to-run noise floor (fixed seed; epoch-count jitter) ⇒ require ≥0.1pp clearly above noise.
- **Untried gaps**: (a) data-mixing regularization (CutMix) — never tried; (b) the core SGD scalars (LS 0.2, wd 5e-4) are STALE — set in EXP-001/002 and never retuned after the big EXP-008 aug change; ReZero α is decayed toward 0 by uniform wd (fights the gate); (c) optimization-geometry / flat-minima (SAM) — never tried; (d) one-cycle SHAPE (linear decay, peak 0.4, pct_start 0.15) set in EXP-001, never revisited.

## Diagnosis — what limits the metric now

Limiter = **generalization on a regularization-bound net with a large epoch surplus**, NOT optimizer convergence (Muon ties SGD, EXP-010) and NOT capacity (capacity adds under-anneal, EXP-005/007). The fully-annealed minimum's generalization is set by the *total regularization dose and its allocation* across input-space aug, weight-space decay, target-space smoothing, and loss-geometry. The only lever class with a >noise win here is throughput-free regularization (EXP-008). So the high-value moves either (i) add a *complementary* regularizer that converts more of the epoch surplus into a higher annealed ceiling without cutting epochs, or (ii) refresh the now-stale regularization scalars to a better allocation, or (iii) directly target flat minima — accepting an epoch cost only if kept above the ~110-epoch under-anneal threshold.

## Collected Ideas

- **CutMix region-mixing augmentation** (literature / orthogonal regularizer) — paste a real patch from another image, area-split the label; throughput-free, complementary to occlusion aug.
- **Mixup** (literature) — input-space convex mix; REJECTED as standalone lead: needs long schedules, weaker than CutMix at ~150 ep.
- **Weight-decay shaping** (history/bag-of-tricks) — no-WD on BN γ/β + ReZero α; optionally retune conv WD.
- **Label-smoothing retune** 0.2→0.1 (stale-scalar refresh) — rebalance target-side reg now that input aug is stronger.
- **One-cycle reshape** (orthogonal) — cosine decay / shorter ramp / hotter peak to exploit the low-LR tail; throughput-free.
- **SAM — sharpness-aware minimization** (moonshot, optimization geometry) — flat-minima seeking; tail-gate to bound the 2× cost.
- **GELU activation swap** (representation) — smoother activation à la hlb/airbench; marginal, slight throughput cost — deprioritized.
- **Stronger combined aug stack** (push cutout/RE harder) — risk over-augmentation/under-fit; subsumed by CutMix test.

## Combinations

- **CutMix + LS retune**: CutMix already provides soft (area-mixed) labels, so LS 0.2 on top may over-soften; pairing CutMix with reduced LS (0.1/0) is plausibly stronger than either alone. (Folded into idea-01 as the pre-registered first follow-up to keep single-variable attribution this loop.)
- **CutMix off-in-tail + EMA**: disabling CutMix in the low-LR tail lets EMA average clean-image iterates (the tail is where accuracy is set) — a curriculum that beats constant-p. (Folded into idea-01.)
- **SAM-in-tail + EMA + flip-TTA**: SAM concentrated on the same tail phase where EMA/TTA already act — flat-basin iterates that EMA then centers. (Folded into idea-04.)
- **WD-shaping + LS retune**: the two stale scalars refreshed together as one "recipe-scalar refresh" (idea-02), read as a small mini-sweep.

## Candidate Ideas

### 1. CutMix data-mixing regularization
**Summary**: Add CutMix to the training step (proposal `proposals/idea-01.md`): with per-batch prob `p=0.5`, sample λ~Beta(1,1), paste a random area-`(1-λ)` box from a batch permutation into each image, and train on the area-corrected mixed loss `λ·CE(out,y)+(1-λ)·CE(out,y_perm)`. Disable CutMix in the final 15% (clean low-LR tail for EMA). One forward/one backward per step (the criterion is just called twice on the same logits) ⇒ throughput-free. Hold LS=0.2 this loop for clean attribution; pre-register LS→0.1 as the first follow-up if the run reads under-fit.

**What it targets**: the regularization-bound limiter — converts more of the ~4× epoch surplus into a higher annealed ceiling via a regularizer mechanistically *complementary* to the EXP-008 single-image occlusion aug (real-content region mixing + soft labels vs zeroing a box). Cites the inline diagnosis + `03-experiment-learnings.md` Patterns (EXP-008).

**Reasoning**: Throughput-free regularization is the only proven >noise lever here (EXP-008 +0.38). CutMix is the canonical strong CIFAR mixing aug, untried, and — unlike mixup — pays off at short schedules (ResNet-18 CutMix ≈96.2 @200ep). Cannot under-anneal (GPU cost ~0; pre-registered `num_epochs`∈[142,155] guard). Mechanistically distinct from occlusion ⇒ should stack rather than be redundant.

**Sources**: `proposals/idea-01.md`; arXiv:1905.04899; OpenMixup/TransformMix short-schedule benchmarks; EXP-008 (`experiments/008/04-analysis.md`) Next Steps explicitly name a second complementary augmentation.

**Estimated Effort**: low–medium (one ~12-line helper + ~10-line loss-branch edit; one 300s run + smoke).

**Risk Assessment**: redundant with existing occlusion aug → marginal gain sub-noise on the saturated base; OR Cutout+RE+CutMix+LS-0.2 over-regularizes → under-fit. Worst realistic case is "flat at normal epochs" (cannot under-anneal). ep25<~91.5 + flat final ⇒ over-softening → reduce p/LS (pre-registered).

### 2. Recipe-scalar refresh — weight-decay shaping + label-smoothing retune
**Summary**: Refresh the stale SGD scalars (proposal `proposals/idea-02.md`): (a) split optimizer param groups so weight decay applies only to conv/fc weight matrices and is removed (wd=0) for BN γ/β and the ReZero α scalar (clean `p.ndim<=1` split — verified there are no bias params; whitening conv stays excluded via `requires_grad`); (b) drop LS 0.2→0.1. Best executed as a small 3-cell read (A: WD-shaping only; B: shaping+LS0.1 — the headline; C: shaping+LS0.1+conv-wd 8e-4). Hold PEAK_LR fixed; zero throughput cost.

**What it targets**: the regularization *allocation* on the regularization-bound net — the scalars were set for the EXP-001/002 recipe (95.2–95.7%) and never retuned after EXP-008 changed the total reg budget; uniform wd also actively fights the ReZero α capacity gate (EXP-004 measured α.grad≈0.0179, decay term non-negligible).

**Reasoning**: Throughput-free regularization-shaping = the proven lever class; explicitly listed in EXP-008's Next Steps. Net-specific α-decoupling is not priced into generic bag-of-tricks accounting. Cannot under-anneal (zero compute change).

**Sources**: `proposals/idea-02.md`; arXiv:1812.01187; EXP-008 Next Steps + `experiments/008/proposals/idea-01.md` (decoupled-WD, never executed); EXP-004 (α.grad).

**Estimated Effort**: low (one ~10-line optimizer partition + one constant; 1–3 full runs).

**Risk Assessment**: each knob individually sub-noise on this saturated base → unprovable single-run; bundling muddies attribution. WD-shaping is per literature a mixed/diminishing effective-LR knob. Honest: modest-confidence, low-downside, not a likely decisive win.

### 3. Tail-gated Sharpness-Aware Minimization (SAM) — moonshot
**Summary**: Hand-roll SAM (proposal `proposals/idea-04.md`) and apply it ONLY in the final ~25% of the budget (`SAM_START_FRAC=0.75`, ρ=0.05, BN running-stats frozen on the ascent pass): plain SGD for the first 75% (full epoch rate), then the 2× ascend/descend step in the low-LR tail where accuracy and basin-shape are set. Estimated realized epochs ~133 (≥110 threshold), schedule stays time-keyed so it fully anneals.

**What it targets**: the generalization limiter directly — flat minima. SAM changes the loss *geometry sought* (not the update direction/scale), a *different class* than the optimizer-direction Muon that tied; flat-minima-seeking is in the same class as augmentation (the lever that won).

**Reasoning**: Net is generalization-bound; SAM is famous for +0.3–1.0pp on CIFAR. Tail-gating is engineered to dodge the under-anneal trap that sank EXP-005/007 by paying the 2× cost only where it buys flatness, and synergizes with the EMA-tail/TTA-tail.

**Sources**: `proposals/idea-04.md`; arXiv:2010.01412; EXP-001 (tail Pattern), EXP-010 (optimizer-tie caveat), EXP-007 (under-anneal threshold).

**Estimated Effort**: medium (~40 lines: SAM helpers + BN-freeze + step rewrite; smoke + one full run).

**Risk Assessment**: highest-risk. Likeliest failure = optimizer-axis null (SGD's one-cycle tail minimum already fairly flat → gain sub-bar), per EXP-010 prior. Also under-anneal if GPU 1 contended (read `num_epochs` first). Author's honest estimate: ~25–30% to clear the bar; central outcome a tie within noise; kill-criterion pre-registered.

## Review

Cross-model adversarial review (Codex) in `01-idea-review.md`. **Pick: CutMix** (evidence/reasoning 8/10, impact 7/10) over Recipe-scalar refresh (6.5/5) and tail-gated SAM (5.5/6.5): "attacks the diagnosed limiter most directly, untried, strongest external + local support, avoids the capacity/epoch trap." No hard-constraint violations in any candidate. Two concrete CutMix refinements raised, both adopted:

1. **Throughput gotcha (must-fix)**: `torch.randint(..., device=inputs.device).item()` for the box center forces a CUDA scalar sync inside the timed step → could quietly cut epochs and break the "throughput-free" claim. **Resolution**: draw the box center and λ on **CPU** with the already-seeded torch CPU RNG (or Python ints) — only `torch.randperm` for the batch permutation needs to be on-device (it's used purely for indexing, no sync). Pre-registered guard `num_epochs∈[142,155]` confirms it stayed free.
2. **LS=0.2 is the central design risk, not an afterthought**: CutMix + LS-0.2 both soften targets → possible over-softening/under-fit; idea-02 independently argues LS-0.2 is stale post-EXP-008. **Resolution**: treat this as a planned **2-cell decision**, not a single held-LS run — cell-1 CutMix@LS0.2 (clean single-variable delta vs the 96.38 baseline), cell-2 CutMix@LS0.1 (the reviewer's companion). Run cell-1 first; if it's flat/under-fit (ep25 depressed) or below the bar, cell-2 directly tests the over-softening hypothesis. This keeps attribution clean while not shipping a deliberately-weakened CutMix. (LS-0.1 also partially captures idea-02's highest-prior stale-scalar fix as a free rider.)

Other candidates' concerns (recorded, not blocking this loop): idea-02's cell C (8e-4 conv-WD) is a confound — drop/replace with LS-only if idea-02 is run later; idea-04 SAM perturbs BN/α (should perturb conv/fc only) and its ~133-epoch estimate is optimistic given EXP-005 lost at 131 — keep the kill-criterion.

## Idea Evaluation

Adopt the reviewer's pick (CutMix). It dominates on both axes: it is the only finalist on the *proven* lever class (throughput-free regularization, EXP-008 +0.38) via a genuinely *new, complementary* mechanism (region mixing + soft labels vs occlusion), it cannot under-anneal, and its central estimate (+0.10–0.30pp) overlaps the bar — whereas idea-02 is honestly sub-noise-per-knob and idea-04 is a ~25–30% moonshot fighting the EXP-010 optimizer-tie prior. idea-02 and idea-04 are retained as future loops (idea-02 as the next throughput-free refresh; idea-04 only if the regularization axis stalls). Full scored critique in `01-idea-review.md`.

## Chosen Idea
**Selected**: CutMix data-mixing regularization (`proposals/idea-01.md`), with the two review refinements above.

**Why this idea**:
The net is regularization-bound with a ~4× epoch surplus, and the only >noise lever in this project's history is throughput-free regularization (EXP-008, +0.38pp). CutMix is the canonical strong CIFAR mixing augmentation, untried here, mechanistically *complementary* to the existing single-image occlusion aug (it pastes real cross-class content and area-splits the label rather than zeroing a box), and — unlike mixup — it pays off at short schedules (ResNet-18 CutMix ≈96.2 @200ep vs mixup ~95.6, vanilla ~94.9). It adds essentially zero GPU cost (one forward/backward; the criterion is just evaluated twice on the same logits), so it cannot under-anneal — the failure mode that sank every capacity experiment. The reviewer ranked it first on both evidence and impact.

**Hypothesis**:
Adding CutMix (α=1, p=0.5, disabled in the final 15% tail), with the box/λ drawn on CPU to stay throughput-free, will raise `best_test_acc` above the 96.48 bar (≥+0.10pp clearly above the ~0.1pp noise floor) while keeping `num_epochs` in the ~142–155 band — because region-mixing supplies a complementary regularization signal that converts surplus epochs into a flatter, better-generalizing annealed minimum. Falsifiable predictions: (a) if `num_epochs` drops below ~135, the box-scalar draw is still syncing and the result is confounded (fix before judging); (b) if cell-1 (LS 0.2) lands flat/under-fit with a depressed ep25 (<~91.5 vs EXP-008's 92.31), the combined Cutout+RE+CutMix+LS-0.2 over-regularized → cell-2 (LS 0.1) should recover and is the path to the bar; (c) if both cells land in [96.30, 96.45] at normal epochs, CutMix is redundant with the existing occlusion aug on this saturated base (no-improvement).
