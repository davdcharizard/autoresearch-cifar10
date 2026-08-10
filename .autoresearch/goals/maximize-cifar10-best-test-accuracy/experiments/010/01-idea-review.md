I've read all the required materials. Here is my adversarial review.

---

# EXP-010 Adversarial Review

## Prioritized Critique (most important first)

### 1. [Averaging] FATAL-ADJACENT: The averaging window forfeits EXP-007's online peak and evaluates only correlated near-zero-LR iterates
This is the single most damaging flaw and it undercuts the whole idea. From 90% progress onward, idea-01 evaluates **only** the averaged model. EXP-007's peak 93.55% occurred at **epoch 67 of 71** — inside the ≥90% window (the proposal itself estimates 90–100% ≈ epochs 64–71). So the run *gives up* the exact online observation that produced the 93.55% baseline and replaces it with averaged evaluations. To reach the 93.65% bar, averaging must first *recover* the 93.55% it forfeited and then exceed it — from checkpoints the proposal's own failure-mode list calls "too close in weight space for averaging to change generalization." Worse, LR over the window runs 0.005→1e-4, so the iterates barely move: this is precisely the regime where weight averaging yields the *least* benefit (SWA/EMA gains come from averaging a spread-out trajectory under higher/constant LR, not the tail of a cosine already collapsed to ~0). The cited paper (`weight-averaging.md`) says "combining averaging with annealing works best" — but that means averaging *across* the annealing span, not the final 10% where LR is effectively dead.
**Fix:** either (a) evaluate BOTH online and averaged and take the max — but this doubles evaluations and the proposal correctly rules it out as a protocol/selection violation; or (b) widen the window to start at the *beginning* of the cosine tail (~80–85%) so it captures trajectory diversity while LR is still meaningful, accepting BN-mismatch risk on higher-LR points. Neither cleanly escapes the forfeit-vs-diversity dilemma, which is why this idea's ceiling is structurally capped.

### 2. [BF16] Mechanism attacks a bottleneck the diagnosis explicitly rules out
The brainstorm and EXP-007 analysis both state the remaining error "is **not** simply an unfinished terminal ascent" and that width-2 won "**despite 29.2% fewer updates**." The limiter is representation quality/capacity, not optimizer exposure. BF16 buys ~15% more steps — more of a thing the history says is not the constraint. The proposal concedes this directly: "the ceiling is modest because EXP-007's final trajectory was already flat; extra steps... do not introduce a new representation or regularizer." So even in the *success* case, expected Δ is ~0 by the proposal's own admission.
**Fix:** reframe and gate on the *strong-phase* exposure story — 15% more N1/M7 plateau steps is more strong-view capacity-building, which EXP-007 *did* identify as the lever. But even this is weak: EXP-004 had 38k steps at width-1 and lost; marginal steps at fixed capacity are low-value. Honestly, this idea's best outcome is a clean preflight no-go.

### 3. [BF16] High probability the speedup gate fails on this launch-bound model — TF32 is already active
PyTorch 2.9 defaults `torch.backends.cudnn.allow_tf32=True`, so FP32 convolutions already run on Hopper tensor cores via TF32. The proposal flags this ("cuDNN FP32 may already use TF32") but doesn't weight it enough: BF16-over-TF32 on a 1M-param, 32×32, launch/BN/sync-bound ResNet-20 is unlikely to clear ≥15% synchronized speedup. Combined with #2, the joint probability of "gate passes AND accuracy improves" is low.
**Fix:** none that rescues the mechanism — the rigorous gate is the idea's saving grace (no wasted run), but it also means it likely never runs.

### 4. [CutMix] Directly contradicts EXP-007's explicit next-step caution — must be confronted head-on
EXP-007 `04-analysis.md` line 44: *"do not stack CutMix on the already hard strong phase."* The accepted strong phase is already the hard part (N1/M7 + width-2 reached only 90.08% strong checkpoint; EXP-008/009 show the plateau underfits easily). Adding CutMix soft targets to ~40% of plateau steps in a ~240s high-LR window risks the same underfit signature EXP-008 produced. This is the idea's most serious risk and the proposal addresses it only partially (capacity argument + 50% + hard tail).
**Fix:** the p=0.5 + hard-tail + compose-not-replace design *is* a materially different attempt than the EXP-006 replacement failure and the advisory caution, so it is not a disqualifying retry. Strengthen it by pre-registering the strong-checkpoint accuracy as an underfit tripwire (e.g., if the 80% switch checkpoint falls > ~3 points below EXP-007's 90.08%, attribute to compounded over-regularization) — which the proposal already partially does via trajectory diagnostics.

### 5. [CutMix] Soft-target convergence cost is a documented failure pattern in this exact setup
EXP-003 (label smoothing) is the closest prior: soft targets "lowered loss but not top-1 and reduced steps 6.7%." CutMix targets are soft over ~40% of plateau steps, and mixup/CutMix generally need long horizons (100s of epochs) to pay off; this run has ~70 epochs. The ≥97% step-retention gate handles the throughput half, but not the convergence-slowdown-within-fixed-time half.
**Fix:** the hard-label tail is the right mitigation (restores clean confidence/BN), and the paper evidence for CutMix's *top-1* gains (not just NLL) is stronger than smoothing's — so this is a real risk, not a fatal one. Keep alpha=1.0/p=0.5 fixed as declared; do not tune post-hoc.

### 6. [CutMix] Area-as-target-mass is semantically noisy on 32×32 CIFAR (minor)
CIFAR objects aren't spatially uniform, so pasted-area target mass can misstate class evidence, and independently-RandAugmented donor patches can create artificial edge shortcuts. This is inherent to CutMix and bounded by p=0.5; acceptable. No change needed beyond the declared area-correctness functional gate.

### 7. [Averaging] BN-buffer policy is a genuine but second-order confound
Pairing averaged weights with the online model's most recent running_mean/var is an approximation (`weight-averaging.md` warns BN state must be handled explicitly). Because both weights and buffers come from nearby late weak-tail points, mismatch is small — but it can erase the already-tiny averaging gain. The proposal correctly rejects a BN-recompute pass (costs time/RNG). This is honest and about as good as possible given constraints; it just further caps an already low ceiling.

---

## Scored Verdict

**Late Weak-Tail Checkpoint Averaging (idea-01)**
- Evidence & reasoning: **5/10** — meticulous, honest engineering, but the cited averaging literature's regime (spread-out, higher-LR trajectory) doesn't match averaging only the last 10% at near-dead LR; the proposal's own analysis argues against a gain.
- Potential impact: **3/10** — structurally capped: forfeits the online 93.55% peak and averages highly correlated near-converged points; best plausible case is a wash.

**CUDA BF16 Autocast Throughput (idea-02)**
- Evidence & reasoning: **4/10** — rigorous preflight and clean scope, but the mechanism (more steps) contradicts the diagnosis (error is not an unfinished ascent; width-2 won with *fewer* steps), and TF32 likely already captures the tensor-core speedup.
- Potential impact: **3/10** — near-zero expected Δ even on gate-pass by the proposal's own admission; likely resolves to a preflight no-go.

**Conservative Plateau CutMix (idea-03)**
- Evidence & reasoning: **6.5/10** — strongest literature grounding (CutMix CIFAR top-1 gains), a mechanism-specific fix to EXP-006's information-deletion failure, careful RNG/throughput/target-correctness gating; docked for the direct EXP-007 caution and the EXP-003 soft-target precedent.
- Potential impact: **7/10** — the only candidate introducing a genuinely new regularizer/target geometry rather than refining an already-flat solution; plausible +0.10 to +0.35 with a real, testable upside.

### Pick: **Conservative Plateau CutMix (idea-03)**

It wins on the two things that matter. Both alternatives concede — in their own text and against the goal's own diagnosis — that their expected impact is ~0: averaging summarizes correlated near-zero-LR iterates (and structurally forfeits the online peak it must first recover), and BF16 adds steps the history says are not the limiter. CutMix is the only idea that attacks the *diagnosed* remaining error — generalization/representation geometry — with a mechanism backed by direct CIFAR literature and specifically engineered to avoid the prior Cutout failure. Its risks (the EXP-007 "don't stack" caution, soft-target slowdown) are real and must be watched via the strong-checkpoint underfit tripwire, but they are managed and pre-registered, not fatal, and they do not cap the ceiling the way the other two ideas are capped. On merit — highest defensible upside plus strongest evidence — it is the clear choice.
