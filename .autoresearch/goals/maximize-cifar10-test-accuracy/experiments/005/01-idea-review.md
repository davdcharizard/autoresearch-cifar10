I have all the context. Here is the adversarial review.

---

# Adversarial Review — EXP-005 Candidate Ideas

## Prioritized Feedback (most important first)

### 1. [Cross-cutting, near-fatal for the timing line] The entire signal is at or below single-run noise, and there is no repeat-seed budget
Every negative result this line rests on is one seed. EXP-004's "50–65% window is valuable" conclusion is a **0.16-point single-run delta** (93.91 vs 94.07), and seed rerolling is a hard-constraint prohibition (`01-definition.md:17`). The acceptance bar is +0.10 (`01-definition.md:25`). So any idea whose *expected* effect is ≤~0.15 points is betting on out-running noise it cannot measure. The 75% idea's own risk note concedes the effect is "modest and near the single-run noise scale" (`01-brainstorm.md:70`) — that is close to self-disqualifying against a +0.10 bar.
- **Fix / path:** Prefer the idea with the largest *expected magnitude*, not the cleanest narrative. If a timing idea is chosen anyway, pre-register that a result inside 94.07±0.15 is "noise, not directional," so a marginal 94.18 is not over-read.

### 2. [Later 75% Cutoff] Mechanism gap — the two effects oppose, and EXP-002 evidence argues *against* it
The idea's logic is "shortening the window hurt (EXP-004), so lengthening it helps." That symmetry is unjustified. Extending mixup to 75% **shrinks the hard-label tail from 105s to 75s** — and EXP-002's own report states accuracy "continued improving throughout the hard-label tail" (`002/04-analysis.md:24`). You are trading more soft-label regularization for *less* of a phase the data shows is still productive. The net can easily be negative, and the diagnosis (`01-brainstorm.md:21`) already flags "75 seconds may be insufficient for hard-label margins to recover."
- **Fix / path:** This is the weakest bet as framed. If run, it should be justified by tail-productivity data (does EXP-002's accuracy curve still climb in the *last* 30s?), not by mirror-symmetry with EXP-004.

### 3. [WRN-16-3] Confounds four variables — a win is un-attributable, a loss is ambiguous
The proposal changes `WIDEN_FACTOR`, `BATCH_SIZE` (256→384), `LR` (0.2→0.3), and `MIN_LR` (0.002→0.003) simultaneously (`idea-03.md:34-38`). The "one utilization/optimization pairing" argument (`idea-03.md:54-59`) is reasonable engineering but means a positive result cannot separate capacity from the batch/LR change, and **larger-batch generalization loss** (a real, known effect at batch 384 on a 50k-image set) is confounded with the capacity test the idea is designed to run.
- **Fix / path:** Hold batch/LR at the validated 256/0.2 for a first width-3 pass (accept lower utilization), *then* tune batch/LR only if width alone is promising. Isolates the variable the hypothesis is actually about.

### 4. [WRN-16-3] Severe throughput risk with a high probability of an *inconclusive* run
Width-3 roughly doubles convolutional work; the proposal's own smoke gate floor is **80 passes vs EXP-002's 141.9** — a ~45% exposure cut — and its decision rules explicitly classify a sub-80-pass regression as "compute-limited... do not infer" (`idea-03.md:101-103`). EXP-001 established exposure is a first-order driver of accuracy here (`001/04-analysis.md:27`). So the modal outcome is a run that either regresses on exposure or lands in the ambiguous band, burning the experiment slot without a clean verdict.
- **Fix / path:** The smoke gate is good and should be kept strictly. But recognize the expected value is dragged down by a real chance of "valid but uninterpretable." Worth it only if the ceiling justifies the ambiguity risk.

### 5. [Alpha-0.4] No local evidence 0.2 is under-strength — direction is a genuine coin-flip
The idea concedes "no local evidence alpha 0.2 is under-strength" (`01-brainstorm.md:44`), and `mixup.md` gives no setting-specific optimal alpha (`mixup.md:13-15`). Alpha could as easily be already-optimal or too high. Additionally, note the implementation uses **batchwise mixup** — one shared `mix` scalar per batch (`train.py:121`), not per-sample — so raising alpha changes the distribution of a single per-batch λ; the effect is real but weaker than per-sample mixup would deliver.
- **Fix / path:** This is inherent exploratory risk, acceptable *because it probes the one untested regularization axis (strength) on the stated limiter*. Strengthen the case by pre-committing the mechanism check: if 0.4 helps, late test-loss should drop (mirroring EXP-004's finding that less regularization *raised* test loss to 0.2708, `004/04-analysis.md:23`).

### 6. [Cross-cutting] "Generalization is the limiter" is asserted, not re-measured
All three ideas inherit the diagnosis that generalization bounds the metric (`01-brainstorm.md:20`), but no EXP-005 artifact re-confirms the train/test gap on the *current* 94.07 model. Ideas 1 and 3 (more/relocated regularization) are only correct if the model is still over-fitting; width-3 (idea 2) is the only one that raises the *capacity* ceiling if the gap is actually small. The brainstorm does not test which regime we are in.
- **Fix / path:** Cheap: report the final train loss / train-acc from the EXP-002 log. Near-zero train loss with 94% test → regularization axis (ideas 1/3) is right; sizable train error → capacity axis (idea 2) is right. This single number should arbitrate the whole choice.

---

## Scored Verdict

| Idea | Evidence & Reasoning | Potential Impact |
|---|---|---|
| **Stronger Alpha-0.4 @ 65%** | **6/10** — clean single-variable design on the correct (generalization) axis, but no evidence 0.2 is under-strength; direction unproven. | **6/10** — moderate ceiling; strength is the one untested regularization lever, plausibly bigger than a 30s tail shift, no exposure cost. |
| **Mixup-Regularized WRN-16-3** | **6/10** — best-developed proposal, answers the prior unregularized-width objection, grounded in EXP-001+002 — but confounds 4 variables and risks an inconclusive run. | **7/10** — highest ceiling (raises capacity, not just re-tunes regularization); proposal's 94.2–94.6 claim is credible *if* exposure survives. |
| **Later 75% Cutoff** | **4/10** — relies on unjustified mirror-symmetry with EXP-004 and contradicts EXP-002's "tail still improving" evidence. | **3/10** — self-described as near single-run noise; low odds of clearing the +0.10 bar. |

### Pick: **Stronger Alpha-0.4 Mixup at 65%**

It wins on expected value, not ceiling. It attacks the diagnosed limiter (generalization) on the single regularization axis never yet probed — strength — while every prior experiment moved timing (EXP-004, EXP-002) or spatial form (EXP-003). It is a clean one-line, single-variable change (`MIXUP_ALPHA` only), so a result of either sign is attributable, and it carries **no exposure penalty**, unlike width-3, whose most likely outcome is a throughput-confounded or sub-80-pass inconclusive run (feedback #3–#4). It has a materially higher expected magnitude than the 75% cutoff, which by its own admission sits in the noise and additionally sacrifices a tail the data shows is still productive (feedback #2).

The honest caveat: alpha direction is unproven (feedback #5), so this is an exploratory bet — but it is the correct-axis exploratory bet with clean attribution and low downside. **WRN-16-3 is the right #2** and the better bet *if* the EXP-002 train loss turns out not to be near-zero (feedback #6) — its ceiling is highest and it is the only idea that lifts capacity rather than re-tuning regularization. Recommend queuing it next with batch/LR held at 256/0.2 to de-confound. Drop the 75% cutoff unless the EXP-002 tail curve is shown to have already flattened.
