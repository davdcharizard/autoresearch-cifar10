I have everything I need. Emitting the adversarial review.

---

# EXP-004 Adversarial Idea Review

## Prioritized Feedback (most important first)

### 1. Batch-256 (idea-02): its own cited evidence argues *against* it — likely fatal
- **The diagnosis contradicts the intervention.** The proposal invokes EXP-003 as its "cautionary result": a **6.7%** step reduction produced a flat/failed top-1. Yet its own throughput table (`idea-02.md:78-84`) projects the *expected* regime at ~25,000 steps vs EXP-002's 38,629 — a **35% reduction in optimizer updates** (65–86% retained even optimistically). That is **2–5× larger** than the step loss that already sank EXP-003, in the exact direction this benchmark has demonstrated it is sensitive to. The benchmark is update-limited, not example-limited: at ~99 epochs of the same 50k images under crop/flip only, ResNet-20 is not starved for fresh examples, so "29–73% more exposure" is unlikely to compensate for 14–35% fewer updates.
- **No measured throughput floor.** The entire case rests on "the H20 has headroom," but for a 269k-param model the batch-128 step (7.77 ms) is dominated by kernel-launch + per-step `torch.cuda.synchronize()` (`train.py:196`) + Python/H2D overhead, **not** compute. Whether batch 256 actually raises examples/s is unverified. The proposal *admits* the pessimistic regime "performs no more example work... and only half as many optimizer updates" (`idea-02.md:84`) — i.e. a coin-flip on a strictly-worse outcome. **Concrete fix:** run a batch-256 step-time microbenchmark *first* (mirror idea-01's preflight gate) and only proceed if examples/s materially rises; even then, the update-count math makes the accuracy case weak. This is the one idea that reintroduces a known-failed mechanism at larger magnitude.

### 2. RandAugment (idea-01): the elaborate preflight guards the *wrong* failure
- **The dominant risk is accuracy underfit, which the preflight cannot catch.** Single-op magnitude-7 PIL transforms on 32×32 images across 8 workers are cheap — the throughput gate (`R_candidate ≥ 80 batches/s`, `idea-01.md:146-153`) will almost certainly pass and yield false confidence. But RandAugment's CIFAR gains in Cubuk et al. come from **long** schedules on wide nets; here the budget is ~100 epochs / 300 s. Stronger augmentation raises effective task difficulty, and in a short fixed horizon the model can underfit → **flat or negative** best_test_acc. The proposal names this (risk #4, `idea-01.md:196-197`) but does not mitigate it. **Concrete fix:** apply RandAugment during the high-LR plateau and revert to crop/flip for the 20% refinement tail (parallels EXP-003's suggested smoothing-off-tail), or ramp magnitude — either converts "harder task, same budget" into "regularize then fit." At minimum, state that the preflight validates *feasibility only*, not the mechanism.
- **BN train/eval mismatch** (risk #5): BN stats learned on augmented images, evaluated on clean — bounded by the conservative single-op choice, acceptable.
- **Strength to keep:** this is the *only* idea whose mechanism is immune to the update-count penalty (host-side, outside the timer at `train.py:172-173`), directly attacking the diagnosed "weak image-level invariance" limiter without paying EXP-003's cost.

### 3. Preactivation (idea-03): the mechanism is discounted by its own source at this depth
- **Shallow-depth non-transfer is the crux.** He et al.'s gain is ResNet-**110** (6.61%→6.37%, Δ0.24), and the paper explicitly states post-addition truncation is *less severe* at low depth (`idea-03.md:18,41,160`). ResNet-20 has 9 blocks — the mechanism is weakest exactly where it's being applied, and published ResNet-20 preactivation-vs-post deltas are often within ±0.1–0.2 (noise relative to the 0.10 threshold), with **uncertain sign**. Expected effect may sit below the required margin. There is no mitigation because none is possible without changing depth. **This caps potential impact, not correctness.**
- **lr mis-tuning confound:** preactivation shifts activation scales into the convs; the fixed lr=0.1 (`idea-03.md:163`) was tuned for post-activation, so a null result could reflect mis-tuning rather than the architecture. Fixing lr for clean attribution is defensible, but flag that a null is not conclusive.
- **Strengths to keep:** exact param parity (269,722, statically asserted `idea-03.md:184-188`), same conv shapes → throughput within ~3% (no update penalty), cleanest isolation of the three, and it is the *only* untested architecture lever. Implementation (shortcut from raw `x`, final BN-ReLU, Option-A retained) is correct as written.

### Cross-cutting
- No hard-constraint violations in any candidate: all touch only `train.py`, no new deps (torchvision RandAugment is in installed 0.24.1), VRAM trivial, no seed hacking, single fixed-seed runs. RNG-draw reordering in idea-01 is a property of the method, not seed hacking (`idea-01.md:202-203`) — correctly framed.
- **Asymmetry worth noting:** the idea most dependent on an unverified throughput assumption (idea-02) is the *only* one without a preflight or static check to catch its central failure mode.

---

## Scored Verdict

| Idea | Evidence & Reasoning | Potential Impact |
|---|---|---|
| **RandAugment (idea-01)** | **7/10** — direct CIFAR literature + mechanism aligned with the diagnosed limiter (input invariance) and, uniquely, immune to the update-count penalty that bounded EXP-001/003; weakened by short-horizon underfit evidence pointing the other way. | **7/10** — credible +0.1–0.4 ceiling in the tail; downside bounded by conservative single-op mag-7, but real chance of flat/negative in ~100 epochs. |
| **Preactivation (idea-03)** | **6/10** — cleanest isolation, correct implementation, param parity, no throughput risk; but the primary source explicitly discounts the mechanism at ResNet-20 depth, so evidence for a *top-1* move here is thin. | **5/10** — modest and uncertain-sign; expected effect plausibly below the 0.10 threshold. |
| **Batch-256 (idea-02)** | **3/10** — coherent noise-scale coupling argument, but self-undermining: its own cited EXP-003 evidence and the update-limited nature of this benchmark argue against it, and the throughput premise is unmeasured. | **3/10** — asymmetric downward; pessimistic/expected regimes give strictly-or-likely-worse outcomes (14–35% fewer updates). |

## Winner: **One-Operation Magnitude-7 RandAugment (idea-01)**

It wins on merit, not caution. It attacks the diagnosis's own top remaining limiter — weak image-level invariance — with a mechanism that (a) has direct CIFAR-10 literature support and (b) is the *only* candidate that adds regularization **without** sacrificing optimizer updates, the recurring failure mode that killed EXP-001 (lost high-LR regularization) and EXP-003 (lost 6.7% of steps). Preactivation (idea-03) is a close second and the most rigorous experiment, but the source itself predicts its effect fades at ResNet-20 depth, capping expected impact near the noise floor. Batch-256 (idea-02) should be rejected: it reintroduces — at 2–5× magnitude — the exact update-count reduction that already failed, on an unverified throughput assumption, and its cited evidence points against it.

**Required refinement before adopting idea-01:** address the underfit risk head-on (plateau-only augmentation or a magnitude ramp, reverting to crop/flip for the refinement tail), and reframe the preflight explicitly as a feasibility gate — it does not de-risk the actual accuracy question.
