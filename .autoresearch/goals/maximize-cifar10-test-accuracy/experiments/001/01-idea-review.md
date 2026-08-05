**Prioritized Constructive Feedback**

1. **Runtime calibration must stay inside the training budget.** The goal fixes 300s of training time, and `train.py` starts accounting only inside the step loop ([train.py:155](</SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/train.py:155>), [train.py:178](</SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/train.py:178>)). If ideas 01/02/03 add “real” optimizer-update calibration before that timer, it becomes off-budget training and is a hard-constraint violation. Fix by using no-update warmup/profiling before the timer, or include calibration updates in `total_training_time` and scheduler step accounting.

2. **Normalization consistency is non-negotiable.** Frozen eval uses `mean=(0.4914,0.4822,0.4465), std=(1,1,1)` ([prepare.py:13](</SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/prepare.py:13>)). Idea 02 and 03 correctly call this out, but implementation should centralize/assert it. Idea 03 especially must compute whitening patches in this exact space; importing reference CIFAR std normalization would silently poison eval.

3. **Idea 03 has the highest ceiling but an unresolved correctness blocker.** Its logit scaling is ambiguous: the proposal sketches dividing after the head, then warns to divide by pre-head/fan-in instead. That scale interacts with label smoothing and high LR, so this can cause divergence or undertraining. Fix before pursuing: use an explicit constant matching the source recipe, e.g. `LOGIT_DIVISOR = 256` if that is the flattened feature dimension.

4. **Idea 03’s 95-96% evidence is less transferable than stated.** The airbench numbers rely on a very specific stack: whitening details, BN behavior, bias handling, LR groups, optimized data path/precision, and sometimes optimizer/kernel choices. The proposal’s “safe v1” drops or changes several of these. Treat the 94-net as the real first target; only claim 96% after porting the exact high-accuracy variant.

5. **Idea 02 should not blindly stretch one-cycle to fill all 300s.** The DavidNet evidence is strongest for ~24-35 epoch completed cycles. A much longer single cycle may keep LR high too long and delay the annealed model. Recommended refinement: target a completed 24-40 epoch cycle first, guard `scheduler.step()` against overrun, then extend only if measured throughput and validation curve justify it.

6. **Idea 02’s LR/loss convention is the main implementation hazard.** The proposal correctly identifies mean-loss `PEAK_LR=0.4`, `weight_decay=5e-4`, and `scale_out=0.125`. Preserve those together. Accidentally mixing summed loss with mean-loss LR would be catastrophic.

7. **Idea 01 is sound but ceiling-limited.** It directly fixes the current broken schedule: baseline has `MAX_STEPS=64000` and milestones at 32k/48k ([train.py:24](</SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/train.py:24>), [train.py:145](</SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/train.py:145>)), while the diagnosis says only ~37k steps fit. But RandomErasing + label smoothing + higher weight decay on small ResNet-20 could over-regularize. If used, stage it: schedule fix first, then add regularizers.

8. **Eval-mode flip TTA is legitimate but should be budget-aware.** `prepare.py` calls `model.eval()` and then `model(inputs)` ([prepare.py:32](</SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/prepare.py:32>)), so gated TTA inside `forward` is within `train.py` scope. But it doubles eval work every epoch; keep an eye on the 10-minute wall-clock kill condition.

**Scored Verdict**

- **Idea 01: ResNet-20 bag of tricks**
  - Evidence/reasoning: **7/10** — the schedule diagnosis is directly supported by current code, but the additive bag-of-tricks gains are mostly indirect for this exact harness.
  - Potential impact: **5.5/10** — likely clears 91.57% by a real margin, but staying on 270k-param ResNet-20 caps the upside.

- **Idea 02: ResNet-9/DavidNet + one-cycle**
  - Evidence/reasoning: **9/10** — directly attacks the schedule and capacity bottlenecks with a widely replicated CIFAR-10 fast recipe, and the proposal handles the subtle LR/loss/output-scale conventions.
  - Potential impact: **8/10** — credible path to ~93.5-94.3%, roughly +2pp or more, with moderate implementation risk inside 300s.

- **Idea 03: Airbench whitening net**
  - Evidence/reasoning: **7.5/10** — source evidence is strong, but the proposal’s simplified port and ambiguous logit scaling weaken the direct claim.
  - Potential impact: **9.5/10** — highest ceiling if ported faithfully, plausibly 95%+, but the first clean implementation is more likely a 93-94% result.

**Pick: Idea 02, ResNet-9/DavidNet + one-cycle.**

It is the strongest experiment 001 choice because it combines high upside with the cleanest path to a correct implementation under the constraints. Idea 03 is the best follow-up once the pipeline has a stable fast-CIFAR rewrite, but for the first experiment it has too many recipe-fidelity traps. Idea 01 is a good fallback, not the best maximization play.
