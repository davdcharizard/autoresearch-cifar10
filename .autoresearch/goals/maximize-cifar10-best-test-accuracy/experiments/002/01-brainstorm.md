# Brainstorm EXP-002
**Created**: 2026-08-05

## Web Search & Literature Review

- **SGDR: Stochastic Gradient Descent with Warm Restarts** (`goals/maximize-cifar10-best-test-accuracy/knowledge/papers/sgdr.md`, https://arxiv.org/abs/1608.03983)
  Smooth schedules should follow the actual training horizon, but EXP-001 shows that CIFAR-10 ResNet20 also needs substantial high-LR exploration before refinement.
- **EXP-001 external idea review** (`experiments/001/01-idea-review.md`)
  The reviewer identified early annealing as the main schedule risk and recommended a de-bundled wider preactivation model as the strongest later capacity bet.

## Experimental History Review

- The moving baseline remains `91.67%`. EXP-001 reached `91.57%` with a 15% hold, time-based cosine, and Nesterov (`04-results.tsv`).
- EXP-001 executed the intended local schedule: LR reached `0.0001`, train loss fell to `0.0215`, and test accuracy increased monotonically to the final checkpoint. The failure is therefore generalization, not missing refinement or measurement sparsity (`experiments/001/04-analysis.md`).
- The most plausible failure mechanism is loss of high-LR implicit regularization after shrinking the plateau from about 83% to 15%. Nesterov remains a causal confound.
- Persistent training workers and seven budget-positioned evaluations reduced total runtime to `321.7s` while preserving 300 seconds of training, 99 epochs, and approximately baseline step count. These are validated protocol controls and should be retained.
- Untried gaps with direct evidence are a baseline-like time-fraction step schedule, a long-hold cosine under standard momentum, and the de-bundled wider preactivation architecture.

## Candidate Ideas

### Baseline-Like Time-Fraction Milestones
**Summary**: Keep standard SGD momentum and replace absolute milestones with elapsed-time phases: `lr=0.1` through 70% of counted training, `0.01` from 70%-90%, and `0.001` for the final 10%. Retain persistent workers and the seven checkpoint evaluations from EXP-001, with every other model/data/loss parameter unchanged.

**What it targets**: EXP-001 showed terminal refinement alone is insufficient when high-LR exploration is shortened too aggressively. This schedule preserves most of the baseline plateau while guaranteeing both decay phases occur within 300 seconds.

**Reasoning**: The baseline's first decay occurs around 83% and its second is unreachable; EXP-001 decayed from 15% and regressed. A 70/90 split interpolates conservatively between those observations, adds about 3,800 updates at `0.001`, and removes the Nesterov confound. Discrete phases also match the optimizer behavior for which the ResNet20 baseline was originally tuned.

**Sources**: `experiments/001/04-analysis.md`; `goals/maximize-cifar10-best-test-accuracy/03-experiment-learnings.md`; current `train.py` milestones.

**Estimated Effort**: low

**Risk Assessment**: Moving the first drop from 83% to 70% may still shorten exploration too much, while only 10% at `0.001` may be insufficient. The schedule is nevertheless a clean, low-risk causal test with baseline-like dynamics.

### Wider PreAct ResNet in Eager FP32
**Summary**: Replace the post-activation width-16 ResNet20 with a full-preactivation width-32 ResNet20, but keep eager FP32, batch 128, CPU crop/flip, standard SGD, baseline weight decay, persistent workers, and bounded evaluation. Use the conservative 70/90 time-fraction milestones so the larger model receives a complete schedule without the prior compilation, BF16, large-batch, or GPU-augmentation bundle.

**What it targets**: The baseline model has only 269,722 parameters and 330 MB peak VRAM on a 98 GB H20. Added representational capacity is an orthogonal route past the apparent generalization ceiling of the small architecture.

**Reasoning**: The EXP-001 reviewer assigned the original capacity idea the highest raw upside but rejected its six-way systems bundle. This version implements the reviewer's requested de-bundling: architecture plus the minimum horizon correction, with no precision, compilation, batch, augmentation, or weight-decay changes.

**Sources**: `experiments/001/01-idea-review.md` feedback 2 and scored verdict; `experiments/001/proposals/idea-03.md`; baseline parameter/VRAM measurements.

**Estimated Effort**: medium

**Risk Assessment**: A roughly 4x larger model may complete fewer optimizer steps in 300 seconds and may need different weight decay or LR tuning. Architecture and schedule still change together, so a neutral result would require throughput and loss-trajectory diagnosis.

### Long-Plateau Cosine with Standard Momentum
**Summary**: Hold `lr=0.1` for 65% of counted training, then cosine-decay to `1e-4` over the final 35%, using the baseline's standard momentum rather than Nesterov. Retain EXP-001's persistent-worker and seven-evaluation protocol controls.

**What it targets**: This directly tests the inferred EXP-001 failure mechanism by restoring most high-LR exploration while retaining a smooth low-LR tail and removing the Nesterov confound.

**Reasoning**: EXP-001's final accuracy remained monotonic as LR approached zero, so late annealing was locally useful; the problem was likely starting it at 15%. A 65% hold provides roughly 25,000 high-LR steps and about 13,000 annealed steps, with several thousand below `0.01`. SGDR supports smooth horizon-aware annealing, while standard momentum isolates the schedule from the failed combined operating point.

**Sources**: `goals/maximize-cifar10-best-test-accuracy/knowledge/papers/sgdr.md`; `experiments/001/04-analysis.md`; `experiments/001/01-idea-review.md`.

**Estimated Effort**: low

**Risk Assessment**: The 65% breakpoint remains a hand-selected operating point, and smooth decay may still spend too few updates at a stable minimum LR. A failure would not fully distinguish hold length from cosine shape.

## Review

The external review in `01-idea-review.md` selected Long-Plateau Cosine with Standard Momentum. Its key correction is to anchor the high-LR plateau at 78-80%, close to the baseline's known-good 83%, rather than interpolating from the failed 15% point. Adopt an 80% hold, step to `0.01`, and cosine-anneal from `0.01` to `1e-4` over the final 20%. This preserves the exploration and low-LR regimes that reached `91.67%`, adds guaranteed deeper refinement, and removes the Nesterov confound.

The reviewer rejected the 70/90 milestone schedule as lower-upside because its hard drop sacrifices exploration abruptly. The wider preactivation model retains the highest ceiling but is deferred: near-zero EXP-001 train loss indicates generalization rather than capacity is the current limiter, and a future wider-model attempt should first gate on a measured short-run throughput projection.

## Idea Evaluation

Adopt the scored verdict from `01-idea-review.md`. Long-Plateau Cosine received the strongest evidence/reasoning score (`8/10`) because it changes the failed 15% hold materially, preserves the validated smooth tail, and restores standard momentum. Refining the hold to 80% strengthens it further by matching the only schedule already known to reach the baseline while adding the previously missing terminal refinement.

## Chosen Idea
**Selected**: Long-Plateau Cosine with Standard Momentum

**Why this idea**:
It is the most direct response to EXP-001: retain persistent-worker and bounded-evaluation protocol gains, remove Nesterov, preserve the baseline's high-LR exploration for 80% of counted time, then spend the full final 20% at or below `0.01`. It tests a materially different operating point without introducing architecture, data, loss, or dependency changes.

**Hypothesis**:
Holding `lr=0.1` for 80% of the fixed 300-second training budget, then stepping to `0.01` and cosine-decaying to `1e-4` under standard momentum, will raise `best_test_acc` from `91.67%` to at least `91.77%`. The long plateau should retain baseline generalization while the final roughly 7,700 updates preserve the baseline's `0.01` refinement and add the deeper phase that the absolute 48,000-step milestone never reaches.
