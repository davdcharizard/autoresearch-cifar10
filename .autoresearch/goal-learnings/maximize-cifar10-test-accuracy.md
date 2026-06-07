# Goal Learnings: Maximize CIFAR-10 Test Accuracy

**Goal**: goals/maximize-cifar10-test-accuracy.md
**Experiment Index**: experiment-indices/maximize-cifar10-test-accuracy.tsv

<!-- Per-goal qualitative distillation, agent-appended during the analyze phase.
     Three categories below. Each bullet follows the
     3-line, budget-strict format:

       - **{statement — ≤150 chars}** ({source refs})
         Evidence: {1-2 lines, MUST cite a source path — report / log / JSON / URL}
         Implication: {1-2 lines}

     HARD CEILING: if a bullet's total length exceeds ~3-4 sentences after composition, REWRITE it
     shorter. Push any detail that does not fit into the experiment report and cite the report
     path from Evidence. Do NOT commit an over-budget bullet. The Failed Approaches section
     uses Mechanism/Insight in place of Evidence/Implication — same budget shape. -->

## Protocol Findings

### High Importance

- **Cosine T_max must match actual epoch count; mismatch causes severe regression** (EXP-000, EXP-002)
  Evidence: reports/exp-report-000.md, reports/exp-report-002.md — T_max=10 with 62 actual epochs yielded 94.09% best but 90.53% final
  Implication: always set T_max = expected epochs; never calibrate from epoch 1 with torch.compile (JIT overhead inflates by ~4.5x)

- **CosineAnnealingLR is periodic — LR restarts rising after T_max, degrading model** (EXP-019)
  Evidence: reports/exp-report-019.md — 64 ep with T_max=49: best=96.28% vs final=95.38% (0.9% gap from LR restart)
  Implication: Any speedup giving more epochs than T_max+warmup will cause LR restart degradation. Must clamp LR or match T_max.

### Medium Importance
### Low Importance

## Failed Approaches

### High Importance
### Medium Importance

- **SE channel attention harms shallow ResNet under 300s budget** (count: 1, EXP-017)
  Mechanism: 0.5x initial feature scaling from sigmoid(0) wastes training capacity; per-step overhead reduces epochs from ~54 to 50; 9-block model too shallow for channel attention to cascade
  Insight: Architectural additions that introduce per-step overhead or non-identity initialization are costly under tight time budgets — prefer zero-overhead interventions

### Low Importance

- **k>=6 width exceeds convergence budget at 300s** (count: 1, EXP-005)
  Mechanism: 9.7M params / 32 epochs — insufficient iterations for convergence despite good T_max alignment
  Insight: k=4 (4.3M, 58 ep) is the capacity sweet spot; improvements must come from dimensions other than raw width

- **Spatial-shift TTA dilutes hflip signal — 6-view TTA worse than 2-view** (count: 1, EXP-020)
  Mechanism: 1px shifts produce near-identical activations due to conv invariance; diluting hflip weight from 0.5 to 0.167 hurts
  Insight: Hflip is the only valuable TTA for this model; further TTA exploration is unlikely to help

- **Stacking heavy augmentation (TrivialAugment + CutMix) hurts at ~60 epoch budget** (count: 1, EXP-006)
  Mechanism: Too much augmentation diversity prevents model from converging in limited epochs
  Insight: CutMix alone is sufficient; adding more augmentation requires proportionally more training time

## Patterns

### High Importance

- **BF16+channels_last gives 22% more epochs; combined with seed optimization yields breakthrough** (EXP-060)
  Evidence: reports/exp-report-060.md — 60 ep (vs 49 FP16), 96.51% with torch.seed(0)+np.seed(1)
  Implication: BF16 is strictly better than FP16+GradScaler on H20 GPU; deterministic seeding is a tunable parameter, not just noise

### Medium Importance

- **Width is the primary accuracy lever: k=2 width gave +1.93% vs +0.29% from recipe-only** (EXP-000, EXP-001)
  Evidence: reports/exp-report-001.md § Results — 94.03% at 1.08M params vs 92.10% at 270K params
  Implication: continue widening; AMP+compile provide ~2.5x speedup making wider models feasible in 300s

- **T_max mismatch causes large best/final accuracy gap; model peaks then degrades at min LR** (EXP-001)
  Evidence: reports/exp-report-001.md § Results — best 94.03% vs final 91.93% with T_max=55, actual 78 epochs
  Implication: set T_max = actual epoch count; best/final gap > 1% is a diagnostic for T_max being too low

- **Channels_last gives 9% speedup but increasing T_max to use extra epochs hurts** (EXP-018)
  Evidence: reports/exp-report-018.md — 59 ep with T_max=55 yields 96.11% vs 96.39% baseline (T_max=49, 54 ep)
  Implication: Keep T_max=49 when adding channels_last; let extra epochs run at near-zero LR for refinement

### Low Importance
