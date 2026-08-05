# Report EXP-001: Time-Aligned Pre-Activation WRN-16-2
- **Created**: 2026-07-24

## Goal

Increase CIFAR-10 `best_test_acc` under the frozen 300-second counted training budget, with higher values better. The baseline entering EXP-001 was 91.54%; success required at least 91.64%, a complete run below 10 minutes total, and compliance with the `train.py`-only intervention scope.

## Idea & Hypothesis

The chosen idea combined a canonical pre-activation WRN-16-2 with a learning-rate schedule keyed to counted training time. It was selected because the baseline used only 330.1 MiB of the H20 and reached its first LR decay after roughly 84% of realized steps while never reaching its second. The hypothesis predicted that moderate width, batch 256, selective decay, Nesterov SGD, and a complete warmup/cosine trajectory would exceed 91.64%, plausibly reaching at least 92.0%.

## Approach

`train.py` now uses six pre-activation residual blocks with stage widths 32/64/128, learned projection shortcuts, final BN/ReLU, and explicit initialization. Batch size increased to 256; convolution/linear weights receive `5e-4` decay while BN/bias parameters receive none; SGD uses momentum 0.9 and Nesterov. A 5% time warmup rises from 0.002 to 0.2, then cosine-decays to 0.002 by the 300-second boundary. Evaluation runs every fifth epoch plus the final epoch. Deterministic cuDNN benchmarking, efficient gradient clearing, and persistent DataLoader workers support throughput and reproducibility.

The persistent-worker setting was an execution-time deviation added after Run 1 exposed large per-epoch worker startup overhead. It does not change the model, evaluation harness, seed, or counted training budget.

## Execution

Two full attempts were launched locally on one NVIDIA H20. Run 1 produced a provisional 93.18% at epoch 130 but timed out at 600 seconds with only 91% counted training complete; recreating eight DataLoader workers on every epoch consumed the excluded wall-time margin. Run 2 enabled persistent workers and completed successfully in 342.5 seconds total. An earlier CUDA smoke test also caught and corrected a stale stem-BN reference before either full run.

Run 2 completed 28,540 optimizer steps and 147 epochs. It crossed the original baseline at epoch 120, peaked at 93.38% at epoch 145, and finished at 93.34%, showing the peak was retained rather than being a transient spike.

## Results

- **Primary metric**: 93.38% (baseline: 91.54%, delta: +1.84 percentage points, +2.01%)
- **Observations**: Peak VRAM rose from 330.1 MiB to 1,092.0 MiB but remained negligible relative to the 97,871 MiB H20. Batch 256 and high throughput yielded about 146 dataset-equivalent passes versus roughly 98 for the baseline, despite fewer optimizer steps. The final accuracy was only 0.04 points below the best.
- **Analysis**: The hypothesis is supported. The combined architecture, batch, optimizer, and time-aligned schedule substantially improved accuracy and fully used the low-LR convergence phase. The experiment does not isolate how much gain came from width versus schedule versus higher image exposure, but it establishes the bundle as a strong new baseline. Persistent workers were essential for validity under the separate 10-minute total limit but are not evidence for the accuracy mechanism.
- **Key Learning**: A moderate WRN plus budget-aligned cosine converts unused H20 capacity into +1.84 accuracy points; persistent workers are mandatory for high-epoch wall-time compliance.

## Verification

- **Conditions**: all passed
- **Review Notes**: Results confirmed trustworthy. Run 2 emitted a fresh complete summary, used the frozen evaluator on one H20, retained seed 42, modified only `train.py`, evaluated no more than once per epoch, and completed in 342.5 seconds total. The 1.84-point gain is well beyond the 0.1-point acceptance threshold.
- **Verdict**: improvement
- **Verdict Basis**: all necessary conditions passed and `best_test_acc` improved meaningfully from 91.54% to 93.38%.

## Unexplored Avenues

- Isolate the schedule contribution by retaining WRN-16-2 and testing nearby peak LR, warmup, and decay-floor settings; the current bundle is successful but not attribution-clean.
- Add low-overhead EMA or early-only mixup/Cutout to the proven WRN baseline; the final 93.34% suggests stable convergence with remaining generalization headroom.
- Test a modest additional width increase or AMP/compile acceleration; only 1.1 GiB VRAM was used, though throughput and fixed-time convergence must be re-measured.

## Next Steps

- **High confidence**: tune the successful WRN schedule around peak LR and decay floor, using EXP-001 as the 93.38% moving baseline.
- **Medium confidence**: add EMA as an isolated low-overhead generalization improvement while keeping the proven architecture and schedule fixed.
- **Medium confidence**: test early-only mixup or Cutout followed by clean cosine refinement, avoiding CPU-heavy RandAugment under the wall limit.

## Exit Action Results

No exit actions were defined for this local-only goal.
