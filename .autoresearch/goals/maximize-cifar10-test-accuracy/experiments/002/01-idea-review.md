## Prioritized Feedback

Baseline facts: EXP-001 reached **95.22%** with **192 epochs / 300s training / 447.4s wall** and only **1.6 GB VRAM** [report](/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/.autoresearch/goals/maximize-cifar10-test-accuracy/experiments/001/04-analysis.md:19). The frozen eval does `model.eval()` then `outputs = model(inputs)`, so model-contained TTA and `AveragedModel` are feasible [prepare.py](/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/prepare.py:32).

1. **Idea-02, EMA + flip-TTA: wall-clock risk is tighter than stated.**  
   EXP-001 had only ~153s wall headroom. Running mirror TTA for all ~190 evals could push close to the 600s cap, especially because eval is outside `training_seconds`.  
   **Fix:** gate flip-TTA to the final tail, e.g. `progress >= 0.80` or `0.85`, where EXP-001 actually produced best accuracy. Keep one `evaluator.evaluate` call per epoch.

2. **Idea-02: EMA can regress below baseline if it replaces raw eval too early or lags the low-LR tail.**  
   The proposal correctly notes that after warmup only EMA is evaluated, so a bad decay can cap `best_acc` below 95.22.  
   **Fix:** keep `LABEL_SMOOTHING=0.2` unchanged, use EMA as a short-horizon tail average (`0.998`, possibly `0.997` if lag appears), and only evaluate EMA after it has enough updates. If implementing a fallback variant, TTA-only is the cleanest floor-preserving fallback.

3. **Idea-02: no fatal frozen-eval issue.**  
   I verified local PyTorch 2.9.1 `AveragedModel.forward` delegates to `self.module(*args, **kwargs)`, so the wrapped `ResNet9.forward` TTA path is reachable. `use_buffers=True` also averages BN buffers; non-floating BN counters are not relevant for eval with default BN momentum.  
   **Fix:** add a small smoke check that `evaluator.evaluate(ema_model, device)` runs and that `ema_model.module.training` becomes `False`.

4. **Idea-01, whitening conv: strong precedent, but the marginal-benefit assumption is weak here.**  
   Whitening is most valuable in epoch-starved fast-CIFAR recipes. This baseline already gets ~192 epochs and reaches 95.22 without whitening, so the likely gain is a few tenths at best and could land inside noise.  
   **Fix:** run whitening-only first, not whitening+GELU, to keep attribution clean and avoid confounding a narrow +0.1pp bar.

5. **Idea-01: normalization consistency is load-bearing and must not drift.**  
   `train.py` and frozen eval use mean subtraction with `std=(1,1,1)` [train.py](/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/train.py:29). Airbench-style channel-std whitening would silently desync the filters from eval inputs.  
   **Fix:** compute whitening patches with exactly `ToTensor()+Normalize(EVAL_MEAN, EVAL_STD)`, assert the constants match `prepare.py`, and do not import airbench’s different normalization convention.

6. **Idea-01: implementation failure modes are real but fixable.**  
   The 2x2/pad-0 whitening conv shrinks 32→31, which breaks the existing final `MaxPool2d(4)` after the pool chain. Init can also be overwritten by `self.apply`.  
   **Fix:** make `AdaptiveMaxPool2d(1)` mandatory, initialize whitening after model construction/device move, freeze it, and filter optimizer params by `requires_grad`.

7. **Idea-01: mild reward-hacking smell if startup preprocessing becomes “free training.”**  
   A small eigendecomposition on 5000 images is acceptable, but the proposal explicitly relies on doing it before the training timer.  
   **Fix:** cap the subset, keep it cheap, print `startup_seconds`, and avoid expanding this into large off-budget data processing.

8. **Idea-03, 1.5x wider net: VRAM headroom is not proof that capacity is the bottleneck.**  
   The diagnosis says the remaining gap is mixed: robustness, capacity, and whitening. Pure width trades away the update count that produced EXP-001’s late gain.  
   **Fix:** if running width, prefer 1.25x or 1.33x as the first variant unless a throughput smoke test predicts plenty of epochs for 1.5x.

9. **Idea-03: same LR/WD recipe is plausible but not guaranteed at 1.5x.**  
   Wider nets often tolerate the same schedule, but `PEAK_LR=0.4`, label smoothing 0.2, and weight decay 5e-4 were validated on the 6.57M-param base, not 14.8M params.  
   **Fix:** keep the recipe unchanged for attribution, but watch epoch-1 loss and final-tail slope; do not add the optional residual-depth change to 1.5x in the same run.

## Scored Verdict

| Idea | Evidence & Reasoning | Potential Impact |
|---|---:|---:|
| **Idea-01: whitening conv** | **7/10** — strong airbench precedent and good implementation awareness, but weaker evidence that whitening still helps after 192 baseline epochs. | **6/10** — credible +0.1 to +0.4pp, but narrow margin over the 95.32% bar. |
| **Idea-02: EMA + flip-TTA** | **8.5/10** — mechanisms are standard, orthogonal, and frozen-eval compatibility is verified; main unresolved issues are wall time and EMA decay. | **8/10** — plausible +0.3 to +0.7pp, enough margin over the bar without perturbing training dynamics. |
| **Idea-03: 1.5x wider net** | **6.5/10** — WideResNet reasoning is sound, but the capacity-bottleneck claim is less proven than the proposal implies. | **7.5/10** — real upside if capacity-bound, possibly toward 96%, but pure width may wash out or under-converge. |

**Pick for EXP-002: Idea-02, EMA + flip-TTA.**

It has the best expected value: it directly improves the evaluated iterate and prediction robustness, works through the frozen `model(inputs)` interface, and leaves the 95.22% training recipe mostly intact. Run it with the concrete fixes above: no label-smoothing change, short-horizon EMA, and final-tail-gated TTA to protect the 600s wall cap.
