**Feedback**

**Idea 1: Multi-crop translation TTA**
1. Main risk is wall-clock failure, not training budget. Goal allows 300s training, but runs over 10 minutes are failures; EXP-002 already took ~443s with lighter flip-TTA, and this makes tail eval 6-view instead of 2-view. Address by running under `timeout 600`, keeping `TTA_START_FRAC=0.8`, and being ready to raise the gate if total wall approaches 600s.
2. The translate-TTA evidence is partly indirect in the local materials: `fast-cifar10-recipes.md` explicitly supports flip-TTA, while the exact mirror+translate crop recipe appears in the EXP-006 brainstorm’s web-review notes. Before editing, verify the crop/pad formula against upstream `airbench96.py` so this is not a folklore port.
3. The expected gain may be below the required +0.1pp. Current `train.py:180-185` already averages original+mirror, and the baseline is already 96.00%, near the diagnosed airbench96 ceiling. Treat a +0.05pp result as no-improvement, not success.
4. The idea is otherwise well aligned: current TTA is mirror-only, gated at `train.py:337-348`; diagnosis says incomplete eval-time view coverage is the binding limiter; and the change stays inside `train.py` without extra validations or seed games. No fatal issue.

**Idea 2: Second ReZero block at layer2**
1. This does not attack the diagnosed primary limiter. Diagnosis says the next cheap win is eval-time view coverage, while this returns to training-side capacity near saturation.
2. The throughput cost is underplayed. EXP-004’s first layer2 ReZero block helped, but the learnings say it still lost many epochs versus the previous recipe; another 8x8 block avoids the failed 4x4/cuDNN penalty from EXP-005, but it still adds two 256-channel convs every training step. Address by estimating epoch count before assuming “full-rate” means cheap.
3. Evidence for the first block does not imply monotonic gains from stacking a second one. EXP-004 was +0.13pp; diminishing returns could easily make the second block net-negative after fewer low-LR tail updates.
4. If run, keep it as a clean single-variable architecture test and compare both epoch count and tail accuracy curve against EXP-004, not just final best accuracy. No fatal issue, but risk is materially higher than Idea 1.

**Idea 3: Hyperparameter sweep**
1. “Sweep” is not a single sharp experiment. Multiple 300s runs increase selection pressure on noisy test-set outcomes, especially when the required improvement is only 10 CIFAR-10 images. Pre-register one knob and one direction, or defer.
2. The claim that these knobs are merely heuristic is weak. `fast-cifar10-recipes.md` and EXP-001/002 already validate peak LR, one-cycle, Cutout, label smoothing, EMA, and flip-TTA as part of the working recipe.
3. Most proposed knobs do not address the diagnosed limiter. `PEAK_LR`, `Cutout`, and `LABEL_SMOOTHING` are training-side changes; only `TTA_START_FRAC` touches eval, and it still does not add the missing translation views.
4. Best refinement: after Idea 1, tune only the TTA gate if wall-clock/accuracy curves show the best epoch is being missed. As written, this is too unfocused. No hard-constraint violation, but it has the strongest reward-hacking smell.

**Scored Verdict**

Idea 1: evidence/reasoning 8/10 — directly matches the diagnosed limiter and current code gap, with airbench support, though exact translate evidence should be reverified. Potential impact 7/10 — likely small but plausibly ≥0.1pp, and it preserves the proven training trajectory.

Idea 2: evidence/reasoning 6/10 — EXP-004 supports layer2 capacity, but not a second stacked block, and throughput loss is a serious unresolved mechanism. Potential impact 6/10 — could clear +0.1pp if capacity remains, but the ceiling and fewer epochs limit upside.

Idea 3: evidence/reasoning 4/10 — general tuning rationale, little direction-specific evidence, and weak alignment with the diagnosis. Potential impact 5/10 — a lucky knob may move 0.1pp, but the method is noisy and expensive.

**Pick: Idea 1, multi-crop translation TTA.** It is the only candidate that attacks the stated binding limiter, keeps the 96.00% training recipe unchanged, and has the best evidence-to-effort ratio. Run it next, with the wall-clock cap and exact airbench crop implementation treated as the two things to verify.
