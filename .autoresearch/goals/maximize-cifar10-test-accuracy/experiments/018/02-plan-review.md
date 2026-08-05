# EXP-018 Plan Review

1. **Independent gate seed can become a score-affecting reroll knob.** The proposed literal 18018 was preregistered but not mechanistically special. Use the project's fixed seed 42 inside the restored CPU RNG fork and never vary it, so the new parameters derive from the user-mandated seed rather than a new optimization axis.
2. **The expected accuracy margin is narrow and removal has uncertain sign.** Gate 0's mean scale of 0.6468 may have contributed despite low example dependence. Keep this as the experiment's explicit causal risk; the single-run/no-reroll rule remains mandatory.
3. **Synthetic throughput is a feasibility proxy, not goal verification.** Keep the retention guard because recovering runtime is part of the treatment, but do not classify a throughput projection as research success. Accuracy and hard constraints remain authoritative.
4. **Two-decimal threshold needs discrete-metric justification.** prepare.py computes integer correct predictions over exactly 10,000 CIFAR-10 examples, so accuracy changes in exact 0.01-point increments. A printed 94.17 represents 9,417 correct and exactly +0.10 over 9,407/94.07; there is no hidden 94.165 rounding case.
5. **Avoid exposing a new tunable seed constant.** Use the existing fixed seed 42 locally inside the restored fork, document it, and close seed changes for this treatment.
