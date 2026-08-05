1. **[C3 genuineness, `02-plan.md:66`; `train.py:155,229,278`]** “Seeds unchanged” does not mean only layer2 width changed. Widening consumes a different CPU RNG stream during `self.apply`, so same-shaped downstream modules like `layer3`’s `Residual(512)` and `fc` get different initial weights; because model init occurs before the first `DataLoader` iteration, shuffle/augmentation worker seeds can also shift. At a ~0.1pp noise floor, this weakens the claim that any gain is attributable only to width.

2. **[Code Changes, `02-plan.md:25`; Brainstorm `01-brainstorm.md:91`]** The PEAK_LR=0.4 rationale leans too hard on ReZero identity-init. The widened `layer2[0]` and `layer3[0]` main-path convolutions are not identity-preserving, so a failure is not cleanly classified as only “under-annealing” or “capacity saturation”; it may be an LR/optimization mismatch.

3. **[Execution Environment / Diagnostics, `02-plan.md:26,34,42,70`]** The throughput risk is under-specified. The plan names the 8x8 GatedResidual and layer3 stem costs, but `layer2[0]` is widened before pooling at 16x16 and adds another substantial compute hit. The “115-130 epochs” estimate and “well below ~115” fallback threshold are fuzzy; `num_steps`/effective img/s need a concrete cutoff, not just `grep 'img/s' | tail -n 3`.

4. **[Milestone 1, `02-plan.md:11`]** `model.layer2[3].alpha` is the wrong module index. After the planned edit, `layer2 = Sequential(conv_bn, MaxPool2d, GatedResidual)`, so the gate is `layer2[2]`; following the plan literally raises an index error.

5. **[Milestone 2 / C1, `02-plan.md:14,50`]** The launch command does not actually create `run_exit.txt`. C1 can fail on a missing file or, worse, read a stale exit code from an earlier run unless the exact command writes `$?` immediately after `timeout`.

6. **[C2, `02-plan.md:59`; `train.py:21`]** The PEAK_LR verification command will false-fail if treated literally. `grep '^PEAK_LR' train.py` currently returns `PEAK_LR = 0.4  # mean-loss one-cycle peak...`, not exactly `PEAK_LR = 0.4`.

7. **[C2, `02-plan.md:60`]** The parameter cross-check is too loose as written. The expected total parameter count should be exact: `9,997,235` total params, not “≈10.0M”; also specify matching the comma-formatted `num_params:` summary, not learnable params.

8. **[C2 scope base, `02-plan.md:57-59`]** Diffing against `autoresearch/maximize-cifar10-test-accuracy-dev` is only valid while that branch remains at `ae31206` and the EXP-007 edit is uncommitted. Since the plan already identifies `ae31206` as baseline, using the mutable branch name can silently invalidate the scope check if the branch advances.

9. **[C2/C3 scope and reward-hacking, `02-plan.md:58-66`]** `git diff --name-only` ignores untracked files. A run could pass the tracked diff checks while an untracked importable file such as `sitecustomize.py` or another cwd shadowing module affects execution. The plan needs an untracked-file check with an explicit artifact whitelist.
