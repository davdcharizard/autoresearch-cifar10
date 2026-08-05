1. **[Current `train.py` vs Plan Milestone 1]** Current `train.py:205-212` still has `Cutout(8)` and no `RandomErasing`; `git diff ae31206 -- train.py` is empty. Running the official command now would retest EXP-004 and any EXP-008 attribution would be invalid.

2. **[Throughput/budget verification, plan lines 27, 76]** The plan’s CPU-augmentation throughput check is blind to DataLoader wait time. In `train.py`, `t0` is set after `for inputs, targets in train_loader` yields the batch, so CPU transform stalls can be outside `training_seconds` and step `img/s`. A RandomErasing-induced worker bottleneck could add wall time/off-budget preprocessing while `num_epochs` and logged `img/s` look normal.

3. **[C3 metric-genuineness awk, plan line 71]** The cross-check takes max of per-epoch `best:` fields, not `test_acc:` fields. That is circular because `best:` is already the running `best_acc`; a bug or fabricated running-best print could match the summary while not proving the summary equals max evaluator accuracy.

4. **[C2 untracked scope check, plan line 63]** `git status --porcelain` will not show ignored files. This repo ignores `*.log`, `.autoresearch/`, `__pycache__/`, and `.venv` (`.gitignore:59`, `:225`, `:2`, `:153`), so the stated expectation that `run.log` appears is wrong, and ignored runtime-affecting Python/env files can evade this check.

5. **[C2 pass criteria vs confound handling, plan lines 66, 73, 75-77]** `num_epochs`/`img/s` are called “FIRST-CLASS” but placed under “Informational Metrics (Optional)” and omitted from the hard pass condition. A `BEST>=96.10` run with abnormal epoch count or wall-time behavior could still be labeled improvement, despite the plan saying throughput preservation is central to attribution.

6. **[RandomErasing strength semantics, proposal lines 48, 64; plan line 33]** The “second smaller erase” description is inaccurate. `scale=(0.02,0.15)` on 32×32 can erase up to ~154 pixels, larger than an unclipped 12×12 cutout’s 144 pixels, and Cutout is often clipped at borders. Overlap also means combined erased area is variable, so under-fit/regularization strength may be mis-attributed.

7. **[Param check command, plan line 65]** `grep '^num_params:' run.log` outputs the whole labeled line (`num_params:       7,784,627`), not exactly `7,784,627`. As written, the command is not an exact machine check, and param count alone would not prove architecture/forward/schedule integrity without the manual diff read.
