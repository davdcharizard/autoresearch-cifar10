1. **[Verification Protocol, `02-plan.md:115-118`]** `best_test_acc` is trusted from `grep` on `run.log`. A fake or accidental earlier `best_test_acc: 99.00%` print would pass before any proof it came from `Eval.evaluate`. This is a direct reward-hacking path; the log metric must be tied to the max per-epoch eval result.

2. **[Whitening builder, `02-plan.md:32-41`]** The `n_patches=50000` cap is applied after materializing all valid patches from 5000 images: about `4.5M x 27` floats, not 50k. This contradicts `02-plan.md:49`/`:122` and can hide large off-budget CPU/RAM work or fail the `<5s` startup check.

3. **[Scope verification, `02-plan.md:116` and `:119`]** Scope checking is under-specified. `git diff --name-only <integration_branch>` uses an undefined placeholder, while `git diff --quiet -- prepare.py` only catches uncommitted prepare changes. A committed or wrong-base change outside `train.py` can evade this.

4. **[Whitening correctness, `02-plan.md:37` vs `:54-56`]** The whitening covariance is built from unpadded interior patches, but the actual frozen conv uses `padding=1`. Border activations during train/eval include zero-normalized padding that is absent from the covariance estimate, so the plan’s “exact eval space” claim is false at borders.

5. **[Budget verification, `02-plan.md:115-116` and `:122`]** The verification trusts printed `training_seconds >= 295` and does not explicitly diff-check that the existing `torch.cuda.synchronize()`/`dt` budget accounting stayed intact. A bad edit could undercount real training time while staying under the 600s wall guard.

6. **[Off-budget timing text, `02-plan.md:89` and `:115`]** The plan says `total_seconds` excludes whitening, but the planned block is inside `main()` after current `t_start` and before `t_start_training`, so it should be included in `total_seconds`/`startup_seconds`. This ambiguity can mask misplaced timing code.

7. **[Abort Criteria, `02-plan.md:103-108`]** The abort policy contradicts itself: it says “kill and mark failed, not mutate-and-rerun,” then allows fixing a divergence bug and rerunning once. That creates a post-hoc tuning opening, especially for `eps`, scale, padding, or whitening placement.

8. **[Smoke test, `02-plan.md:16`]** The scratch smoke test calls the real frozen `Eval` twice before the official run. Even if intended only for finite-output checking, it is extra test-set access outside the run protocol and can be abused to debug or tune eval behavior.
