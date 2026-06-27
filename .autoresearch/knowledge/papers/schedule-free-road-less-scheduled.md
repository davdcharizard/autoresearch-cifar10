# The Road Less Scheduled — Schedule-Free Optimization (arXiv 2405.15682, NeurIPS 2024)

**Authors**: Defazio, Yang, Khaled, Mishchenko, Mehta, Cutkosky (Meta FAIR)
**Reference impl**: https://github.com/facebookresearch/schedule_free (MIT)
**Status in project**: measured EXP-062 — **REFUTED in this regime**: at the 300s/13.5k-step horizon, eval-at-x read 94.87 (−1.84 vs baseline, mean−10.6σ) with a smooth monotone x-curve STILL CLIMBING at budget end (3 evals within 0.15 of best — no plateau). The any-horizon claim fails at short horizons: averaging a hot constant-lr trajectory cancels noise but does not reproduce the anneal's late-phase basin refinement. Implementation + BN-refresh machinery verified clean and reusable.

## Claim

LR schedules can be eschewed entirely: train at constant LR (after warmup) while maintaining
three coupled sequences, and the AVERAGED point matches or exceeds the endpoint of every
cosine schedule length simultaneously — on CIFAR-10 (SGD), CIFAR-100, SVHN, ImageNet, and a
broad suite. Theory unifies scheduling and iterate averaging (the linear-decay schedule is
shown equivalent to a particular online averaging); the schedule-free iterate provably tracks
the Pareto frontier that any single schedule can only touch at its own horizon. No new
hyperparameters vs momentum-SGD (lr, momentum β; warmup recommended).

## Exact algorithm (SGDScheduleFree, foreach path — verified from source 2026-06-11)

- `param.data` holds **y** during training; `state['z']` holds the SGD iterate; **x** is never
  stored — recovered by interpolation.
- Mode switches (in-place lerp, exact):
  - eval (y→x): `p.lerp_(z, weight = 1 - 1/β)`  [x = y/β − ((1−β)/β)z]
  - train (x→y): `p.lerp_(z, weight = 1 - β)`   [y = βx + (1−β)z]
- step k (params at y):
  1. warmup: `sched = min((k+1)/warmup_steps, 1.0)`; `lr = lr_base · sched`; `lr_max = max(lr_max, lr)`
  2. averaging weight: `weight = (k+1)^r · lr_max^weight_lr_power` (defaults r=0, weight_lr_power=2);
     `weight_sum += weight`; `ckp1 = weight/weight_sum`
  3. `grad += weight_decay · y` (decay applied AT y)
  4. `y.lerp_(z, ckp1)` then `y += grad · lr·(β(1−ckp1) − 1)`
  5. `z −= lr · grad`
- Defaults: β = 0.9, r = 0, weight_lr_power = 2.0.

## BatchNorm caveat (repo README)

BN running stats accumulate at y during training. Before eval, refresh them at x: switch
optimizer to eval mode (params = x), `model.train()`, forward-only over ~50 train batches
(`itertools.islice(train_loader, 50)`), then `model.eval()` and evaluate. PreciseBN or
closure-form optimizers also work. Maps onto this project's validated EXP-032 update_bn /
EXP-025/033 second-persistent-loader machinery; stats must come from the AUGMENTED train
distribution (EXP-029).

## Project-relevant readings

- The ONLY optimizer-schedule construction absent from the 61-experiment record; the schedule
  closures (EXP-010/014/016/032/049) are all anneal-family-INTERNAL.
- EXP-032's conclusion ("the time-keyed cosine already performs implicit iterate averaging")
  is the exact claim this paper formalizes and then exceeds — EXP-062 tests the converse the
  record never ran (averaging WITHOUT anneal).
- Max-statistic relevance: x's accuracy curve is smooth/monotone → potentially ~6× more
  near-ceiling evals than the cosine's last-10% plateau.
- Risk precedents: Muon EXP-028 (per-step optimizer gains decayed at plateau — but that claim
  was transit speed, this one is endpoint level at every horizon); EXP-029 (BN stats must
  match the evaluated weight point — addressed by the refresh).
