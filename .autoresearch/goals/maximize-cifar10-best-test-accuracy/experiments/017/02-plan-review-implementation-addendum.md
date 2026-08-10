# EXP-017 Implementation Addendum Review

I've read the plan, goal, brainstorm, idea-01, both controllers, `train.py`, and `prepare.py`, and empirically verified the load-bearing invariants (shared-state equality, RNG equality, param count 1,084,586, seed-42 projection draws, scope = `train.py` only, 202-batch CutMix = 0.50, 8-worker shutdown). No blocking correctness issue found. Prioritized observations below.

**Verdict: APPROVED** - no blocking correctness defect. Every load-bearing structural/RNG/param/scope claim reproduces exactly on execution. The concerns below are hardening notes (mostly false-FAIL / attribution risks), not false-PASS defects that would let an invalid metric through.

## 1. Tight timing dispersion gates can false-FAIL a correct candidate (`timing_shortcut.py:136-137`, `140`)

`candidate_cv < 0.03`, `ratio_cv < 0.02`, `inference_cv < 0.03` are computed as population CV over only 5 trial means. On a shared H20 these are plausibly tripped by benign clock/thermal jitter unrelated to the candidate. Because `controller()` `raise SystemExit("TIMING_GATE_FAIL")` on any single gate, one noisy trial aborts the whole experiment even when the true overhead is well under 1.0548. This is the highest-probability way the run never launches. It does not corrupt a result (fail-safe), but the researcher should expect possible spurious aborts and be ready to re-run the timing controller (which is legitimate - timing is not the scored metric) rather than treating the abort as a design defect.

## 2. `projected_steps` is a synthetic-batch estimate; the real run can land < 25,500 (`timing_shortcut.py:118`, plan step 8)

`projected_steps = floor(26_898 / ratio)` is derived from a single cached synthetic batch reused for all 500 steps. Real production step time (counted `dt`, `train.py:253-278`) excludes loader wait - which the benchmark correctly mirrors - so the estimate is reasonable, but at ratio approximately 1.0548 the projection sits exactly on the 25,500 floor (26898/1.0548 = 25500.5). Any real-run jitter pushes actual steps below the floor. The plan already handles this (step 8 makes <25,500 "accuracy-only, attribution-weak," not invalid), so this is not a false-PASS - just flag that a formal 94.25 pass will very likely carry the weak-attribution caveat given how close the threshold is to the floor.

## 3. Init draws are coupled to the global seed-42 stream (`train.py:92-93`)

`shortcut_generator.manual_seed(torch.initial_seed())` seeds the dedicated generator with the same value (42) that seeds the main stream. This is the reviewed/adopted fix (correctly avoids global-stream perturbation - `rng_equal` verifies True), so it is intentional, but note the residual property: both projection weights are a deterministic function of the same seed-42 initial state as the stem/shared convs, i.e. not statistically independent of them. Negligible for training and explicitly accepted; raising only so it is not mistaken for independence in the attribution write-up.

## 4. `expected_evals` projection vs. the hard <=19 cap (`timing_shortcut.py:126-127`, plan step 8)

`tail_epochs = ceil(60 / (390 * candidate_mean/1000))` assumes exactly 390 full steps per tail epoch and derives `expected_evals = 4 + tail_epochs`. The candidate is strictly slower than control, so it can only produce fewer tail epochs/evals than EXP-010's 19 - there is no inflation path, and step 8's numeric `<=19` recount is the real guard. Sound; noting only that the projection is a soft pre-check and the run-time recount is authoritative.

## 5. First-update/RMS safety gates are load-bearing but single-batch (`preflight_shortcut.py:229-258`)

The hard/soft first-update gates each use exactly one batch (`next(batch ... ndim==1/==2)`). This matches the plan's declared protocol and the 25%-update-norm / replay / concentration gates are the correct load-bearing checks (RMS band correctly demoted to a tripwire per the plan review). No defect - confirming the hooks are order-correct (bn2 fires before shortcut within each block, `residuals`/`shortcuts` zip aligns layer2 to layer3) and the lambdas capture no loop variable, so there is no late-binding aliasing bug.

## Verified clean

- **Scope:** `git diff --name-only` = `train.py` only; diff is exactly the ShortcutConv/init/forward change, nothing out-of-bounds. `prepare.py`/`Eval` untouched.
- **RNG/shared-state:** candidate vs `ControlResNet` - `shared_state_equal=True`, post-construction `rng_equal=True`, so the production data-shuffle stream is bit-identical to the accepted architecture; `ShortcutConv` early-return in `_weights_init` correctly prevents both weight-overwrite and stream advance.
- **Param/topology:** 1,084,586 total, 1,073,962 control (+10,624), 2 transition shortcuts / 7 identities, projection draws match a fresh seed-42 generator in layer2 to layer3 order.
- **Loader lifecycle:** 8 workers materialized and shut down cleanly; 202-batch CutMix rate = 0.500 (comfortably inside [0.45,0.55]) - the tightest deterministic band passes.
- **Timing fidelity:** benchmarked `training_step` reproduces production's counted region (H2D to zero-grad to forward to CE to backward to SGD to sync, loader excluded), matching `train.py`'s `dt` accounting, so `ratio` and `projected_steps` are measured on a faithful interval.

## Provenance

- Reviewer: external Claude CLI, mandatory no-fallback path
- Command outcome: exit code 0
- Completed: 2026-08-06T09:23Z

