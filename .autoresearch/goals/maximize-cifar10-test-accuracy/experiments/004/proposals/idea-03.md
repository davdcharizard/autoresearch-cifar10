# Proposal idea-03: Whitening-enabled one-cycle retune (shorten warmup + modest peak-LR raise)

## One-line
Now that the input is ZCA-whitened (EXP-003), exploit the better-conditioned loss surface with a tight, two-knob one-cycle retune — **shorten the warmup (`PCT_START` 0.15 → 0.10)** and **modestly raise the peak (`PEAK_LR` 0.4 → 0.45)** — to spend more steps in the productive low-LR annealing tail where all prior gains concentrated, while leaving whitening + EMA + flip-TTA untouched.

## Named limiter (from the diagnosis)
EXP-001/002/003 all converged on the same trajectory fact: **the bulk of accuracy arrives in the low-LR annealing tail** (EXP-001 analysis: "the bulk of the gain came in the low-LR tail — direct confirmation of the under-annealing diagnosis"; EXP-002's tail step-up; EXP-003 peaked at ep162 of 174). The schedule allocates ~15% of the 300s budget (~26 epochs) to a linear ramp that is mostly *pre-productive* — early high-LR steps on a now-whitened input are no longer the bottleneck (EXP-003 already hit 60.19% at ep1). EXP-003's explicit #1 next step names the lever: a better-conditioned input "tolerates larger steps; the one-cycle peak could go above 0.4 for a compounding gain (held fixed here for a clean A/B)." The limiter this proposal targets: **the schedule was tuned for un-whitened inputs and now under-uses the conditioning headroom** — warmup is longer than the whitened net needs, and peak LR is conservative.

## Mechanism (causal chain change → metric)
1. **Whitening sphereizes the first-layer loss surface** (EXP-003 confirmed: large early-epoch lead, ep1 60% vs 57%). A better-conditioned surface has a smaller curvature spread, so a given LR is further from the divergence threshold — the net *tolerates a larger peak step* without instability (this is exactly the airbench design logic, arXiv:2404.00498, where whitening is the enabling trick for aggressive schedules).
2. **Raising `PEAK_LR` 0.4 → 0.45** increases effective exploration in the high-LR phase, which (under one-cycle theory / super-convergence) reaches a flatter, lower-loss basin before annealing — provided it does not diverge. This is a *modest* +12.5% raise, deliberately well short of the 0.5–0.6 upper candidate, to stay inside the conditioning headroom margin and keep run-to-run noise from being dominated by a near-divergence tail.
3. **Shortening `PCT_START` 0.15 → 0.10** reallocates ~5% of the budget (~9 epochs at 174 ep/run) from the pre-productive ramp into the annealing decay. Because `lr = PEAK_LR * (1 - progress)/(1 - PCT_START)` for the decay phase, a smaller `PCT_START` makes the decay *start earlier and span more of the budget* — i.e. more steps spent in the mid-and-low-LR regime that EMA denoises and that historically produced the gains. The whitened net no longer needs the long warmup (early convergence is already fast), so this reallocation is close to free on the early side.
4. **Net effect on `best_test_acc`**: a marginally lower / flatter tail minimum (from higher peak exploration) sampled by more annealing steps (from shorter warmup), denoised by the unchanged EMA and read with unchanged flip-TTA → a modestly higher annealed tail, i.e. the same compounding-tail mechanism EXP-001→003 each rode for tenths of a pp.

## Concrete change (in THIS codebase)
All edits in `/SPXvePFS/users/david/autoresearch-benchmarks/autoresearch-cifar10/runs/v3.0.0-opus-4-8/train.py`. Two constant changes; the LR-computation block and everything else stay byte-identical.

- **Line 21**: `PEAK_LR = 0.4` → `PEAK_LR = 0.45`.
- **Line 25**: `PCT_START = 0.15` → `PCT_START = 0.10`.

That is the entire change. Note these two constants feed:
- The optimizer init (`lr=PEAK_LR`, line 224) — harmless, immediately overwritten per-step.
- The per-step one-cycle LR computation (lines 264-268), unchanged:
  ```python
  progress = min(1.0, total_training_time / TIME_BUDGET_S)
  if progress < PCT_START:
      lr = PEAK_LR * progress / PCT_START          # ramp 0 -> PEAK over first PCT_START
  else:
      lr = PEAK_LR * (1.0 - progress) / (1.0 - PCT_START)   # decay PEAK -> 0 by progress=1
  ```

**Interaction to be careful about — EMA warmup coupling.** `EMA_WARMUP_FRAC = 0.15` (line 29) is documented as "start EMA once LR ramp completes (matches PCT_START)" and the loop gates EMA on `progress >= EMA_WARMUP_FRAC` (line 286). If `PCT_START` drops to 0.10 but `EMA_WARMUP_FRAC` stays 0.15, EMA now starts *after* the ramp completes (at 15% while the peak is at 10%), so EMA would begin partway down the decay rather than at the peak. This is benign (EMA still has ~85% of the budget / ~148 epochs to average a denoised tail, and starting slightly later avoids averaging in the highest-LR noisiest iterates) — but to keep the change clean and the "EMA starts at ramp completion" invariant intact, **also set `EMA_WARMUP_FRAC = 0.10` (line 29)** so EMA again begins exactly when the ramp ends. This keeps the two knobs being tested (peak, warmup length) cleanly attributable and preserves the design intent. This is the only secondary edit and it is mechanically forced by the primary one, not an extra free parameter.

No new constants, no new deps, seed `torch.manual_seed(42)` (line 178) untouched, single `evaluator.evaluate` per epoch (line 327) untouched, whitening / EMA / flip-TTA all stay ON.

## Evidence
- **Direct prior-experiment evidence (strongest):** EXP-003 analysis (`experiments/003/04-analysis.md`), "Unexplored Avenues" and "Next Steps": names "Whitening + raised PEAK_LR ... a better-conditioned input tolerates larger steps; the early-convergence headroom suggests the one-cycle peak could go above 0.4 for a compounding gain (held fixed here for a clean A/B). Plausible next tenth." This proposal *is* that pre-registered next step, executed conservatively.
- **Tail-is-the-lever, repeatedly:** EXP-001 ("the bulk of the gain came in the low-LR tail"), EXP-002 (tail step-up at the TTA gate), EXP-003 (peak at ep162/174). Reallocating budget into the tail via shorter warmup is therefore aimed at the empirically dominant phase, not a guess.
- **airbench schedule precedent for whitened inputs:** the airbench *96* variant (the higher-accuracy regime closest to ours) explicitly **reduced warmup duration and let LR decay all the way to zero** relative to airbench94 (confirmed via the airbench repo / arXiv:2404.00498 schedule notes). airbench's whole design premise is that the frozen whitening conv is the enabler for an aggressive triangular schedule with short warmup. Our front-end is the same frozen ZCA conv, so the same "short-warmup, decay-to-zero" schedule shape is the matched precedent.
- **Code fact — decay-to-zero is already satisfied:** at `progress = 1.0`, line 267-268 gives `lr = PEAK_LR * (1 - 1)/(1 - PCT_START) = 0` exactly. So candidate lever (c) "ensure decay-to-zero" is **already true in this codebase** and requires no change — I deliberately drop it from the proposal rather than pad the change. This is an honest scoping note: only (a) and (b) are real, available knobs here.
- **Headroom for higher LR is plausible, not proven:** EXP-001 ran PEAK_LR=0.4 with no divergence on the *un-whitened* net; whitening can only improve conditioning, so 0.45 is very unlikely to destabilize. The literature one-cycle peak for this net class spans ~0.4–0.6 in the mean-loss convention (`fast-cifar10-recipes.md`, johanwind), so 0.45 is squarely inside the documented stable band.

## Strongest risk
**The effect size may not clear the 0.1pp bar robustly given run-to-run noise.** This is the central honest concern, and it has two faces:
1. **Magnitude ceiling.** EXP-003 already runs a *fully annealed* 174-epoch schedule — its own analysis warned "whitening's benefit compresses in a fully-annealed regime." The conditioning headroom may already be largely consumed by the existing 0.4 peak over 174 epochs, leaving only a sliver for a schedule retune. A realistic central estimate is **+0.05 to +0.15pp**, with a real chance the true effect is below +0.1pp.
2. **Noise masking.** This is a single fixed-seed (42) one-shot run. Run-to-run / step-timing-induced variance in this setup is ~±0.05–0.1pp (the time-based schedule means epoch boundaries shift slightly run to run). A true +0.08pp effect could read as anything from −0.02 to +0.18pp. So even a genuinely positive change might not register as ≥95.97%, and conversely a noise-up could *fake* a pass. The verification framing must treat a marginal pass with appropriate skepticism and the mechanism trace (does early loss fall faster? does the tail start earlier and sit higher?) as the real evidence, not just the final number.

Secondary risk: raising peak LR could make the *tail noisier* (higher pre-anneal weights farther from the eventual basin), partially offsetting the EMA denoising. Mitigated by keeping the raise small (0.45, not 0.6) and by EMA being unchanged.

**Assumption that most needs to hold:** that there is residual conditioning headroom not already captured by the 0.4 / 174-epoch schedule — i.e. that the whitened net is genuinely under-stepped at 0.4. If the 0.4 peak is already near-optimal for the whitened surface, the change is a wash within noise.

## Expected effect on the training trajectory
- **Early (ep ≤ 25):** peak reached ~3% earlier in budget (10% vs 15%); slightly higher peak LR → early train loss should fall at least as fast as EXP-003 (which already hit 60% ep1 / 88.8% ep25), possibly a hair faster, possibly a touch noisier at the very peak. Watch for any loss spike/NaN around progress≈0.10 (the divergence canary).
- **Mid/tail (ep > 100):** decay phase starts ~5% earlier and spans more of the budget → LR is lower at any given late epoch than EXP-003, so the EMA-denoised, TTA-read tail should cross the bar earlier and ideally settle a hair higher. The headline `best_test_acc` is read off this tail.
- **Epoch count:** essentially unchanged (~174 ± a couple); no throughput change (same architecture, same per-step cost).

## Rough quantitative estimate
- **Central:** 95.87% → **~95.95%** (+0.08pp). **Optimistic:** ~96.0% (+0.13pp). **Pessimistic / null:** ~95.85% (−0.02pp, within noise).
- Honest read: this is the **cheapest** lever in the EXP-004 pool but plausibly the **lowest-ceiling** one — the central estimate sits *right at* the 0.1pp bar, so a clean pass is not guaranteed. Its value is partly as a near-free probe of whether schedule headroom remains before spending effort on higher-ceiling/higher-variance levers (capacity, Muon).

## Effort
**Low.** Two-to-three constant edits, one 300s training run (~7.5 min wall incl. eval, well under the 10-min kill), no new code paths, no new deps. Implementation + smoke (py_compile + one-batch forward) + run fits comfortably in one experiment loop. The dominant cost is the single run itself.

## Run command (for the planner)
`timeout 600 bash -c 'CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1'` (GPU 1 per project memory; GPU 0 reserved). Verify in the log: no NaN/divergence near progress≈0.10, `training_seconds ≈ 300`, `whitening_seconds` off-budget, exactly one eval line per epoch, summary `best_test_acc` == max per-epoch eval. Compare `best_test_acc` against 95.97% bar, and inspect the early-loss and tail-crossing-epoch trace against EXP-003 to attribute any delta to the mechanism rather than noise.
