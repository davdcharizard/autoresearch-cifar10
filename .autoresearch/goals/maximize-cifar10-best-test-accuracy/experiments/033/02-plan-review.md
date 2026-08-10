I've read the plan, brainstorm, idea-03 proposal, goal definition, learnings, results TSV, `train.py`, and `prepare.py`. Below is the prioritized concern list.

---

## Review: EXP-033 Conservative Small-Area Random Erasing

### 1. Self-imposed geometry gates are calibrated at the exact floor/ceiling the policy can produce — a *correct* implementation will spuriously abort

**Plan §Verification step 3 / proposal gate 4: "conditional achieved area in `[0.015,0.105]` after integer rounding."**
With `scale=(0.02,0.10)` on a 32×32 image, the minimum requested area is `0.02·1024 ≈ 20.5 px`. torchvision floors `h,w` to integers, and at non-square ratios this routinely yields boxes like `5×3 = 15 px = 1.46%` or `4×4 = 16 px = 1.56%`. So individual erased examples will land **below the 1.5% lower bound** as ordinary rounding noise. If this gate is a per-example bound (the wording reads that way, distinct from the separate "conditional mean 4.5–7.5%" clause), the exact-corpus gate over ~6,400 erased examples is essentially guaranteed to see sub-1.5% boxes and abort — killing the run for implementing the policy faithfully. Either widen the lower bound (e.g. 0.010) or restate it explicitly as a distribution-percentile/mean bound, not a hard per-example floor.

**Plan §Config Changes / gate 4: "final per-image effective erased area <=20%."**
The theoretical worst case is `recipient-erased-outside-box (≤10.5%) + erased-donor-inside-box (≤10.5%) ≈ 21%`. The ceiling is set *below* the compound maximum the policy admits, so a rare but legal recipient+donor overlap trips it. Raise to ~22% or make it a distributional bound.

These two gates can invalidate a completely faithful run for arithmetic reasons unrelated to the hypothesis. That is the most damaging failure mode because it burns the entire ~50-min preflight + production and mislabels a valid policy as `invalid`.

### 2. "Exactly EXP010's 19 unique looks" is a timing-dependent evaluator-count gate

**Plan §Verification step 8: "require exactly EXP010's 19 unique looks including terminal and no duplicate epoch."**
The number of evaluations is a function of how `total_training_time/TIME_BUDGET_S` crosses epoch boundaries (`train.py:283–302`: 4 checkpoint evals plus one-per-epoch dense-tail evals from 0.8). Adding per-image erasing changes per-step wall time, which shifts how many epochs fall in the tail → 18 or 20 evals instead of 19. Requiring *exactly* 19 makes a timing perturbation (the very thing being measured) an integrity failure. The evaluator *calls* are unchanged (`Eval.evaluate` untouched), so tie the gate to "≤ one eval per epoch, no duplicate epoch" — not a fixed count that only held for one specific step trajectory.

### 3. The plan and proposal contradict each other on the step-count gate

**Proposal §Production: "at least 26,629 updates" (hard require) vs. Plan §Verification step 7: "Steps informational."**
One treats a low update count as an abort, the other as informational. This matters because node-to-node timing variance moves the realized step count; whether that aborts production must be decided before the run, not discovered mid-verdict. Reconcile explicitly. (Note the abort criteria also say "Do not abort for low intermediate accuracy," which further muddies whether exposure shortfall is fatal.)

### 4. Provenance instrumentation is pure overhead locked into the timed path and forbidden from removal

**Proposal §Exact placement / §Live worker / §Paired timing ("Do not remove them after benchmarking").**
The design mandates a per-batch all-three-channels-zero scan over `[128,3,32,32]` in the strong collator to count `erased_pixels`, and forbids stripping it. This is measurement machinery, not the intervention. It is worker-side work that competes with loader throughput and is charged into the ≤1% paired-timing gate. If this scan is what tips weighted overhead over 1.01, the experiment is scored `invalid` even though the actual hypothesis (mean-fill erasing) may be exposure-neutral. Recommend: keep provenance for the *preflight corpus* gates (where it's untimed and load-bearing), but do not require it inside the *production* timed loop — count erasures cheaply from the transform's own return, or drop production-time counting entirely so the timing gate measures the intervention, not the telemetry.

### 5. RNG-neutral per-image `fork_rng` is the dominant feasibility risk and is self-imposed for corpus purity, not required by the accuracy hypothesis

**Proposal §Exact placement: wrapper runs `RandomErasing` inside `torch.random.fork_rng(devices=[])` per image.**
Per-image save/restore of the ~5 KB Mersenne state runs ~50k×/epoch. This is precisely the EXP-029 failure class (learnings §Failed/Low: literal all-Conv GC was trajectory-safe but added 1.97% and missed the 1% gate). The overhead concentrates entirely in the strong phase — 80% of the 40/40/20 timing weight — so the weighted ≤1.01 gate effectively demands strong-path overhead ≲1.25%. Crucially, RNG neutrality is needed only to make the *paired corpus comparison bitwise-clean*; the production accuracy run does not need the other augmentations' RNG stream held fixed to be a valid training run. The plan elevates a purity nicety into the single most likely blocker. At minimum, flag it as the load-bearing gate; consider whether per-*batch* forking (or accepting stream shift in production only) removes the risk without harming the corpus gates.

### 6. Variable-length collate return is under-specified against the existing main loop

**Proposal §Exact placement (strong batch returns `erased_examples`, `erased_pixels`) vs. `train.py:218,221,223`.**
The main loop is `for inputs, targets in train_iterator` and branches on `targets.ndim`. A strong collator returning a 4-tuple while the weak collator returns a 2-tuple forces the loop to detect the arm and unpack conditionally. The plan says "weak retains hard two-field behavior" but never specifies how the loop distinguishes a 4-tuple from a 2-tuple, nor how `strong_batch_count`/`cutmix_batch_count` (which key off `targets.ndim`) survive the reshuffle. Get this wrong and you either crash or silently mis-assert the weak-tail (`assert targets.ndim == 1` at line 223). This is more `train.py` surface than "add one transform" implies and needs an explicit unpacking contract.

### 7. The acceptance threshold sits at the stated noise floor

**Proposal §Risks: "0.10 point is ten test examples, so a bare pass is protocol-valid but weak" — gate is `best_test_acc ≥ 94.25%` (= 94.15 + 0.10).**
The plan predicts an effect indistinguishable from single-seed noise and then gates on exactly that margin, with no reruns permitted (correctly, per seed-hacking constraint). This is not reward-hacking, but it means a "pass" carries essentially no causal evidence and a "miss" does not falsify the mechanism. The experiment is structured so its own success criterion is at the resolution limit. Worth surfacing so the analyze phase does not over-read a bare pass — the plan's own "diagnostics cannot override metric" rule cuts both ways here.

### 8. Preflight wall-time estimate is internally inconsistent

**Plan §Execution Environment ("preflights 20-35 minutes") vs. §Verification Procedure step budgets.**
Steps 2–6 alone budget 180 + 600 + 600 + 300 + 1500 = 3,180 s ≈ 53 min, before static/baseline/production. The "20–35 minutes" figure is not reconcilable with the listed gate budgets. This matters for the "one idle H20 / confirm idle before run" requirement — underestimating preflight occupancy risks colliding with the idle-GPU window the production run needs.

### 9. Minor: verification hardcodes the baseline instead of pulling it

**Plan §Verification step 1: "require 94.15 at `7c1e7d8`."**
The goal procedure (01-definition §Procedure step 2) says to obtain the moving baseline via `exp-index.sh baseline`. Hardcoding 94.15 is currently correct (TSV confirms) but bypasses the prescribed mechanism; if the moving baseline advances before this runs, the +0.1 gate silently uses a stale reference.

---

**Sound and not flagged:** the exact-zero mean-fill detection is watertight given `std=(1,1,1)` (true black normalizes to −0.49, only mean-fill hits exactly 0, and no 8-bit `k/255` equals the mean in all three channels since 0.4914·255 etc. are non-integer); placement success ~100% at this scale/ratio on 32×32 is correctly assumed; `fork_rng(devices=[])` correctly covers CPU-only draws matching the existing `cutmix_collate` pattern.
