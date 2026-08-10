I've read all four documents plus the actual `train.py`. Here are the prioritized concerns, hardest/most-fundamental first.

---

## Prioritized plan concerns — EXP-027

### 1. The acceptance design cannot distinguish the predicted effect from seed noise, yet a pass is committed as the new frontier (reward-hacking / benchmark-overfit)
**Where:** Proposal hypothesis (point prediction `94.32`, accept at `94.25`); Plan Verification step 7; goal §Necessary Conditions.

Baseline is `94.15`, the threshold is `94.25` — exactly the goal's `+0.10pp` minimum — and the *predicted* effect is `+0.17pp`. Both numbers sit at or below normal CIFAR-10 ResNet-20 run-to-run variability. The plan runs **one** seed-42 run and the goal forbids seed rerolls, so there is no variance estimate and no way to separate a real `+0.1pp` curriculum effect from a lucky fluctuation. Per `.autoresearch/CLAUDE.md`, an `improvement` verdict commits and merges `train.py` to the integration branch and moves the moving baseline. The net result is a mechanism that can ratchet the frontier upward on noise across successive experiments. The proposal's own text ("a bare 0.10-point gain … must be described as weak single-seed causal evidence") concedes this, but the plan still treats `>=94.25` as an accept-and-merge. This is the most fundamental issue: even a flawlessly executed run may not represent a genuine training enhancement, which is exactly what the goal statement demands ("genuine training enhancements … rather than seed selection").

### 2. The central mechanism — a forkserver-shared `mp.Value(..., lock=True)` passed through the pickled `collate_fn` — is likely to fail forkserver worker startup
**Where:** Proposal "Exact implementation" lines 82–85 (`forkserver_context.Value("b", True, lock=True)`, `PhaseCutMixCollator(cutmix_enabled)`, passed to `make_train_loader(collate_fn=...)`); Plan Code Changes bullet 1; existing loader uses `multiprocessing_context="forkserver"` (`train.py:133`).

Synchronized objects (`Value` with a lock) are designed to be shared **by inheritance (fork)**, not by pickling. With `multiprocessing_context="forkserver"`, DataLoader workers receive the `collate_fn` — and therefore the embedded `Value` — via pickle over a pipe, which typically raises *"Synchronized objects should only be shared between processes through inheritance."* The entire idea depends on this object being visible and live in eight forkserver workers. The plan's only contingency is the Milestone-2 gate "survive actual forkserver pickling" → **abort on failure**, and it explicitly forbids the standard cross-forkserver-shared-state remedy (`manager` process, plan Abort Criteria line 52 / proposal line 135). So the single most probable outcome is an `invalid`/aborted experiment on a known IPC pitfall, with no in-scope fallback. This risk deserves an early, isolated feasibility probe *before* the 8–15 min preflight and 5–7 min timing machinery are built.

### 3. The production `train.py` change is over-coupled to EXP026's gitignored local corpus via a hardcoded SHA-256
**Where:** Plan Milestone 2 line 14 / proposal "Immutable-source comparison"; hash `4386e6915d0b…`.

The corpus is only an input to a *diagnostic* integrity gate — the tracked `train.py` change does not need it. Yet the plan makes a missing-or-mismatched artifact a hard abort ("otherwise abort rather than rematerialize"). Experiment-local diagnostics from EXP026 are gitignored and the goal instructs removing completed-experiment files between runs, so this artifact may simply not exist anymore, blocking a scientifically-fine experiment on a bookkeeping dependency. The gate couples EXP027's runnability to EXP026's ephemeral local state.

### 4. Strict, noise-sensitive proxy gates convert otherwise-valid runs into aborts
**Where:** Plan Milestone 3 lines 19–20 / proposal "Timing and exposure gates" lines 148–157; continuation gate loss-EMA `<=1.5` (proposal line 123).

Aggregate candidate/control counted-step ratio `<=1.01`, every pair `<=1.04`, per-arm CV `<3%`, projected total `<540s` — all extrapolated from a **1,000-step** proxy (800 strong + 200 weak) with 100 warmups. A 1000-step proxy will not capture thermal/throttling behavior over a real 300s run, and a `<=1.01` aggregate ratio is inside typical H20 timing jitter. Because these are abort gates, the failure mode is one-directional: they cannot produce a false accept, but they can readily kill a production run that would itself have passed its own in-run timing checks. The plan should justify why a `1.01` aggregate ratio is achievable given measured jitter, or loosen the proxy gate, otherwise the experiment is fragile to hardware noise unrelated to correctness.

### 5. The 24-batch drain gate is validated under continuous iteration, but production recreates the iterator every epoch
**Where:** Proposal "Real-loader transition and cleanup gate" (20,000-collation continuous test, drain bound `2*NUM_WORKERS+8`); production loop recreates `train_iterator = iter(train_loader)` each epoch (`train.py:217`, reset at `:279`) with `persistent_workers=True` and `drop_last=True`.

The 70% CutMix-off request fires mid-epoch. At the next epoch boundary the loop nulls and re-creates the iterator on a persistent-worker loader, which resets prefetch and discards the tail of the current epoch's prefetched batches. The lifecycle proxy exercises one continuous stream and asserts "propagation within 24 delivered batches," but production's drain semantics straddle an epoch boundary and may differ. This risks either (a) a spurious "policy-on after drain / beyond 70.5%" abort in production, or (b) passing a lifecycle gate whose conditions don't actually match the production path it is meant to certify. The gate should reproduce the epoch-boundary iterator recreation, not just continuous iteration.

### 6. The "accept" interpretation overclaims relative to what the intervention can isolate
**Where:** Proposal interpretation table, accept row ("Temporal CutMix removal is a valid fixed-budget improvement"); Attribution risk section.

The proposal elsewhere correctly notes the intervention jointly removes pasted pixels *and* soft targets and "cannot distinguish those two CutMix components," but the accept-row conclusion is stated as a clean causal claim about "temporal CutMix removal." A `+0.10pp` single-seed result establishes neither that the effect is causal (see #1) nor which of the two coupled changes produced it. The verdict language should be constrained to the net curriculum, not attributed to a mechanism the design cannot separate.
