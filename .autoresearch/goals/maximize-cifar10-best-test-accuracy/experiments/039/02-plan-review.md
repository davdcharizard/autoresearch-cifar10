I've read the plan, goal, brainstorm, and the actual `train.py`. The change is 3 lines (lines 103–104 replaced by the avg/rms lerp). The core math checks out — layer3 output is post-ReLU (line 67), so maps are genuinely nonnegative, and the `[1, 71/64]` descriptor / `[63/64, 71/64]` Jacobian bounds and `rms = ||x||/8` are all correct. I will not raise those. Here are the concerns that survive contact with the concrete plan and code, hardest first.

---

## Prioritized concerns

### 1. `vector_norm` NaN gradient at zero, with "No epsilon" hard-coded and no rescue allowed — likely kills production (Code Changes, line 22; Abort, line 43)
The code is `rms = torch.linalg.vector_norm(out, dim=(2,3)) / 8.0` and the plan explicitly forbids a guard: *"No epsilon, parameter, gate..."*. The backward of the L2 norm at a zero vector is `x/‖x‖ = 0/0`. Post-ReLU it is entirely possible (especially early in training / for individual examples) for a channel's whole 8×8 map to be exactly zero — a dead-channel-for-this-input. When that happens, forward is fine (rms=0) but `loss.backward()` produces **NaN gradients**, `optimizer.step()` corrupts weights, and the run diverges. Production only stops for "nonfinite" with **"No coefficient/GeM/gate/phase rescue or reroll"** (line 43), so a single such event wastes the whole experiment with no in-scope fix. The oracle's "zero subgradient" check (line 51) shows awareness, but its verdict is torch-version-dependent, and a clean single-tensor oracle does not guarantee no all-zero channel arises across hundreds of thousands of production examples. This is the most fundamental risk and needs an explicit resolution (confirm the installed torch returns a finite gradient at 0, or the "no epsilon" rule is self-defeating).

### 2. The "≤19 evaluations" gate is unimplemented, unjustified, and entangles the metric with speed (Verification lines 53, 55; brainstorm line 19)
The plan promises no evaluator change, yet lists *"require ... ≤19 unique once-per-epoch evaluations"* as a hard gate and elsewhere speaks of "cap evaluations at 19." No cap exists — the code evaluates on 4 fixed checkpoints **plus every epoch in the dense tail** (lines 283–304). So the evaluation count is throughput-dependent, not fixed at 19, and `best_test_acc` is a max over that variable set. Consequences:
- If the added norm slows throughput even slightly, the candidate gets **fewer** tail epochs → fewer chances to catch a peak than the baseline's 19 — a comparison biased against the candidate that the plan never equalizes.
- The "≤19" phrasing is a baseline-specific number carried over from the *channels-last* idea's speed concern (brainstorm line 19); it is asserted as a gate here without establishing that an RMS run naturally yields exactly that count, so it can false-veto.

The plan needs to either implement a genuine fixed eval schedule or drop the "cap at 19" framing and acknowledge the eval-count/throughput coupling.

### 3. Milestones 1–2 are unfalsifiable safety theater; the actual hypothesis gets one noisy shot (Milestones 1–2; brainstorm lines 68, 74, 89)
The algebraic bounds are mathematical identities on maps that are *guaranteed* nonnegative, and the activity diagnostic ("nonzero and distributed descriptor change") cannot fail unless an 8×8 map is perfectly spatially constant — which never occurs. Two full milestones of FP64/VJP oracles, replay, and activity checks therefore de-risk essentially nothing about the objective. Meanwhile the one thing that matters — does a `1/64` RMS blend raise accuracy — rests on a single seed-42, best-of-~19 run, and the brainstorm itself concedes *"1/64 may be too weak"* / *"safe coefficient may be sub-threshold"* (lines 68, 74). The expected outcome is an uninformative near-baseline miss dressed in heavy verification. Worth flagging that the effort is misallocated relative to what it can conclude.

### 4. Timing gates are over-strict for a negligible-cost op → false abort from jitter (Abort line 42)
The added work is one `vector_norm` over 64 elements per channel — microscopic. Yet timing aborts if *"aggregate ... mean >1.05, any pair >1.08, CV≥3%."* GPU wall-time CV over 1000-step blocks routinely exceeds 3%, and a single noisy pair >1.08 aborts everything. This will veto on measurement noise rather than on any real regression from the change.

### 5. EXP022/028 replay controls: stale-corpus dependency and many ratio gates → complexity/false-abort; scope of "ignored controllers" (Abort line 40; Verification line 52)
The plan hardcodes 64-hex SHAs for EXP022/028 corpora and applies numerous ratio gates ("update >5× preceding-16-step median," "logit ratio >5×", "phase loss EMA >1.5×") to replayed batches. These can trip on legitimate early-training dynamics or a missing/renamed corpus, aborting a sound change before it ever runs. Separately, the "ignored oracle/activity/replay/timing controllers" (Code Changes line 23) must be confirmed to only *read* `train.py` and never edit any tracked file, since the Hard Constraint restricts edits to `train.py` alone (the protected-files hook would block otherwise, but the plan should state it).

---

Not concerns (checked, sound): the lerp arithmetic (`63/64·avg + 1/64·rms`), the `[1,71/64]`/`[63/64,71/64]` bounds, nonnegativity of the pooled map, unchanged param count (1,073,962), the 94.25 threshold (94.15 + 0.1), and the no-reroll/no-seed-hack/no-evaluator-edit discipline.
