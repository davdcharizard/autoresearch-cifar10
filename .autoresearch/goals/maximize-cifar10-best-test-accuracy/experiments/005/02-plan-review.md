I have read the plan, brainstorm, idea-review, goal definition, TASK.md, `train.py`, `prepare.py`, and the EXP-004 analysis. Below is the prioritized concern list.

---

## EXP-005 Plan — Prioritized Concerns

### 1. Reward-hacking via augmentation RNG: the plan will record `improvement` on a single-seed pass that is statistically indistinguishable from a favorable draw (Verification Protocol steps 4 & 7; Milestone 3)
The acceptance bar is `92.40%` = **+0.10 pp = 10 test images out of 10,000**. The idea-review's concern #1 established two facts the plan does not resolve: (a) single-seed ResNet-20/CIFAR-10 run-to-run spread is well above 0.1 pp, and (b) EXP-004's own analysis states *"RandAugment necessarily changes the fixed-seed augmentation stream, so exact causal size cannot be separated from draw changes in one run."* Changing `AUG_SWITCH_FRACTION` moves where the strong→weak loader swap fires, which **re-shuffles the worker RNG stream from 75% onward** — a full augmentation-draw change over the last quarter of training. EXP-004 survived this only because it cleared the bar by **+0.37 pp** (real signal). This change is scored by the review as *uncertain sign, near noise* (impact 5/10). The plan's necessary condition (`best_test_acc >= 92.40%` → `improvement`) therefore has a high probability of certifying augmentation-draw noise as a training enhancement — functionally the outcome the no-seed-hacking constraint exists to prevent, entering through the augmentation stream instead of the seed knob. The plan's only mitigation is a soft "report as weak causal evidence" caveat (step 7), which does not change the recorded verdict.

### 2. The plan foreclosed the one legitimate mitigation, mischaracterizing it as seed hacking (brainstorm `## Review`; Abort Criteria)
The idea-review explicitly clarified: *"using extra seeds to estimate variance, not to cherry-pick the max, is legitimate due diligence, not seed hacking."* The brainstorm rejects this ("Did not adopt its proposed multi-seed variance runs because the user-defined goal forbids rerolling seeds"), conflating variance estimation with metric-chasing. The constraint forbids *rerolling to obtain a favorable metric*, not measuring the noise floor. By declining a variance estimate, the plan removes the only thing that would make a near-threshold result interpretable, and then proceeds to a hard 92.40 cutoff anyway. This is the core methodological gap, not an incidental one.

### 3. The "high-LR weak-phase evidence" verification check is tautological — it proves nothing about the hypothesis (Milestone 3; Verification step 6)
The plan's strongest-sounding necessary condition ("at least one post-switch/pre-80% step logging `lr: 0.1000`") is **true by construction**: the LR math is keyed to `LR_HOLD_FRACTION=0.8` and the switch fires at 0.75, so `lr=0.1` in 75–80% is guaranteed the moment both boundary constants are set correctly. It confirms the *mechanism was wired*, not that it *helped*. Combined with the fact that **no evaluation runs in the 75–80% window** (see #4), the protocol cannot produce any evidence that the weak high-LR phase did anything — the experiment's entire explanatory claim rests on comparing 0.8+ evals against EXP-004's 0.8+ evals across a re-shuffled RNG stream, which #1 already says is unattributable.

### 4. No eval fires at or during the new 75–80% phase, reducing observability of the very phase being tested (`train.py:260-266`, `train.py:281`)
`dense_tail_due = progress >= LR_HOLD_FRACTION` stays at 0.8, and `EVAL_CHECKPOINTS` ends at 0.7. In EXP-004 the switch coincided with the 0.8 dense-tail eval, so a checkpoint captured the model at the boundary. With the switch moved to 0.75, the epoch that breaks at 0.75 has `checkpoint_due=False` (0.7 already consumed) and `dense_tail_due=False`, so **no eval runs at the switch, and none runs until 0.8**. The plan's Milestone 2 claim "training resumes, and logged steps from 75-80% remain at lr=0.1" is verifiable, but there is zero accuracy visibility into the phase whose contribution is the whole point. If the high-LR clean-adaptation produced a transient peak inside 0.75–0.80, `best_acc` would never see it.

### 5. Boundary-semantics fragility: correctness depends on editing *exactly two* differently-typed predicates while leaving two others untouched (Milestone 1; `train.py:207, 252, 264, 281`)
`LR_HOLD_FRACTION` appears in four places with two different meanings:
- `train.py:252` — augmentation break, used as a **time** threshold (`total_training_time >= LR_HOLD_FRACTION * TIME_BUDGET_S`) → must change to `AUG_SWITCH_FRACTION`.
- `train.py:281` — loader switch, used as a **progress** threshold (`progress >= LR_HOLD_FRACTION`) → must change.
- `train.py:207` — LR schedule → must **stay** 0.8.
- `train.py:264` — `dense_tail_due` eval cadence → must **stay** 0.8.

If only one of {252, 281} is changed, the run does not crash but silently degrades: e.g., changing 252 alone (break at 0.75, switch predicate still 0.8) yields epochs that break after ~1 batch and never switch until 0.8, collapsing to a degenerate loop; changing 281 alone reproduces EXP-004 exactly. The plan's guard ("exactly one switch at 75-76%") catches the second case but is worth stating explicitly that the diff must touch precisely `:252` and `:281` and that the two `0.8`-keyed sites are intentionally preserved. The plan text ("change only the two augmentation break/switch conditions") is correct but does not name the line-level distinction, which is where an implementer is most likely to slip.

### 6. Self-imposed necessary conditions can spuriously fail a valid run (Milestone 3; Verification step 5)
The plan adds `300.0 <= training_seconds < 310.0` and `total_seconds < 600` as *necessary* conditions beyond the goal's actual requirements (goal only requires respecting the budget and finishing <10 min). Counted `training_seconds` stops at ~`TIME_BUDGET_S=300`, so the lower bound `>= 300.0` is fine, but pinning a hard `< 310.0` band as a pass/fail gate risks marking a legitimate run `invalid` on a benign timing artifact rather than a real failure. These should be informational, not gating.

---

Concerns #1–#3 are the fundamental ones: the experiment as planned cannot separate a genuine effect from noise, declined the mitigation that would let it, and its verification is structured so that the mechanism check passes regardless of whether the mechanism works. #4–#6 are execution/boundary specifics to harden if the round proceeds.
