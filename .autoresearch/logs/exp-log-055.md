# Experiment Log EXP-055: FreezeOut-style tail freezing of stem+stage1 (FREEZE_FRAC=0.70, dual-graph warmup)

## Execution

- **Created**: 2026-06-11
- **Brainstorm**: brainstorm/brainstorm-055.md
- **Plan**: plans/plan-055.md
- **Autonomy**: autopilot
- **Experiment Branch**: autoresearch/exp-055
- **Commit**: (pending)
- **PR**: (pending)
- **Outcome**: completed

## Implementation Notes

### Summary
Applied the planned 4-part change on branch autoresearch/exp-055 (train.py only): (1) `FREEZE_FRAC = 0.70` constant; (2) optimizer rebuilt as 4 param groups — B-decay, B-nodecay, A-decay, A-nodecay (group A = `base_model.conv1/bn1/layer1` collected by `id()`, B = complement; decay split `ndim > 1` unchanged; B first so the loop's `param_groups[0]["lr"]` print tracks the still-training group; `tag` extra keys preserved by torch.optim); (3) compile warmup extended — after the existing 3 unfrozen iters, A is flipped `requires_grad_(False)`, 2 more autocast forward+backward iters on the same random tensors compile+cache the frozen graph variant (AOTAutograd specializes on param requires_grad), then A is restored and grads zeroed — no optimizer.step anywhere, weights provably unchanged (established EXP-006 uncharged-warmup pattern, extended exactly as pre-registered); (4) timed loop — one-shot freeze flip at `progress >= FREEZE_FRAC` with a single FREEZE marker print (charged, ~µs), per-group LR: `lr_a = lr_at(min(progress / FREEZE_FRAC, 1.0))`, `lr_b = lr_at(progress)`. AST OK. CPU sanity `/tmp/exp055_sanity.py` ALL PASS: (a) params exactly 4,286,026; (b) partition disjoint+complete, |A| = 21 tensors / 223,808 elems, 4-group element sum = total; (c) lr_A(0)=0, lr_A(0.105)=PEAK (compressed warmup end), lr_A(0.70)=0 (anneal complete), pinned at lr_at(1.0) after; (d) frozen A grads None while B grads flow AND bn1 running_mean keeps tracking in train mode; (e) tags preserved, frozen A byte-identical through optimizer.step (grads-None skip ⇒ no WD/momentum on A — true freeze), B updates; (f) after restore A grads flow again, 3-step smoke decreasing (12.47 → 9.55). M1 complete.

### Surprises & Discoveries
- **Group A is ~⅓ of conv FLOPs but only 5.2% of params** (223,808 of 4,286,026): stage widths are FLOPs-balanced (64²·32² = 128²·16² = 256²·8²) while params scale with width² only — so the freeze removes a third of backward COMPUTE while leaving 94.8% of parameters training. Strengthens the mechanism reading: the dt drop (if seen) is pure FLOPs reallocation, and the "capacity stops moving" cost is small in parameter terms.
- Sanity smoke initially failed at lr 0.05 (nesterov momentum overshoot on a single random batch — loss 12.47 → 14.57); test artifact, not a code bug; re-ran at lr 0.01 × 3 steps, decreasing.

### Decisions
- Honest heat accounting recorded in plan: the compressed schedule is the UNSCALED FreezeOut variant — group A's integrated heat is 0.70× its baseline allocation. The amplitude-scaled variant (A peak ≈ 0.57) would preserve the integral but violates the certified-peak heat law (EXP-010); documented as the unexplored alternative if branch (iii) fires.
- Pre-registered branches per plan-055/brainstorm-055: (i) read ≥ 96.81 → byte-identical replicate, improvement iff MEAN ≥ 96.81 (EXP-052 protocol; max never a decision input; reads in (96.73, 96.81) are no-improvement, no single-draw promotion); (ii) [96.41, 96.73] at family signatures WITH ledger step surplus → freeze free but tail steps sub-σ, class closed; (iii) < 96.41 → layer-1 tail refinement load-bearing, parameter-side tail-pressure law; (iv) mid-run recompile signature (one-off ≥15s watchdog stall at the flip / collapsed ledger at clean dt) → ONE fix attempt (strengthen dual-warmup cache), else failed; (v) gate/contention/startup kills → infra relaunch (max 2).

## Run Log

### Run 2

**Description**: Relaunch with the graph-visible detach-flag freeze (the single permitted fix for Run 1's absent dt-drop) after a full validation chain: CPU sanity v2 ALL PASS (detach semantics: A grads None, B grads flow, bn1 stats track, eval forward flag-invariant) and GPU probe PASS (/tmp/exp055_gpu_probe.py: unfrozen 22.04ms, frozen 15.15ms = **31.3% saving**, first post-flip step 0.016s = no recompile, A/B grad split correct under the compiled frozen graph). Probe-revised expectations (recorded in plan before launch): post-freeze windows ~14.0–17.5ms, num_steps ~15,000–15,500 (~+12% surplus vs family), epochs ~148–162, total_seconds tighter vs cap (~157 evals) with WALL_CAP backstop.

**Metadata**:
- Job ID: background task b1uxhseoj (composite; train pid 2003915)
- Log file: run.log
- WandB: N/A
- Status: completed (rc=0)
- Started: 2026-06-11 05:33:50
- Ended: 2026-06-11 ~05:43:30

**Observations**: PRISTINE and the mechanism FULLY engaged. GATES_CLEAR poll 1 (apps=0, load=11). D0 = 22.5ms; all pre-freeze windows 22.0–22.7ms, slow_streak 0 throughout. FREEZE marker at step 9355, progress 0.700 (run.log). The dt drop is textbook: tick 24 (pct 71.0) transition window 19.3ms, then 15.3–16.5ms for the entire tail — inside the probe band [14.0, 17.5]. **num_steps 15,026** (+~1,550 over family 13,400–13,500, the predicted ~12% surplus), **155 epochs** (+~16 tail epochs / +16 plateau evals), params 4,286,026, 300.0s charged, total 551.4s ≤ 600 (the predicted eval+stall growth landed at +~50s vs family), startup 9.9s (FX cache), evals 155 ≤ 155, ep1 35.79 in family band, no NaN, converged-flat plateau (last 8: 96.16–96.29), final_test_loss 0.1921 (family-adjacent). **best_test_acc 96.32 < 96.41 → pre-registered branch (iii)**: with the conversion mechanism instrumented and fully delivered, accuracy went DOWN ~−1.6σ vs recipe mean — freezing stem+layer1 at p=0.70 (with its 0.70×-heat compressed anneal) costs more than +1,550 tail steps for layers 2–3 repay. Parameter-side tail-pressure law established alongside the data-side one (EXP-025/033). No escalation (read < 96.81).

### Run 1

**Description**: Single gated run of the freeze variant — the first compute-reallocation (layers×time) experiment in the project; every catalogued axis is measured-closed after 48 nulls. Launched via the validated composite `/tmp/exp046_composite.sh` (dual gates GPU-0-free + load<60; D0 gate >26ms; contention streak; NaN/divergence guards; wall cap). Expected: D0 22.3–22.8ms pre-freeze; FREEZE marker at progress ≈ 0.700 (~step 9,300); post-freeze watchdog windows dropping to ~17.5–18.5ms (the direct mechanism check); num_steps ~14,200–14,500 (family 13,400–13,500); epochs ≥ 145; params 4,286,026; startup ~45–55s (dual compile). Escalation to a replicate pair only on a bar-clearing read.

**Metadata**:
- Job ID: background task b7ndqrwvc (composite; train pid 1927405)
- Log file: run.log
- WandB: N/A
- Status: running
- Started: 2026-06-11 05:18:54

**Observations**: GATES_CLEAR at poll 1 (apps=0, load=14). GATE_DECISION D0 = 22.5ms (gate windows 22.0/22.5/23.2). Run completed rc=0 (best 96.26, 133 ep, 12,893 steps, 495.1s, startup 12.5s) but is REJECTED on two grounds: (1) **mechanism no-op** — FREEZE marker fired at step 8869/progress 0.700 (run.log L365) yet every post-freeze watchdog window stayed 22.0–22.7ms (expected ~17.5–18.5): the compiled graph silently ignored the mid-run `requires_grad_(False)` flip — no recompile stall at the flip, no second compile during warmup (startup 12.5s < baseline ~23s, pure FX-cache hit), no backward saving; group A only stopped moving because lr_a = 0 after p = 0.70. The hypothesis's conversion mechanism (step surplus) never engaged — ledger 12,893 is BELOW family. (2) **contention episode** — ticks 6–8 windows 33.0/46.0/48.0ms (slow_streak peaked at 3 of 4), a ~45–90s external stall that alone explains the ~500-step deficit. Per plan abort criteria: absent dt-drop = implementation failure → ONE fix attempt; contention = relaunch-eligible. The 96.26 read is not a valid hypothesis read (do not interpret).

## Experimental Adjustments

- **Run 1 → Run 2: freeze made graph-visible (detach-at-boundary) instead of param requires_grad flips.** Evidence: Run 1 watchdog windows flat 22.0–22.7ms after the FREEZE marker (composite task b7ndqrwvc, ticks 25–33) + startup 12.5s (no second variant ever compiled). Mechanism: this torch.compile path does not re-guard on mid-run `requires_grad` mutation of module params — the flip is a silent no-op for the cached graph (it neither recompiles nor drops A's backward). Fix: `self.freeze_stage1` bool on ResNet, `out = out.detach()` between layer1 and layer2 when set; dynamo guards module bools, so the warmup pre-compiles BOTH flag variants and the p=0.70 flip dispatches to the cached frozen graph. Freeze semantics identical (A grads None → SGD skips → no WD/momentum; BN stats keep tracking; forward numerics unchanged). This is the plan's single permitted fix attempt for the absent-dt-drop failure.

## Errors & Dead Ends

### 2026-06-11 — Run 1: requires_grad freeze was a silent no-op under torch.compile; plus mid-run contention
- Error: `no exception — FREEZE marker printed at step 8869 (run.log L365) but post-freeze windows stayed 22.0–22.7ms (no backward saving); num_steps 12,893 < family 13,400–13,500; ticks 6–8 windows 33/46/48ms (external contention, slow_streak 3)`
- Root cause: torch.compile's cached graph does not guard on `requires_grad` of module parameters in this setup — mid-run `p.requires_grad_(False)` neither triggers a recompile nor removes A's backward subgraph; the dual-variant warmup therefore also compiled nothing new (startup 12.5s, pure cache hit). Contention was an unrelated external load episode that recovered.
- Source: composite task b7ndqrwvc output (ticks 6–8, 25–33); run.log L365 + step prints 08900–09250 (dt 22–23ms post-freeze)
- Do NOT retry: never implement train-time graph-topology changes via mid-run requires_grad mutation under torch.compile — make the change graph-visible (module flag + detach, traced control flow) and pre-warm both variants.

## Verification Results

### Conditions Checked

**Integrity pre-condition (PASS, Run 2)**: GATES_CLEAR poll 1; D0 22.5 ∈ [21.5, 23.5]; pre-freeze windows 22.0–22.7ms (≤ 23.5 mean, none > 27) ✓; no kill markers, rc=0 ✓; FREEZE marker at progress 0.700 ✓; post-freeze windows 15.3–16.5ms ∈ [14.0, 17.5] (mechanism engaged) ✓; num_steps 15,026 ∈ [15,000, 15,500] ✓; num_epochs 155 ∈ [148, 162] ✓; params 4,286,026 exact ✓; training_seconds 300.0 ✓; total_seconds 551.4 ≤ 600 ✓; evals 155 ≤ 155 ✓; trajectory criterion: ep1 35.79 (> 30 tripwire), family-shaped climb, converged-flat plateau (last 8 evals 96.16–96.32), final_test_loss 0.1921 family-adjacent ✓; no NaN/EMA spikes ✓. Source: run.log; composite task b1uxhseoj.

**Condition 1 — best_test_acc ≥ 96.81** (baseline 96.71 + 0.1; re-queried via exp-index.sh baseline): **96.32 → FAIL** (first-failure-stop). No escalation (replicate-pair branch required ≥ 96.81). **Pre-registered branch (iii)**: 96.32 < 96.41 → layer-1/stem tail refinement is load-bearing — the freeze-and-convert construction loses ~−1.6σ even with the step surplus fully delivered and instrumented. Class closed with sign.

**Condition 2 — within budget**: rc=0, total_seconds 551.4 ≤ 600 → PASS (informational, after first failure).

**Condition 3 — eval cadence**: 155 evals ≤ 155 epochs → PASS (informational).

**Verdict basis**: no-improvement (valid pristine run with the mechanism verified engaged; condition 1 failed; branch (iii) closure). Run 1 was rejected for integrity (mechanism no-op + contention) and does not enter the decision.

## Human Notes

(autopilot — none)
