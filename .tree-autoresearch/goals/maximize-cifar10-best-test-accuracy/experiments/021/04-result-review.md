All checks are complete. Here is the audit.

# EXP021 Independent Audit

**Scope of audit**: `run.log` (289 lines), `train.py` (726 lines), goal definition, brainstorm/plan/execute artifacts, tree TSV, git state.

## Verdict recomputation
- Parent: tree TSV row `004 … 95.40 improvement` confirms EXP004 = 95.40; branch `…-exp-021` sits on base commit `1a8d0de` (EXP-004). ✔
- Result: `best_test_acc: 95.11%` (run.log L279), equal to the max of all 130 eval lines (independently extracted; max = 95.11 at epoch 129). Δparent = −0.29, Δthreshold(95.50) = −0.39. The transcript's classification **failed / valid no-improvement** is correct — not a crash, not invalid. ✔

## Freshness & scope
- `run.log` mtime 2026-08-06 17:02:08 matches transcript window (16:54:51–17:02:17 UTC); `train.py` mtime 16:44:32 predates the run; the log's config line contains all `companion_*` keys, proving it was produced by the modified code, not a stale parent log. ✔
- `git diff --name-only 1a8d0de` → only `train.py`; `git status` → only ` M train.py`. Evaluation harness (`prepare.py` `Eval`) untouched. ✔

## Completion, summary, evaluation budget
- Exit summary complete (L267–288): charged 300.0 s ∈ [299.5, 301.0], total 433.8 s < 600 s outer limit, startup 1.1 s, peak VRAM 1190.6 MiB, 2,750,180 params. ✔
- Evaluations: 130 `eval ep` lines = `num_epochs` 130 = `eval_events` 130 → exactly once per epoch, within the goal's "no more than once per epoch" budget. ✔

## Companion isolation arithmetic (exact)
- primary_loss 25,336 = num_steps; replay_loss 2,402 = sam_applied; head_forwards 27,738 = 25,336 + 2,402 = expected; the code raises if any eval increments the head-forward counter, and `companion_integrity: status=PASS`. ✔
- Default forward (`train.py:158-175`) returns main logits only; companion pooling/head runs solely under `return_companion=True`, so the evaluator path is structurally clean. ✔

## CutMix/SAM dose
- CutMix 10,180/20,533 (ratio 0.4958 ≈ prob 0.5); eligible = 20,533 = SAM `first_step` 20,534 − 1, exactly the cutmix_end=0.75 / sam_start=0.75 boundary; SAM eligible = 25,336 − 20,533 = 4,803, applied 2,402 (ratio 0.5001, period 2), first_progress 0.7500; code raises on SAM/CutMix overlap. ✔

## Feature-bin/audit counts
- Expected samples 1 + ⌊(25,336−1)/512⌋ = 50 = observed; bins 13+14+14+9 = 50; vectors 3,328/3,584/3,584/2,304 = 256 × batches; the 9-sample last quartile is consistent with ~21 ms late SAM steps vs ~10 ms early steps. Audit `status=PASS`; report-only as planned. ✔

## Tail stats & composition criteria
- Tail-16 values match eval lines for epochs 115–130 exactly; recomputed mean 94.944375, min 94.66, max 95.11, final 95.10, premium 0.165625 — all match printed line L277. ✔
- Composition conjunction correctly reported failed: steps 25,336 ≥ 24,000 (pass), best 95.11 < 95.60 (fail), tail mean 94.944 < 95.50 (fail), best−final 0.01 ≤ 0.15 (pass). ✔

## Initialization / evaluator / reward-hacking risk
- Seed fixed at 42 as in the parent (no reroll); companion head built after `self.apply` under a saved/restored global CPU RNG, initialized from an isolated seed-42021 generator, so inherited weights and downstream RNG streams are parent-identical (`train.py:133-145`). Assertions pin 1,290 head / 2,750,180 total params. ✔
- Both SAM passes use the identical joint `main + 0.15·companion` objective; replay uses hard targets, correct since SAM steps are CutMix-ineligible (progress ≥ 0.75). BN tracking disabled and parameters exception-safely restored on the second pass. All companion work sits inside the charged `t0 → synchronize` window; evaluation stays outside charged time, matching the frozen budget semantics. No adaptive stopping on accuracy; best-of-evals selection is the inherited protocol. No evidence of metric gaming. ✔

## Transcript fidelity
Every quantitative claim in `03-execute.md` (line citations L267–L288, dose counts, bin means 5.593/4.347/4.133/5.988, displacement 11.435, nonfinite 0, timing, tail figures, −0.29/−0.39 deltas) matches the raw log exactly. The one preflight rerun (stale PID 80142 compute-app accounting) was disclosed and falls within the plan's preregistered pre-vector exception; the repaired preflight's gates all passed before the single metric launch.

No discrepancies found. The run is fresh, in-scope, complete, budget-compliant, and honestly reported; the no-improvement verdict (95.11 vs parent 95.40, threshold 95.50) is correctly derived.

AUDIT_VERDICT: PASS
