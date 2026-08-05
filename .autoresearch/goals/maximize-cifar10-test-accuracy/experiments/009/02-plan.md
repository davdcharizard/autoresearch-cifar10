# Plan EXP-009: Muon optimizer (Newton-Schulz orthogonalized momentum) on the conv weights
- **Created**: 2026-06-28

## Summary & Key Design Decision (read first)

Swap the **conv weights only** from SGD-Nesterov to a from-scratch **Muon** optimizer (orthogonalized momentum via a 3-step Newton-Schulz quintic), keeping every non-conv learnable param (fc weight, BN γ/β, ReZero α) on the **unchanged EXP-008 SGD-Nesterov** (lr 0.4, mom 0.9, wd 5e-4). All architecture / augmentation / EMA / whitening / TTA / schedule-shape / batch / seeds stay byte-identical to EXP-008.

**This plan deviates from proposal idea-03 in three deliberate, better-grounded ways**, all of which were forced by the cross-model review and by reading the canonical reference `airbench94_muon.py` (Keller Jordan; fetched 2026-06-28):

1. **Resolves review concern #2 (coupled-WD-through-orthogonalization distorts the L2 penalty) by removing weight decay from the Muon group entirely** and instead using airbench's **weight re-normalization** (`p ← p · √out / ‖p‖` each step). Rationale: every conv is followed by BatchNorm, which is scale-invariant to the conv weight magnitude, so pinning ‖p‖ is a *safe* norm control that fully replaces L2 on those tensors (this is exactly idea-01's "BN-followed weights don't need decay" argument, taken to its conclusion). This is airbench's actual proven choice on this exact net family — strictly cleaner than the reviewer's suggested decoupled-WD patch, because it removes WD from the orthogonalized path rather than re-deriving its magnitude.
2. **Grounds the load-bearing Muon LR in airbench's proven value 0.24** (review concern #1: "LR is the load-bearing unverified assumption"), not the proposal's guessed 0.02. The 12× gap is purely a scaling-convention difference: the proposal used an update-scale convention where ‖p‖ drifts; airbench's weight-renorm pins ‖p‖=√out, which makes the LR meaning stable. Because Newton-Schulz normalizes its input (`X/‖X‖`) and drives singular values→1, the update magnitude is independent of momentum, so the *momentum* axis (airbench 0.6 → our 0.9) transfers cleanly. **Honesty (plan-review #7):** 0.24 is the *best-grounded starting value* (proven on the same whitened wide-shallow ResNet family) — far better than a blind guess — but airbench differs in schedule length (8 vs ~150 epochs), batch size, and has no EMA, so the transfer across our longer schedule + EMA tail is NOT guaranteed. The first run is therefore also a *directional calibration* of (PEAK_LR_MUON), and its result stands as the experiment's verdict either way (see below).
3. **Sends only 4D conv weights to Muon (`p.ndim==4`), keeping the 2D `fc.weight` on SGD** (matches airbench's `len(shape)==4` filter). Rationale: the fc classifier is *not* BN-followed, so weight-renorm would distort the logit scale — airbench deliberately routes the head to SGD. fc is 5,120 params; routing it to SGD is the safe choice and keeps the classifier's exact EXP-008 dynamics.

Net effect vs EXP-008: the **conv-weight training method** changes as a *package* — orthogonalized Muon steps + airbench weight-renorm (norm control) replacing raw SGD steps + L2 5e-4. **Attribution honesty (plan-review #5):** this is NOT an isolated "SGD→Muon update" swap; it also removes conv L2 and adds per-step norm projection. A win is therefore attributable to *the airbench Muon package as a coherent unit*, not to the orthogonalization alone — isolating the components (Muon+decoupled-WD vs Muon+renorm) is a deliberate follow-up ablation, not this run. The **core brainstorm hypothesis is intact** (plan-review #6): "Muon orthogonalized optimization on the conv weights beats tuned SGD-Nesterov on this net within budget, clearing 96.48" — only the implementation specifics (WD handling, scaling convention, LR, NS steps, fc routing) were refined toward the better-grounded airbench reference in response to the idea-review's WD-correctness and LR-guess concerns.

**Verdict discipline (plan-review #1 — no "discarded calibration"):** this run's `best_test_acc` IS the experiment's outcome and will be recorded honestly in the TSV (improvement / no-improvement) regardless of result. The trajectory diagnostics inform a *possible future experiment* at a retuned LR; they do NOT excuse, discard, or re-roll this run. No seed is re-rolled and no within-experiment sweep is performed — a single fixed-seed run, one verdict.

## Milestones

### Milestone 1: Muon implemented + smoke-tested (no full run yet)
- [ ] Add `zeropower_via_newtonschulz5(G, steps=3, eps=1e-7)` and `class Muon(torch.optim.Optimizer)` near the top of `train.py` (torch-only, no new import).
- [ ] Add hyperparameters `PEAK_LR_MUON = 0.24`, `MUON_NS_STEPS = 3` (and keep `PEAK_LR = 0.4` for the SGD group; remove no other constant).
- [ ] Replace the single-optimizer construction (`train.py:244-250`) with the two-group split (4D→Muon, else→SGD) and replace the LR-set / zero_grad / step sites in the loop to drive both optimizers off the shared triangular `frac`.
- [ ] **Smoke test (must pass before the official run)** — run the snippet in the Verification section: assert `zeropower_via_newtonschulz5` on random `[512,4608]` and tall `[4608,512]` returns all-finite output with **no blow-up** (max singular value < ~2); assert one `Muon.step()` runs, leaves params finite, and the weight-renorm gives `‖p‖≈√out`. Confirm `python -c "import ast; ast.parse(open('train.py').read())"` parses. (Note: a *random Gaussian* input won't have all singular values pulled to ~1 in 3 NS steps — that's expected; real structured gradients converge better. The gate is finiteness + no blow-up + correct renorm, not a tight sv band.)
- [ ] Confirm `ls *.py` = exactly `prepare.py  train.py` (no stray modules) and `git diff --quiet -- prepare.py` (frozen eval untouched).

### Milestone 2: Official run launched and confirmed healthy
- [ ] Confirm the Milestone-1 edits are actually in `train.py` (`grep -n "class Muon" train.py` non-empty) BEFORE launch.
- [ ] Launch `CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1` under `timeout 600`, backgrounded with a sentinel file.
- [ ] Within the first ~60 steps, confirm `run.log` shows a finite, *decreasing* `loss:` (no NaN/inf) and `num_params:` is not yet printed (still training). Record img/s and early loss.

### Milestone 3: Run completed, metrics extracted, verdict diagnostics read
- [ ] Run reaches the `---` summary block; extract `best_test_acc`, `num_epochs`, `total_seconds`, `peak_vram_mb`, `num_params`.
- [ ] Read the pre-registered trajectory diagnostics (ep10/ep25 acc, tail shape) to classify the outcome (working / LR-too-low / LR-too-high) even if it loses.

## Code Changes

All edits in `train.py` only.

- **`train.py` (new top-level code, after the imports / near the hyperparameters block ~line 35):** add the Newton-Schulz function and the `Muon` optimizer class.
  - `zeropower_via_newtonschulz5(G, steps=3, eps=1e-7)`: assert 2D; `a,b,c=(3.4445,-4.7750,2.0315)`; `X=G.bfloat16(); X=X/(X.norm()+eps)`; if `G.size(0)>G.size(1)` transpose; 3 iters of `A=X@X.T; B=b*A+c*(A@A); X=a*X+B@X`; transpose back; return `X`.
  - `Muon(params, lr, momentum, nesterov=True, ns_steps=3)` with a `@torch.no_grad() step()`: per param with a grad — nesterov-momentum buffer (`buf.mul_(mom).add_(g); g=g.add(buf,alpha=mom)`), reshape to `[out, -1]`, `update = zeropower_via_newtonschulz5(gmat, ns).view_as(p)`, **weight-renorm** `p.data.mul_(p.size(0)**0.5/(p.data.norm()+1e-7))`, then `p.data.add_(update.to(p.dtype), alpha=-lr)`. **No weight decay term** (deliberate — see Summary decision #1).
  - *Why:* tests the hypothesis that orthogonalized conv-weight updates reach a better-conditioned minimum + complete the EXP-008 under-annealed tail, at near-zero throughput cost.
  - *Risk/edge:* NS under bf16 on a rank-deficient/tiny-norm matrix — guarded by `X/(X.norm()+eps)` and the singular-value smoke test; weight-renorm div-by-zero — guarded by `+1e-7`.

- **`train.py:244-250` (optimizer construction in `main()`):** replace the single `optim.SGD(...)` with:
  - Partition `model.parameters()` (requires_grad only): `p.ndim==4 → muon_params`, else `→ sgd_params`.
  - `muon_opt = Muon(muon_params, lr=PEAK_LR_MUON, momentum=MOMENTUM, nesterov=True, ns_steps=MUON_NS_STEPS)` (momentum = the existing `MOMENTUM=0.9`, shared — unchanged from EXP-008).
  - `sgd_opt = optim.SGD(sgd_params, lr=PEAK_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True)` (byte-identical to EXP-008's SGD, now over the non-conv params only).
  - *Why:* scopes the change to the conv weights only (the airbench Muon package) — every other param keeps its exact EXP-008 SGD optimizer, so the *only* thing that changes vs EXP-008 is the conv-weight training method (see Summary: package-level, not isolated-orthogonalization, attribution).

- **`train.py:286-304` (loop body — LR schedule, zero_grad, step):** compute the shared triangular fraction once, drive both optimizers:
  - Replace the `lr = PEAK_LR * ...` + `for g in optimizer.param_groups` block with: compute `frac = progress/PCT_START` (if `progress<PCT_START`) else `(1.0-progress)/(1.0-PCT_START)`; then `lr_muon = PEAK_LR_MUON*frac`, `lr_sgd = PEAK_LR*frac`; `for g in muon_opt.param_groups: g["lr"]=lr_muon` and `for g in sgd_opt.param_groups: g["lr"]=lr_sgd`.
  - **MUST-FIX (review #4 — crash guard):** the existing step-50 print (`train.py:330`) references a now-undefined `lr`. Update that print's `lr: {lr:.4f}` to `lr_m: {lr_muon:.4f} lr_s: {lr_sgd:.4f}` (both variables defined above). Failing to do this is a guaranteed `NameError` at step 50.
  - `optimizer.zero_grad(set_to_none=True)` → `muon_opt.zero_grad(set_to_none=True); sgd_opt.zero_grad(set_to_none=True)`.
  - `optimizer.step()` → `muon_opt.step(); sgd_opt.step()`.
  - *Why:* keeps the proven time-based triangular anneal-to-~0 (necessary for the EMA tail) for both groups, each scaled by its own peak.
  - *Edge:* the EMA update (`train.py:308-309`) reads `model.parameters()` after the step and is optimizer-agnostic — **left untouched**. *EMA×renorm watch-item (review #3):* because the renorm pins ‖conv‖≈√out each step while BN buffers are EMA-averaged separately (`use_buffers=True`), the scored `ema_model` averages near-norm-pinned conv weights against separately-averaged BN stats. This is the SAME mechanism EXP-002 validated (+0.50pp, EMA weights ≠ instantaneous, BN buffers EMA'd) and is mitigated by BatchNorm's scale-invariance to conv-weight magnitude — but it is a genuine unknown, so watch the EMA eval trajectory: an anomalously low or NaN eval acc relative to the (decreasing) train loss is the signal this interaction broke.

## Configuration Changes
- Conv-weight optimizer: `SGD-Nesterov(lr=0.4, mom=0.9, wd=5e-4)` → `Muon(lr=0.24, mom=0.9, ns_steps=3, weight-renorm, no wd)` (airbench-proven Muon config for this net family; LR grounded in airbench's 0.24, not a guess).
- `PEAK_LR_MUON`: new `= 0.24` (airbench94_muon Muon-group LR; transfers across momentum because NS normalizes update scale).
- `MUON_NS_STEPS`: new `= 3` (airbench value; cheaper than the proposal's 5 → less throughput risk, review concern #5).
- Non-conv params (fc/BN/α): `SGD-Nesterov(lr=PEAK_LR=0.4, mom=0.9, wd=5e-4)` — **unchanged** from EXP-008 (no new guess on the secondary group).
- Everything else (BATCH_SIZE 512, MOMENTUM 0.9, WEIGHT_DECAY 5e-4, LABEL_SMOOTHING 0.2, PCT_START 0.15, SCALE_OUT 0.125, EMA_DECAY 0.998, EMA_WARMUP_FRAC 0.15, TTA_START_FRAC 0.8, augmentation, seeds) — **unchanged**.

## Execution Environment
- Method: local — `CUDA_VISIBLE_DEVICES=1 uv run train.py > run.log 2>&1`, wrapped in `timeout 600`, launched in background with a sentinel-file completion guard (per EXP-008 protocol).
- Resources: single GPU **1** (mandatory — GPU 0 in use), H20, ~1.6 GB VRAM expected (optimizer-only change; Muon adds one momentum buffer per conv, same as SGD had).
- Estimated runtime: ~440–460s wall (300s training budget + eval/startup), well under the 600s/10-min kill.
- Log output: `run.log` (gitignored); primary source of truth. Remove after recording per goal procedure.
- Tool skill: none (local run).

## Abort Criteria
- **NaN/inf or diverging loss** in the first ~100 steps (Muon LR too high / NS unstable): `grep -qi "nan\|inf" run.log` or `loss:` visibly exploding → kill the process, classify the trajectory as "LR-too-high", do NOT retry blindly (record directional retune for loop 2).
- **No output / log not growing** for >120s after launch → infra stall; kill and inspect.
- **Wall > 600s** (`timeout` exit 124) → wall-kill failure.
- **Smoke test fails in Milestone 1** (non-finite NS output, max singular value ≥ 2 = blow-up, or the renorm assertion fails) → do NOT launch the official run; fix the implementation first. (Consistent with the Milestone-1 gate: finiteness + no blow-up + correct renorm; the lower singular-value end is informational, not a fail condition.)

## Verification Protocol

### Verification Procedure

Baseline (from `exp-index.sh baseline`): **96.38%**. Bar: **best_test_acc ≥ 96.48%** (≥ +0.10pp) AND clearly above the ~0.1pp noise floor.

**Pre-run smoke (Milestone 1, gate before launch):**
```bash
cd <project-root>
CUDA_VISIBLE_DEVICES=1 uv run python -c "
import torch
from train import zeropower_via_newtonschulz5, Muon
dev='cuda'
for shape in [(512,4608),(4608,512),(256,2304),(10,512)]:
    G=torch.randn(*shape,device=dev)                 # exercise the real CUDA bf16 path
    O=zeropower_via_newtonschulz5(G,3)
    sv=torch.linalg.svdvals(O.float())
    assert torch.isfinite(O).all(), ('nonfinite',shape)
    assert float(sv.max())<2.0, ('blowup',shape,float(sv.max()))
    print(shape,'sv[min,max]=',round(float(sv.min()),3),round(float(sv.max()),3))
# one optimizer step on a tiny conv param, on CUDA
import math
p=torch.nn.Parameter(torch.randn(8,4,3,3,device=dev)); p.grad=torch.randn_like(p)
opt=Muon([p],lr=0.24,momentum=0.9,nesterov=True,ns_steps=3); opt.step()
assert torch.isfinite(p).all(), 'nonfinite param after step'
# renorm pins ||p|| to sqrt(out)=sqrt(8) BEFORE the -lr*update nudge, so the
# post-step norm sits near sqrt(8) but not exactly (update moves it off) — assert it is in a tight band.
n=float(p.data.norm()); assert abs(n-math.sqrt(8))<1.0, ('renorm-off',n)
print('step OK, |p|=',round(n,3),'(target ~%.3f)'%math.sqrt(8))
print('SMOKE_PASS')
"
```
Pass = prints `SMOKE_PASS` (all outputs finite, max singular value < 2, and `|p|` within 1.0 of √8≈2.828 confirming the renorm fired). Lower singular values on the random test input are expected and fine. **The smoke runs on CUDA (GPU 1) so the bf16 Newton-Schulz path that the real run uses is exercised.**

**Necessary condition 1 — run completes within budget, prints valid metric, no crash, ≤10 min wall:**
```bash
CUDA_VISIBLE_DEVICES=1 timeout 600 uv run train.py > run.log 2>&1   # launched per Execution Environment
grep -E "^best_test_acc:|^training_seconds:|^num_epochs:|^total_seconds:|^peak_vram_mb:|^num_params:" run.log
```
Pass = non-empty grep with a valid `best_test_acc`; `timeout` exit ≠ 124. Empty grep ⇒ crash → `tail -n 60 run.log`.
- **Budget-used hard check (plan-review #11):** `TS=$(awk '/^training_seconds:/{print $2}' run.log)`; require `299 ≤ TS ≤ 301` (`awk "BEGIN{exit !($TS>=299 && $TS<=301)}"`). A `training_seconds` materially below 300 means a premature stop / timer bug — fail (the run did not use the budget), not a valid metric.

**Necessary condition 2 — improvement ≥ +0.10pp over baseline:**
- Parse `BEST=$(awk -F'[: %]+' '/^best_test_acc:/{print $2}' run.log)`. Pass iff `BEST ≥ 96.48` (i.e. `awk "BEGIN{exit !($BEST>=96.48)}"`). Use `best_test_acc:` (NOT the per-epoch `best:` field) to avoid circularity.
- **Anti-bookkeeping check (plan-review #10):** confirm the summary `best_test_acc` equals the true max of the per-epoch evaluator trace: `MAXEP=$(grep -oE "test_acc: [0-9.]+" run.log | awk '{print $2}' | sort -g | tail -1)`; require `MAXEP == BEST` (within 0.01). A mismatch ⇒ the summary best was fabricated/mis-bookkept (reward-hack path, since only train.py is editable) → classify `invalid`.

**Necessary condition 3 — genuine, in-scope, no throughput confound:**
- **True scope (plan-review #12):** `git status --porcelain` shows ONLY `M train.py` (run.log is gitignored). Any other staged/modified/untracked tracked file ⇒ out-of-scope change → `invalid`. Also `git diff --quiet -- prepare.py` (frozen eval byte-untouched) and `ls *.py` = `prepare.py train.py` (no stray imported modules).
- `num_params:` == `7,784,627` (optimizer-only change; `awk '/^num_params:/{print $2}' run.log` exact compare) — confirms architecture byte-identical.
- **Throughput gate (plan-review #2 — separate Muon-overhead from host contention):** record `num_epochs`, `total_seconds`, and per-step **img/s** (from the step prints, e.g. `grep "img/s" run.log | tail -5`). EXP-008 ran ~25–26k img/s @150 epochs. Two distinct cases if `num_epochs < 142`:
  - **img/s materially below ~24k** ⇒ Muon's Newton-Schulz work itself is too expensive → this is a **REAL method failure** under the 300s budget (Muon's cost is part of the method), NOT an external confound. Count it as the verdict; loop-2 action is `ns_steps 3→1` or a cheaper NS, not a free pass.
  - **img/s ≈ EXP-008 (≥~24k) but epochs still low** ⇒ external shared-host (GPU 0) contention inflating wall time → genuine confound; flag and do not treat a loss as refutation of Muon.
  ns_steps=3 is estimated <2% step overhead, so img/s≈EXP-008 and an in-band epoch count [142,150] is the expected/healthy case.
- Seed lines (`torch.manual_seed(42)`, `torch.cuda.manual_seed(42)`) intact; ≤1 `evaluator.evaluate` per epoch (loop structure unchanged).

**Pre-registered trajectory diagnostics (read regardless of pass/fail — the informative part):**
```bash
grep "eval ep" run.log | head -30   # ep-by-ep test_acc
```
- *Muon working:* ep10 ≳ 85%, ep25 ≳ 92.3% (at/above EXP-008), tail fully anneals (final not still-rising) and finishes ≥ 96.48.
- *LR too low:* stable but slower than EXP-008 (ep25 well below 92.3, e.g. ~90%), monotone, finishing below baseline → loop-2 action: raise PEAK_LR_MUON (~0.4–0.5).
- *LR too high / NS unstable:* early loss spikes / jagged acc / NaN → loop-2 action: lower PEAK_LR_MUON (~0.12).
- *Falsification of the better-minimum claim:* trajectory stable AND tail fully annealed (not best==final-rising) yet `best_test_acc < 96.38` → Muon helped convergence but not generalization on this net; lever exhausted.

### Informational Metrics (Optional)
- peak_vram_mb: `awk '/^peak_vram_mb:/{print $2}' run.log` — soft-constraint awareness (expect ~1.6 GB).
- training_seconds / num_epochs / num_steps: `grep -E "^training_seconds:|^num_epochs:|^num_steps:" run.log` — budget/epoch confirmation.
- num_params: `awk '/^num_params:/{print $2}' run.log` — expect 7,784,627 (unchanged).
