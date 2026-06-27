# Plan EXP-056: Full pre-activation block reorder (ResNet v2 / WRN-native B(3,3))
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-056.md

## Context

Baseline 96.71 @ 1990397 (bar ≥ 96.81; recipe mean ≈ 96.57, σ ≈ 0.16 per EXP-027). Chosen idea: reorder all 9 BasicBlocks to full pre-activation (He et al. 2016, arXiv:1603.05027) — BN(in)→ReLU→conv1→BN(out)→ReLU→conv2 added to a CLEAN identity (no post-addition ReLU; pad shortcut takes raw x), stem becomes a bare conv (first block's BN normalizes it), and a final BN→ReLU precedes global pooling. This is the block WRN (the source of our 4× widths) used natively at our depth/width range. Zero new kernels, zero noise/heat change, nothing learnable added; params net out to exactly the baseline count by the BN-size cancellation arithmetic below. The last un-enumerated structural class (op ORDER) and the last unread standard-modernization entry — every branch terminal.

**Failed-approaches screen**: distinct from EXP-020 (added LEARNABLE projections — this removes ops from the identity path, adds nothing learnable), EXP-018 (init-time turn-on deferral — pre-act has no gate to open; He trains it from scratch on CIFAR), EXP-017/034/040+ (capacity/shape changes — this is FLOPs/param/kernel-identical). No High-Importance or count ≥ 2 entry matches op reordering. Laws: deferral (no turn-on; ep1 tripwire kept), numerics (same kernel set — only fusion ORDER could shift dt: probed before launch per the EXP-055 protocol finding), noise/heat/tail-pressure untouched, absorption priced as the central (ii) branch.

**Param cancellation arithmetic** (sanity hard-asserts the exact total): stem bn1(64) removed = −128 affine params; transition block 64→128 bn1 moves out→in = −128; transition 128→256 = −256; final BN(256) added = +512. Net 0 → expect exactly 4,286,026.

## Milestones

### Milestone 1: Code changes implemented and passing CPU sanity — COMPLETE (sanity ALL PASS incl. exact 4,286,026; probe 23.08ms PASS)
- [x] Branch `autoresearch/exp-056` created from `autoresearch/dev`
- [ ] train.py: BasicBlock — `bn1 = BatchNorm2d(in_channels)`, forward = `relu(bn1(x)) → conv1 → relu(bn2(·)) → conv2 → + raw-x pad shortcut`, NO post-addition ReLU; ResNet — stem bare (`self.bn1` removed), `self.bn_final = nn.BatchNorm2d(w3)` added, forward = conv1 → layers → `relu(bn_final(·))` → GAP → fc. Nothing else changes (optimizer/warmup/loop byte-identical to baseline).
- [ ] `uv run python -c "import ast; ast.parse(open('train.py').read())"` passes
- [ ] CPU sanity `/tmp/exp056_sanity.py` (`CUDA_VISIBLE_DEVICES="" uv run python /tmp/exp056_sanity.py`) ALL PASS:
  - params exactly **4,286,026** (the cancellation arithmetic; if the assert fails, STOP and reconcile the breakdown — do not pin a different number without explaining the delta)
  - structure: each block's `bn1.num_features == in_channels` (64,64,64, 64,128,128, 128,256,256), `bn2.num_features == out_channels`; `hasattr(model, "bn_final")` with 256 features; `not hasattr(model, "bn1")` on ResNet; block forward has no post-addition ReLU (output of a block can be negative: assert `(model.layer1[0](x_neg) < 0).any()` on a suitable input)
  - forward: output shape (8, 10), finite, train and eval modes
  - grads: after one backward, EVERY parameter (incl. bn_final, all moved bn1s) has a grad tensor (no dead branches)
  - BN stats: bn_final.running_mean and layer1[0].bn1.running_mean change across a train-mode forward
  - 3-step eager smoke at lr 0.01: loss decreases
- [ ] GPU probe `/tmp/exp056_gpu_probe.py` (EXP-055 protocol: gate-check GPU 0 free + load < 60, then compile + 3-iter warmup + time 40 steps, ~90s): **dt ≤ 24.0ms → proceed** (expected 22.0–23.0, same kernel set); 24.0–26.0 → launch anyway, the deficit prices itself in the full read; > 26.0 → branch (iv) cost-closure, do NOT launch (record the probe dt as the measurement; consult exp-report-040 gate-kill precedent for verdict bookkeeping at analyze)

### Milestone 2: Experiment launched and gates clear
- [ ] `git status` clean except train.py; tree on `autoresearch/exp-056`
- [ ] Launch via composite `/tmp/exp046_composite.sh` (reuse verbatim): Bash `run_in_background` + background until-grep watcher for `GATE_DECISION|GATE_KILL|STARTUP_KILL|GATE_TIMEOUT`, then TaskOutput(block) to wait
- [ ] GATES_CLEAR and GATE_DECISION D0 within probe-measured ± 0.5ms (expected 22.3–23.0); D0 > 26 → GATE_KILL = branch (iv) cost-closure (architecture's own cost, NOT infra — no relaunch of the same code)
- [ ] exp-log-056.md created with Implementation Notes

### Milestone 3: Run completes with family signals
- [ ] No kill markers; windows steady (no freeze/transition expected this time — single graph variant)
- [ ] rc=0; summary parses; startup expected 25–45s (fresh inductor compile for the reordered graph — no FX cache hit; STARTUP_KILL at 180s has ample margin)

### Milestone 4: Verification executed and verdict rendered — COMPLETE (M2/M3 also complete: gates poll 1, D0 24.0 in acceptance, pristine run rc=0)
- [x] Integrity pre-condition evaluated — PASS (all bands hit; params exact; trajectory criterion satisfied)
- [x] Necessary conditions checked in order, first-failure-stop — Condition 1 FAIL: 96.49 < 96.81
- [x] Pre-registered branch identified and recorded in exp-log-056.md — branch (ii): absorption-null, block-order class closed, modernization audit COMPLETE

## Code Changes

All in `train.py` on branch `autoresearch/exp-056`:

1. **BasicBlock.__init__** (L46–60): `self.bn1 = nn.BatchNorm2d(in_channels)` (was `out_channels`). conv1/conv2/bn2/stride/need_pad/pad_channels unchanged.

2. **BasicBlock.forward** (L62–70) → full pre-activation:
   ```python
   def forward(self, x):
       out = F.relu(self.bn1(x))
       out = self.conv1(out)
       out = F.relu(self.bn2(out))
       out = self.conv2(out)
       shortcut = x
       if self.need_pad:
           shortcut = shortcut[:, :, :: self.stride, :: self.stride]
           shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, self.pad_channels))
       return out + shortcut
   ```
   Decision made explicit: the pad shortcut takes RAW x (clean identity end-to-end — the design point of v2; He's shared-pre-activation shortcut applies to learnable projections, which we do not have). No post-addition ReLU anywhere.

3. **ResNet.__init__** (L73–83): delete `self.bn1 = nn.BatchNorm2d(w1)`; add `self.bn_final = nn.BatchNorm2d(w3)` after `self.layer3`. (Keeping the attribute name `bn_final` distinct avoids any stale-reference confusion.)

4. **ResNet.forward** (L101–108):
   ```python
   def forward(self, x):
       out = self.conv1(x)
       out = self.layer1(out)
       out = self.layer2(out)
       out = self.layer3(out)
       out = F.relu(self.bn_final(out))
       out = F.adaptive_avg_pool2d(out, 1)
       out = out.view(out.size(0), -1)
       return self.fc(out)
   ```

   Risks: (a) inductor fusion order changes could shift dt ±~0.5ms — probed before launch (M1) and gated at D0; (b) the bare stem feeds un-normalized conv output into layer1[0].bn1 — that BN normalizes it exactly (this is the reference design); (c) eval path unchanged (`evaluator.evaluate(base_model)`; BNs use running stats incl. bn_final, tracked normally in train mode).

   Nothing else changes: hyperparameters, optimizer (two groups, ndim split — the moved BNs remain ndim ≤ 1 no-decay), warmup (3 iters, single graph variant — no flag/freeze machinery this time), timed loop, eval, summary all byte-identical to baseline.

## Configuration Changes

- None. All constants identical to the certified recipe. The experiment variable is purely the operation ORDER of the existing modules.

## Execution Environment

- Method: local, via `/tmp/exp046_composite.sh` (verified present; reuse verbatim): dual gates (GPU-0 apps = 0 AND load < 60, poll 30s), `rm -f run.log`, `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1 &`, watchdog 44×15s (D0 gate > 26ms; contention streak 4 > max(26, D0×1.25); NaN; divergence < 15% after ep5; wall cap). Launch with Bash `run_in_background` + background until-grep gate watcher; TaskOutput(block, 600000) to wait.
- Resources: GPU 0 ONLY (gate-enforced); ~10 cores; VRAM ≈ family (~1.6GB).
- Estimated runtime: startup 25–45s (fresh compile of the reordered graph), 300.0s charged, ~138 evals ≈ 180s — total ≈ 490–530s ≤ 600.
- Log output: `run.log` at project root (source of truth); composite stdout in the background task output file. run.log deleted at experiment end per goal procedure.
- Tool skill: none (local).

## Abort Criteria

- **GATE_TIMEOUT / CONTENTION_KILL / STARTUP_KILL**: infra → relaunch byte-identically once gates clear (max 2), branch (v).
- **GATE_KILL (D0 > 26ms)**: NOT infra for this experiment — the reordered graph's own cost → branch (iv) cost-closure; do not relaunch the same code; no fix attempt (a fusion regression of that size IS the result).
- **NAN_KILL / DIVERGENCE_KILL**: research failure (pre-act instability would itself be a sign read) → Outcome failed, no blind retry; one fix-retry ONLY for a demonstrable implementation bug (e.g., wrong BN size crashing shapes — but that is caught by CPU sanity).
- **ep1 < 30%** with run alive (visible at first eval): deferral tripwire — do not kill (the trajectory criterion judges at completion), but record; a sub-30 ep1 with family plateau later is informational only (EXP-048 criterion).
- **WALL_CAP / total > 600s**: should not occur at family epoch counts; if it does, classify per evidence (loader vs foreign load) and relaunch once if transient.

## Verification Protocol

### Verification Procedure

All commands from the project root. Source of truth: `run.log` + composite task output. Timeout: TaskOutput block 600s, re-block as needed; composite silent > 20 min with no kill marker = infrastructure failure.

**Integrity pre-condition (gates all conditions)**:
1. Composite: GATES_CLEAR; D0 within [probe − 0.5, probe + 1.0]ms and ≤ 23.5 expected band (22.0–23.5); all windows mean ≤ 23.5, none > 27; no kill markers; rc=0.
2. run.log (**bands revised pre-launch from the GPU probe: pre-act graph = 23.08ms vs 22.04 family-probe — a +1.0ms fusion/op-order toll ≈ −6 epochs ≈ −0.08 by the conversion law, recorded as part of the read's interpretation**): `num_params: 4,286,026` EXACT (the cancellation arithmetic — any other value is an implementation error, not a read); `training_seconds: 300.0`; `num_epochs` ∈ [128, 138] (expect ~131–134); `num_steps` ∈ [12,400, 13,400]; D0 acceptance [22.6, 24.1] (probe ± [−0.5, +1.0]); windows mean ≤ 24.5, none > 27; `total_seconds` ≤ 600; eval count (`tr '\r' '\n' < run.log | grep -cE "^  eval ep"`) ≤ num_epochs; ep1 ≥ 30 preferred (sub-30 = informational deferral note unless plateau also depressed); trajectory rejoins family, converged plateau, final_test_loss informational (~0.18–0.20 family band; pre-act may shift CE slightly — non-gating); no NaN (`grep -ciE "loss: (nan|inf)"` = 0).
3. Contention signature (windows > 27 or alternation) → relaunch (max 2). Single-variant graph: no freeze/recompile checks this time.

**Necessary conditions (first-failure-stop; baseline re-queried at verification via `bash /SPXvePFS/users/david/Deoxys/plugins/autoresearch/skills/shared/scripts/exp-index.sh baseline .autoresearch/experiment-indices/maximize-cifar10-test-accuracy.tsv`)**:
1. **best_test_acc ≥ 96.81** (= baseline 96.71 + 0.1). Extract `tr '\r' '\n' < run.log | grep "^best_test_acc:"`; empty grep = crash → `tail -n 50 run.log`.
   - **Escalation (EXP-052 protocol, pre-registered)**: a single read ≥ 96.81 does NOT decide — launch a byte-identical second run via the same composite; improvement iff the MEAN of the pair ≥ 96.81. Max never a decision input. Reads in (96.73, 96.81): no-improvement, never single-draw promoted.
2. **Within budget**: rc=0 AND total_seconds ≤ 600.
3. **Eval cadence**: eval-line count ≤ num_epochs.

**Pre-registered outcome branches (brainstorm-056 Hypothesis, verbatim mapping)**:
- (i) ≥ 96.81 → replicate-pair escalation; improvement iff MEAN ≥ 96.81.
- (ii) ∈ [96.41, 96.73] at family signatures → absorption-null; block-order class closed; standard-modernization audit COMPLETE. Verdict no-improvement.
- (iii) < 96.41 → post-activation ordering is load-bearing at shallow depth (sign-closure, consistent with He's gains-grow-with-depth). Verdict no-improvement.
- (iv) probe dt > 26 (no launch) or GATE_KILL → cost-closure of the class; verdict bookkeeping per exp-report-040 gate-kill precedent (consult at analyze).
- (v) infra kills → relaunch (max 2); exhausted → Outcome failed, verdict crash.

### Informational Metrics (Optional)

- peak_vram_mb: `grep "^peak_vram_mb:"` — expect ≈ family (~1.6GB)
- num_epochs: `grep "^num_epochs:"` — expect 136–141
- num_steps: `grep "^num_steps:"` — family ledger ~13,300–13,600
- startup_seconds: `grep "^startup_seconds:"` — expect 25–45s (fresh compile, uncharged)
- final_test_loss: family-band check; pre-act CE shift is itself an informational datum for the report
