# Plan EXP-053: Cross-axis compound of certified-free components — anti-aliased shortcut + de-overhead prefetch (n=2, MEAN decision)
- **Created**: 2026-06-11
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-053.md

Baseline at planning time (exp-index.sh baseline): **96.71** @ 1990397 → bar = **96.81**. σ context (EXP-027): mean ≈ 96.57, σ ≈ 0.16, σ_mean(n=2) ≈ 0.113. Component point estimates: shortcut +0.11 (pooled n=3: 96.65/96.84/96.56), prefetch +0.02 (+1 epoch measured); combined expectation ≈ +0.13. Decision statistic: **MEAN of two fresh byte-identical runs ≥ 96.81** (the max is never used; protocol validated in EXP-052 where it correctly declined a bar-clearing single draw).

Projection: params **4,286,026** (both changes zero-param); dt ≈ 22.0–22.5ms (shortcut measured zero-dt; prefetch measured −0.15ms); epochs 136–143 (046-family 138–139, 048 measured 140); steps ≈ 13,350–13,650 (union of 046-family 13,349–13,515 and 048's 13,515, ±1%).

**Failed-Approaches retry justification**: both components carry count-≥1 entries with correct individual verdicts (shortcut count-2, closed as sub-bar SINGLE change; prefetch count-1, closed as invisible-alone). This plan tests a NEW question — cross-axis additivity of the compound — which no prior experiment measured. The EXP-009 precedent (same-axis regularizer stacking, −0.46) is documented and distinguished: these two components share no currency (function quality vs step time; both measured free in heat/noise/params/numerics).

**Charging-semantics note (integrity, carried from plan-048)**: timer code and TIME_BUDGET untouched; `torch.cuda.synchronize()` fences ALL streams so overlapped prefetch copies complete inside charged windows; the layout permutation moves to uncharged DataLoader workers where the aug pipeline already lives. Throughput engineering of the EXP-000/006 class, not timer manipulation.

## Milestones

### Milestone 1: Both diffs re-applied and passing CPU sanity
- [x] **Shortcut (EXP-046/052 diff)**: in `BasicBlock.forward`, replace `shortcut = shortcut[:, :, :: self.stride, :: self.stride]` with `if self.stride != 1:` / `shortcut = F.avg_pool2d(shortcut, self.stride)` (channel zero-pad line unchanged)
- [x] **Collate (EXP-048 diff)**: module-level `def collate_channels_last(batch):` → `x, y = torch.utils.data.default_collate(batch); return x.contiguous(memory_format=torch.channels_last), y`; pass `collate_fn=collate_channels_last` to the DataLoader
- [x] **Prefetcher (EXP-048 diff)**: module-level `class CUDAPrefetcher:` — `iter(loader)` + dedicated `torch.cuda.Stream()`; `_preload()` issues `.to(device, non_blocking=True)` inside `with torch.cuda.stream(self.stream)`; `__next__` does `current_stream().wait_stream(self.stream)`, `record_stream(current_stream())` on both tensors, preloads the next pair, returns; StopIteration on exhaustion; CPU passthrough fallback when `device.type != "cuda"`
- [x] **Loop wiring**: `for inputs, targets in CUDAPrefetcher(train_loader, device):` (fresh prefetcher each epoch); DELETE the two in-step `.to(device...)`/`.to(channels_last)` lines; timed region otherwise byte-identical
- [x] Diff check: `git diff --stat` shows 1 file (train.py); hunks confined to BasicBlock.forward + module-level defs + DataLoader call + loop header
- [x] CPU sanity `/tmp/exp053_sanity.py` (merge of validated exp052 + exp048 patterns, CUDA_VISIBLE_DEVICES=""): params == 4,286,026; forward (4,3,32,32) → (4,10) finite; shortcut semantic check (avg_pool == strided slice on constant input, differs on random; pad sites exactly layer2[0]/layer3[0]); collate value-identity (`torch.equal` vs default_collate + channels_last contiguity); CPU-fallback prefetcher sequence-identity (7-batch synthetic loader, per-batch `torch.equal`, two passes); 2-epoch mini train smoke through real DataLoader with new collate + fallback prefetcher: finite decreasing loss
- [x] Static check: `uv run python -c "import ast; ast.parse(open('train.py').read())"`

### Milestone 2: Run A — gated launch, completion, pristine check
- [x] Verified `/tmp/exp046_composite.sh` (4023 bytes); launched AS-IS
- [x] GATE_DECISION D0 = 22.3ms ≤ 23; completed rc=0; best_A = 96.61
- [x] Pristine check PASS (windows 21.7–22.7ms; 139 ep; 13,428 steps; ep1 37.66 in band); run.log preserved to /tmp/exp053_runA.log
- [x] No contention kills — first launch clean

### Milestone 3: Run B — byte-identical second run
- [x] Working tree asserted unchanged (train.py +52/−7 only)
- [x] Same composite launched; D0 = 22.5ms; pristine PASS (139 ep, 13,434 steps); best_B = 96.28
- [x] MEAN = (96.61 + 96.28)/2 = **96.445** — the decision statistic

### Milestone 4: Verification executed (first-failure-stop on the MEAN) and exp-log updated
- [x] Integrity PASS both runs; Condition 1 MEAN 96.445 < 96.81 → FAIL → branch (iii) sub-additive/null; results in exp-log-053.md

## Code Changes
- **train.py** (only file; both diffs previously validated in isolation):
  1. *BasicBlock.forward shortcut* (one logic line): removes the aliasing 75%-discard strided slice at both stage transitions; carries the project's only positive pooled point estimate (+0.11, n=3). Zero params, zero dt (measured 3×).
  2. *Collate + CUDAPrefetcher + loop wiring* (~30 lines): moves layout permutation to uncharged workers and overlaps H2D on a side stream; numerics-identical (byte-equal values, EXP-048 sanity-proven); +0.15ms/step → +1 epoch.
  - Why this tests the hypothesis: the components act on provably disjoint currencies (function quality vs step time); the compound isolates cross-axis additivity — the only unfalsified positive-direction region. Interaction surface is nil: the prefetcher delivers tensors identical to EXP-048's, and the model consuming them differs only at two forward sites measured clean in EXP-046/052.
  - Risks: prefetcher lifecycle bugs (covered by sequence-identity sanity + divergence guard); `record_stream` omission (included per standard pattern); both engineering-only — each component ran pristine before.

## Configuration Changes
- None. Every training constant is the certified recipe value. DataLoader gains only `collate_fn=`.

## Execution Environment
- Method: local, `/tmp/exp046_composite.sh` verbatim, run TWICE sequentially (Run B starts only after Run A's data is extracted and run.log A preserved to /tmp/exp053_runA.log). Composite: dual launch gates (zero GPU-0 compute apps AND load < 60, poll 30s×240) → `rm -f run.log` → background `uv run train.py > run.log 2>&1` → watchdog 44×15s (GATE_KILL D0 > 26ms; CONTENTION_KILL 4 consecutive > max(D0×1.25, 26); STARTUP_KILL; NaN; divergence eval < 15% after ep5; WALL_CAP)
- Resources: GPU 0 only (H20); VRAM ~1.7GB; CIFAR-10 cached in `data/`
- Estimated runtime: ~17–18 min total for two clean runs (each ~470–505s + gate polls)
- Log output: run.log per run (Run A copied to /tmp/exp053_runA.log before Run B) + composite stdout per background task
- Tool skill: none (local)

## Abort Criteria
- Per run: NaN, eval < 15% after ep5 (auto-kill) → research/implementation failure; ep1 < 30% with clean dt → prefetcher defect: kill, fix per code-error retry rules (max 2)
- GATE_KILL D0 > 26ms → implementation defect (neither component can slow the step): fix-or-fail per code-error retry rules
- CONTENTION_KILL / STARTUP_KILL → infra: relaunch that run byte-identically when gates clear (max 2 per run, then Outcome failed)
- Wall ≥ ~660s per run → kill, failure
- If Run A completes pristine but Run B cannot be obtained within retries → record Run A only, verdict no-improvement UNLESS best_A ≥ 96.81 alone (pre-registered standard-protocol fallback, same as plan-052)

## Verification Protocol

### Verification Procedure
First-failure-stop. **Integrity pre-condition (each run independently)**: pristine profile — ≥200-step windows mean ≤ 23.5ms, none > 27; num_epochs 136–143; num_steps 13,290–13,650 (~1% around the component-union band); params 4,286,026; training_seconds 300.0; evals ≤ epochs; numerics judged by the EXP-048 trajectory criterion (trajectory rejoins family by ~ep7, plateau at family level, family test_loss ~0.18–0.19) — single ep1 reads are informational only EXCEPT the < 30% defect tripwire. A contention-tainted run is rerun (infra), never averaged.

1. **MEAN(best_A, best_B) ≥ 96.81** (pre-registered pair decision; baseline 96.71 + 0.1pp): extract each via `grep "^best_test_acc:" <log>` (timeout 10s; empty ⇒ crash path `tail -n 50`). Pass: mean ≥ 96.81 → improvement (commit BOTH diffs; TSV metric = the mean). Fail: mean < 96.81 — branches: (ii) mean ∈ [96.61, 96.80] → weak-positive; compound-of-frees closed at the resolution limit; (iii) mean ≤ 96.60 → sub-additive/null; compound-of-frees closed with a negative datum. The max of the pair is NOT a decision input.
2. **Within budget (each run)**: composite rc == 0 AND `grep "^total_seconds:"` ≤ 600 (timeout 10s)
3. **Eval cadence (each run)**: `grep -c "eval ep"` ≤ num_epochs (timeout 10s)

Cleanup per goal Procedure: delete run.log and /tmp/exp053_runA.log at loop end.

### Informational Metrics (Optional)
- Per-run num_epochs/num_steps (steps ledger: expect ≥ 13,420 if the prefetch saving reproduces; the epochs-delivered datum), final_test_loss (family ~0.18–0.19), peak_vram_mb (~1.6–1.7GB), D0 (expect 22.0–22.5ms)
- Spread |best_A − best_B| (σ check: > 0.5 flags an integrity question)
- Pooled view: compound draws vs the shortcut-only pool (96.65/96.84/96.56) — does the +0.02 epoch term show at all (recorded in the report)
