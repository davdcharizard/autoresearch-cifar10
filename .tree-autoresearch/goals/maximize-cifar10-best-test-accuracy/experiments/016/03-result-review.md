# Adversarial Audit — EXP-016 (106-State Trailing Uniform Clean-Tail SWA)

## 1. Numerical consistency (recomputed from the embedded round times)

Parent `[3.147468, 3.138780, 3.165140, 3.140729, 3.163192]`, candidate `[3.145776, 3.155481, 3.146648, 3.145871, 3.143007]`.

- Paired ratios recompute to `0.9994624, 1.0053208, 0.9941576, 1.0016373, 0.9936187` — matches the recorded `[0.99946230, 1.00532079, 0.99415781, 1.00163726, 0.99361867]` to 7 decimals.
- Median ratio = `0.99946230` ✓ (passes ≤1.005). Max ratio = `1.00532079` ✓ (passes ≤1.02).
- MAD about the median: deviations `{0, 0.00217496, 0.00530449, 0.00584363, 0.00585849}` → MAD `0.00530449`; MAD/median = `0.00530449/0.99946230 = 0.0053073` — reproduces the recorded `0.0053073426` exactly, and exceeds the preregistered `0.005` ceiling. **The FAIL is arithmetically correct.**
- Parent drift `0.00837499` = (max−min)/median = `0.026360/3.147468` ✓, under the 0.03 gate.

No fabricated or internally contradictory numbers found.

## 2. Process integrity

- Only `train.py` modified; `git diff --check` clean; `py_compile` exit 0 — scope constraint respected.
- GPU identity confirmed as physical GPU 0, `NVIDIA H20`, UUID `GPU-b1bc897d-...c02a633`, single visible device.
- **No metric run was launched**, no `run.log` created, no seed change, no gate relaxation, no retry of the decisive numeric preflight. The failing gate is recorded verbatim with an explicit "Do NOT retry" note. This is the honest disposition; the agent could have quietly widened the ceiling and did not.
- Two harness repairs occurred **before** any numeric gate output and are covered by the plan's "one recorded repair/retry" allowance per harness: (a) `sys.path` fix in the CPU smoke, (b) replacing a byte-exact `total_memory == 97871 MiB` assertion with the plan's preregistered *approximate* H20 tolerance (PyTorch reports 97,508.75 MiB usable vs. nvidia-smi's 97,871 MiB physical). (b) tightened-then-corrected a harness bug, not a preregistered gate — the plan text itself says "approximately 97,871 MiB." Not a relaxation. No accuracy-bearing or timing-gate output was discarded.

## 3. Concerns

- **Scientific false-failure (non-blocking for the verdict):** the leaf died on a self-imposed *dispersion* gate, not on any evidence of candidate overhead — median ratio was 0.9995 (candidate marginally *faster*) and every individual ratio was within ±0.54%. With n=5, MAD/median is a high-variance statistic; 0.00531 vs. a 0.00500 ceiling is a coin-flip-level distinction. Per the review rules the preflight cannot be rerun or relaxed, so this stands, but future leaves should preregister more rounds or a ceiling calibrated to observed n=5 noise rather than importing a 0.5% figure.
- **Unverifiable projection:** projected optimizer steps `25,573.751` implies ≈85.2 steps/s, whereas the timed rounds ran 248 steps in ~3.146 s ≈ 78.8 steps/s. The gap is presumably a reweighting of the round's 12.5% SAM fraction to the production fraction, but no formula is shown. Moot here (gate never reached), yet it sits close to the 25,400 floor and should be made auditable before the next dose-gated launch.
- **Minor:** the timed round's tail (31 ordinary / 31 SAM, i.e. period-2 SAM) is an odd match for parent EXP-004's "clean-finish periodic SAM" label; worth reconfirming the emulated schedule matches production.
- No reward hacking, no evaluation-side manipulation, no hard-constraint violation.

## 4. Verdict

No primary metric result exists: execution stopped at the preregistered pre-metric gate before any training run, so `best_test_acc` was never produced. This is not "a valid metric result that fails a quality gate" (→ not `no-improvement`), and there is no hard-constraint or integrity violation (→ not `invalid`).

REVIEW: PASS
RECOMMENDED_VERDICT: crash
RECOMMENDED_METRIC: NaN
