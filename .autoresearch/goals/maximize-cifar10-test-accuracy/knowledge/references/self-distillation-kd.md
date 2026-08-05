# EMA self-distillation / knowledge distillation (the loss/learning-signal axis)

Chosen lever for EXP-023 — the first attack on the TRAINING-TARGET axis after EXP-022 showed the ~96.4 ceiling is backbone-family-independent (recipe/data/loss-bound, not topology).

## Sources
- **Knowledge Distillation** (Hinton, Vinyals, Dean 2015, arXiv:1503.02531): soft teacher targets at temperature T carry "dark knowledge" (inter-class similarity). Loss `L = (1−α)·CE(y, student) + α·T²·KL(p_teacher^T ‖ p_student^T)`, where `p^T = softmax(logits/T)`. The `T²` factor restores gradient magnitude scaled down by `1/T²`.
- **Born-Again Networks** (Furlanello et al., ICML 2018): a student distilled from a same-capacity teacher (or its own earlier self) BEATS the teacher; ~0.3–0.9pp on CIFAR.
- **Mean Teacher** (Tarvainen & Valpola, NeurIPS 2017): the EMA-of-weights model is a better predictor than the raw student; a consistency/KD loss toward the EMA teacher improves generalization. Ramp the weight up over training (early EMA ≈ random).
- **When Does Label Smoothing Help?** (Müller, Kornblith, Hinton, NeurIPS 2019): LS and KD both soften targets, BUT LS ERASES inter-class similarity structure while KD PRESERVES it → KD is NOT redundant with LS; a LS teacher can even hurt a KD student. Implication: KD-on-top-of-LS can add signal, but watch over-softening — test a reduced-LS arm.

## Recipe for THIS goal (EXP-023)
- **Free teacher**: the EMA model (decay 0.998, `use_buffers=True`) is ALREADY maintained every step for eval (EXP-002). Teacher forward = eval mode, `torch.no_grad()`, same input batch. Only new cost = ONE extra forward.
- **Loss** (CORRECT direction — teacher is the TARGET): `L = (1−α)·CE_LS(student, y) + α·T²·KL(softmax(teacher/T).detach() ‖ softmax(student/T))`. Equivalent to cross-entropy of student against detached soft teacher labels. Do NOT reverse (student-as-target is wrong). T≈4 typical.
- **Ramp/gate α**: 0 while the EMA teacher is near-random; rise after `progress ≥ EMA_WARMUP_FRAC`. Tail-only gate (`progress ≥ ~0.5`) both avoids early-teacher poisoning AND bounds the extra-forward cost to protect the anneal budget.

## Constraint notes (300s budget)
- **#1 risk = under-anneal**: the extra teacher forward cuts epochs. Mitigate: torch.compile the teacher forward (banked +12%, thrice-validated EXP-014/021/022), tail-only gate, and a MANDATORY throughput smoke requiring `num_steps ≥ 12610` (130 ep) before the official run.
- **≤1 eval/epoch**: eval the EMA only (as now) — do NOT add a separate raw-model eval.
- **Anti-reward-hack**: pre-register a SMALL α/T/gate grid (≤3–4 cells), same-session α=0 control, confirm winners on a 2nd pair (the low-control-draw artifact has recurred 4×).
- **Redundancy risk**: LS0.2 + KD may over-soften → include a reduced-LS (0.1 or 0) + KD arm.
