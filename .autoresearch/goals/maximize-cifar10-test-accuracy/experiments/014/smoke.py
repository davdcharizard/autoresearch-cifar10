"""EXP-014 smoke: torch.compile correctness + off-budget invariant.

Replicates the EXACT warmup block from train.py and asserts (for BOTH layer2_width
320 and 256): finite compile+warmup, param aliasing under torch.compile, BN-buffer
restore, off-budget invariant (post-warmup step dt<60ms), eval-boundary recompile
guard, and global-RNG isolation. Run:

  CUDA_VISIBLE_DEVICES=1 TORCHINDUCTOR_CACHE_DIR=$(pwd)/experiments/014/.inductor_cache \
    uv run python experiments/014/smoke.py
"""
import time

import torch
import torch.nn as nn

# Real components from the experiment's train.py (imports run module top-level,
# which builds `evaluator = Eval()` — loads the CIFAR test loader; main() is guarded).
from train import (
    BATCH_SIZE,
    EMA_DECAY,
    LABEL_SMOOTHING,
    MOMENTUM,
    NUM_CLASSES,
    PEAK_LR,
    WEIGHT_DECAY,
    ResNet9,
    evaluator,
)
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

DT_CAP_MS = 60.0
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True


def build(layer2_width):
    model = ResNet9(NUM_CLASSES, layer2_width=layer2_width).to(
        device, memory_format=torch.channels_last
    )
    # NOTE: smoke skips the off-budget ZCA whitening load (frozen conv values are
    # irrelevant to compile/aliasing/BN/dt behavior) to keep the smoke fast.
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=PEAK_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY, nesterov=True,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    ema_model = AveragedModel(
        model, multi_avg_fn=get_ema_multi_avg_fn(EMA_DECAY), use_buffers=True
    ).to(device, memory_format=torch.channels_last)
    return model, optimizer, criterion, ema_model


def warmup(model, optimizer, criterion, train_fwd):
    """EXACT copy of the train.py off-budget warmup block."""
    t_warm = time.time()
    model.train()
    bn_backup = [
        (m, m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
        for m in model.modules() if isinstance(m, nn.BatchNorm2d)
    ]
    gen = torch.Generator(device=device).manual_seed(0)
    dummy = torch.randn(BATCH_SIZE, 3, 32, 32, generator=gen, device=device).to(
        memory_format=torch.channels_last
    )
    dtgt = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), generator=gen, device=device)
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            warm_loss = criterion(train_fwd(dummy), dtgt)
        warm_loss.backward()
    optimizer.zero_grad(set_to_none=True)
    for m, rm, rv, nbt in bn_backup:
        m.running_mean.copy_(rm); m.running_var.copy_(rv); m.num_batches_tracked.copy_(nbt)
    torch.cuda.synchronize()
    return time.time() - t_warm, bn_backup


def timed_step(model, optimizer, criterion, train_fwd, x, y):
    model.train()
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = criterion(train_fwd(x), y)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0, loss


def check_width(layer2_width):
    print(f"\n=== smoke layer2_width={layer2_width} (USE_COMPILE=1) ===")
    model, optimizer, criterion, ema_model = build(layer2_width)
    train_fwd = torch.compile(model, mode="default")

    # Snapshot BN externally to validate restore independently of the warmup's own.
    bn_pre = {id(m): (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
              for m in model.modules() if isinstance(m, nn.BatchNorm2d)}

    # --- global-RNG isolation: same global draw before vs after warmup ---
    torch.manual_seed(123); a = torch.randn(4, device=device)
    warm_s, _ = warmup(model, optimizer, criterion, train_fwd)
    torch.manual_seed(123); b = torch.randn(4, device=device)
    assert torch.allclose(a, b), "global RNG consumed by warmup (local-gen isolation broken)"
    print(f"warmup_seconds:   {warm_s:.1f}")

    # --- (3) BN restore: model BN == external pre-warmup snapshot ---
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            rm, rv, nbt = bn_pre[id(m)]
            assert torch.equal(m.running_mean, rm), "BN running_mean not restored"
            assert torch.equal(m.running_var, rv), "BN running_var not restored"
            assert torch.equal(m.num_batches_tracked, nbt), "BN num_batches_tracked not restored"
    print("BN restore:       OK")

    # --- (1) finite outputs ---
    x = torch.randn(BATCH_SIZE, 3, 32, 32, device=device).to(memory_format=torch.channels_last)
    y = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,), device=device)
    model.train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = train_fwd(x)
    assert torch.isfinite(out).all(), "non-finite compiled output"
    print("finite outputs:   OK")

    # --- (2) param aliasing: a real optimizer.step changes the compiled forward ---
    model.eval()  # eval mode => deterministic forward on fixed x (BN uses running stats)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits1 = train_fwd(x).float().clone()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        criterion(train_fwd(x), y).backward()
    optimizer.step()  # update params the compiled forward reads
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits2 = train_fwd(x).float().clone()
    assert not torch.allclose(logits1, logits2, atol=1e-4), \
        "optimizer.step did NOT change compiled forward output (aliasing broken)"
    print("param aliasing:   OK (optimizer.step propagates to compiled forward)")

    # --- (4) off-budget invariant: post-warmup steps are fast (no in-loop recompile) ---
    dts = [timed_step(model, optimizer, criterion, train_fwd, x, y)[0] for _ in range(3)]
    print(f"post-warmup dts:  {[f'{d:.0f}ms' for d in dts]}")
    assert max(dts) < DT_CAP_MS, f"post-warmup step dt {max(dts):.0f}ms >= {DT_CAP_MS}ms (in-loop recompile?)"

    # --- (5) eval-boundary recompile guard: eval, then a train step stays fast ---
    _ = evaluator.evaluate(ema_model, device)   # off-budget eval on uncompiled EMA copy
    _ = evaluator.evaluate(model, device)       # raw uncompiled eval (pre-EMA path)
    dt_after_eval, _ = timed_step(model, optimizer, criterion, train_fwd, x, y)
    print(f"dt after eval:    {dt_after_eval:.0f}ms")
    assert dt_after_eval < DT_CAP_MS, \
        f"train step after eval dt {dt_after_eval:.0f}ms >= {DT_CAP_MS}ms (mode-transition recompile)"
    print(f"width {layer2_width}: PASS")
    return warm_s


warm_320 = check_width(320)  # headline cell-B config
warm_256 = check_width(256)  # cell-A config
print("\nSMOKE: ALL PASS")
print(f"warmup_seconds_320: {warm_320:.1f}")
print(f"warmup_seconds_256: {warm_256:.1f}")
