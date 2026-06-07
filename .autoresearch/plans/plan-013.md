# Plan EXP-013: Custom SpeedNet (VGG-style, airbench-inspired)
- **Created**: 2026-05-29
- **Goal**: goals/maximize-cifar10-test-accuracy.md
- **Brainstorm**: brainstorm/brainstorm-013.md

## Milestones

### Milestone 1: Implement SpeedNet architecture
- [ ] Replace ResNet class with SpeedNet: 3 ConvBlocks (128→256→512), MaxPool, GELU, residual pairs
- [ ] Each ConvBlock: Conv(in→out) → BN → GELU → MaxPool(2) → Conv(out→out) → BN → GELU → Conv(out→out) → BN + residual → GELU
- [ ] GlobalMaxPool → Linear(512→10)
- [ ] Update EMA model creation to use SpeedNet
- [ ] Set COSINE_T_MAX appropriately (est ~60 epochs for this lighter architecture)
- [ ] Kaiming init for all conv/linear layers
- [ ] Verify ruff passes

### Milestone 2: Run and verify
- [ ] best_test_acc >= 95.83% (baseline 95.73% + 0.1%)

## Code Changes

- **train.py — Complete architecture replacement**: Remove BasicBlock, ResNet classes. Replace with SpeedNet:

```
SpeedNet architecture:
  Block 1: Conv3x3(3→128) → BN → GELU → MaxPool2x2
           Conv3x3(128→128) → BN → GELU → Conv3x3(128→128) → BN + residual → GELU
  
  Block 2: Conv3x3(128→256) → BN → GELU → MaxPool2x2
           Conv3x3(256→256) → BN → GELU → Conv3x3(256→256) → BN + residual → GELU
  
  Block 3: Conv3x3(256→512) → BN → GELU → MaxPool2x2
           Conv3x3(512→512) → BN → GELU → Conv3x3(512→512) → BN + residual → GELU
  
  GlobalMaxPool(1) → Linear(512→10)
```

9 conv layers total. Residual connections over the pairs of same-dimension convs in each block. GELU activations throughout.

- **train.py — Hyperparameters**: Remove WIDTH_MULT, NUM_BLOCKS (no longer needed). Set COSINE_T_MAX=55 (conservative estimate — lighter architecture should get ~60 epochs). Keep LR=0.1, batch=128, WD=5e-4, EMA=0.999, CutMix, label smoothing=0.1.

## Configuration Changes
- Architecture: ResNet-20 → SpeedNet (VGG-style, 9 conv layers, 128/256/512)
- Activation: ReLU → GELU
- Pooling: strided conv → MaxPool2d + GlobalMaxPool
- COSINE_T_MAX: 49 → 55

## Execution Environment
- Method: `CUDA_VISIBLE_DEVICES=0 uv run train.py > run.log 2>&1`
- Single H20 GPU, ~8 min

## Abort Criteria
- Run exceeds 10 min, traceback, loss NaN/inf

## Verification Protocol
### Verification Procedure
1. Run completion, time budget <= 300, best_test_acc >= 95.83%, eval count == epochs
