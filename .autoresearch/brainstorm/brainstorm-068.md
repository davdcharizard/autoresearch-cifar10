# EXP-068: GELU activation replacing ReLU
GELU is smoother than ReLU (no dead neurons, continuous gradient). Used in ConvNeXt, ViT, etc.
Replace F.relu with F.gelu in BasicBlock.forward() and ResNet._features().
