# EXP-067: AutoAugment CIFAR-10 policy replacing RandomHorizontalFlip
torchvision.transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10) uses the RL-discovered optimal augmentation policy for CIFAR-10. Different from TrivialAugment (EXP-006 failed) — this is a FIXED optimal policy, not random.
Use: RandomCrop + AutoAugment + ToTensor + Normalize. Remove RandomHorizontalFlip (AutoAugment includes it).
