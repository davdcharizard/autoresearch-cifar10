import gc
import os
import sys

import torch
import torch.nn.functional as F
from torchvision import transforms

sys.path.insert(0, os.getcwd())
import train


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    mean, std = (0.4914, 0.4822, 0.4465), (1, 1, 1)
    strong = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=1, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    loader = train.make_train_loader(strong, collate_fn=train.cutmix_collate)
    model = train.ResNet(3, 10, 2).cuda().train()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )
    inputs, targets = next(iter(loader))
    inputs = inputs.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)
    optimizer.zero_grad()
    logits0 = model(inputs)
    loss0 = F.cross_entropy(logits0, targets)
    loss0.backward()
    fc_weight_norm0 = model.fc.weight.norm().item()
    fc_grad_norm0 = model.fc.weight.grad.norm().item()
    max_grad_norm0 = model.max_fc.weight.grad.norm().item()
    optimizer.step()
    ratio1 = (model.max_fc.weight.norm() / model.fc.weight.norm()).item()
    with torch.no_grad():
        logits1 = model(inputs)
        loss1 = F.cross_entropy(logits1, targets).item()
        prediction_counts = torch.bincount(logits1.argmax(1), minlength=10).cpu()
    print(f"initial_loss={loss0.item():.6f}")
    print(f"initial_fc_weight_norm={fc_weight_norm0:.6f}")
    print(f"first_fc_grad_norm={fc_grad_norm0:.6f}")
    print(f"first_max_grad_norm={max_grad_norm0:.6f}")
    print(f"max_to_fc_grad_ratio={max_grad_norm0 / fc_grad_norm0:.6f}")
    print(f"post_first_step_max_readout_ratio={ratio1:.6f}")
    print(f"same_batch_post_step_loss={loss1:.6f}")
    print(f"same_batch_post_step_prediction_counts={prediction_counts.tolist()}")
    train.shutdown_train_loader(loader)
    del loader, model, optimizer
    gc.collect()


if __name__ == "__main__":
    main()
