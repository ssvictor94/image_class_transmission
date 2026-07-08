"""
Imagenette 数据集加载（论文 Section VI 实验设置）
===============================================

论文在 Imagenette 子集（10 类, 224×224）上评估:
  - 训练集: RandAugment + ColorJitter + RandomHorizontalFlip
  - 验证集: Resize + CenterCrop

与官方 configs/pretraining_pipeline/imagenette224.yaml 一致。
"""
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import IMAGENETTE_ROOT, MICRO_BATCH_SIZE


def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def get_dataloaders(batch_size=MICRO_BATCH_SIZE, num_workers=0):
    train_tf, val_tf = get_transforms()
    root = str(IMAGENETTE_ROOT)

    train_ds = datasets.ImageFolder(f"{root}/train", transform=train_tf)
    val_ds = datasets.ImageFolder(f"{root}/val", transform=val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader
