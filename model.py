# model.py
# loads resnet50 and trains it on the stanford dogs dataset
# using transfer learning - freeze the backbone first then unfreeze later

import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split


# some constants i use throughout
NUM_CLASSES = 120
IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_HEAD = 3      # just train the last layer first
EPOCHS_FULL = 7      # then train everything
LR_HEAD     = 1e-3
LR_FULL     = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# transforms for training - added some augmentation to help with overfitting
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# validation doesnt need augmentation
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def get_dataloaders(data_dir: str):
    # expects the stanford dogs folder structure with Images/ inside
    # splits into train/val/test (we dont actually use test here)
    full_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "Images"),
        transform=train_transform,
    )
    class_names = full_dataset.classes

    n_val   = int(0.15 * len(full_dataset))
    n_test  = int(0.15 * len(full_dataset))
    n_train = len(full_dataset) - n_val - n_test
    train_set, val_set, _ = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    # need to use val transform for validation set
    val_set.dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, "Images"),
        transform=val_transform,
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, class_names


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    # load pretrained resnet50 and replace the final layer
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # swap out the classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_model(model, train_loader, val_loader, save_path="resnet50_dogs.pth"):
    # two phase training:
    # first just train the new head, then unfreeze and fine-tune everything
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler_kwargs = dict(T_max=EPOCHS_HEAD, eta_min=1e-6)

    # phase 1 - only the head
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR_HEAD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
    print("Phase 1: training head only …")
    _run_epochs(model, train_loader, val_loader, criterion,
                optimizer, scheduler, EPOCHS_HEAD)

    # phase 2 - unfreeze everything and use a smaller lr
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_FULL, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_FULL, eta_min=1e-7)
    print("Phase 2: full fine-tuning …")
    _run_epochs(model, train_loader, val_loader, criterion,
                optimizer, scheduler, EPOCHS_FULL)

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
    return model


def _run_epochs(model, train_loader, val_loader,
                criterion, optimizer, scheduler, n_epochs):
    for epoch in range(1, n_epochs + 1):
        # training loop
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
        scheduler.step()

        # validation loop
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds       = model(imgs).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += imgs.size(0)

        print(f"  Epoch {epoch}/{n_epochs} | "
              f"Train Loss: {total_loss/total:.4f} | "
              f"Train Acc: {correct/total:.3f} | "
              f"Val Acc: {val_correct/val_total:.3f}")


def load_model(path: str, num_classes: int = NUM_CLASSES) -> nn.Module:
    # load a saved model from file
    model = build_model(num_classes)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model
