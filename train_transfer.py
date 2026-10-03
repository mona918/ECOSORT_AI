import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from transfer_model import EcoSortResNet
from sklearn.metrics import accuracy_score, classification_report
import numpy as np


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# =========================
# PATHS
# =========================

base_dir = Path(__file__).resolve().parent.parent

train_dir = base_dir / "data" / "train"
val_dir = base_dir / "data" / "validation"

model_dir = Path(__file__).resolve().parent

best_model_path = model_dir / "transfer_resnet18_best.pth"


# =========================
# TRANSFORMS
# =========================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        224,
        scale=(0.85, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        12
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# DATASET
# =========================

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=val_transform
)

print("\nClasses:")
print(train_dataset.classes)

print("\nTraining images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# =========================
# CLASS COUNTS
# =========================

class_counts = np.bincount(
    train_dataset.targets
)

print("\nClass counts:")

for class_name, count in zip(
    train_dataset.classes,
    class_counts
):
    print(f"{class_name}: {count}")


# =========================
# CLASS WEIGHTS
# =========================

# Inverse-frequency weighting
weights = 1.0 / torch.tensor(
    class_counts,
    dtype=torch.float32
)

# Normalize weights so average weight = 1
weights = weights / weights.mean()

weights = weights.to(device)

print("\nClass weights:")

for class_name, weight in zip(
    train_dataset.classes,
    weights
):
    print(
        f"{class_name}: {weight.item():.4f}"
    )


# =========================
# DATALOADERS
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


# =========================
# MODEL
# =========================

model = EcoSortResNet()

model = model.to(device)


# =========================
# LOSS
# =========================

criterion = nn.CrossEntropyLoss(
    weight=weights,
    label_smoothing=0.05
)


# =========================
# OPTIMIZER
# =========================

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=1e-4,
    weight_decay=1e-4
)


# =========================
# LR SCHEDULER
# =========================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)


# =========================
# TRAINING
# =========================

epochs = 12

best_val_accuracy = 0.0

print("\nStarting training...\n")


for epoch in range(epochs):

    # ---------------------
    # TRAIN
    # ---------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(
        train_loader
    ):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = outputs.argmax(
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if (batch_idx + 1) % 25 == 0:

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Batch [{batch_idx+1}/{len(train_loader)}]"
            )

    train_loss = running_loss / total
    train_accuracy = correct / total


    # ---------------------
    # VALIDATION
    # ---------------------

    model.eval()

    val_loss = 0.0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )


    val_loss /= len(val_dataset)

    val_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )


    # ---------------------
    # LR SCHEDULER
    # ---------------------

    scheduler.step(
        val_accuracy
    )


    current_lr = optimizer.param_groups[0]["lr"]


    # ---------------------
    # PRINT RESULTS
    # ---------------------

    print("\n" + "=" * 60)

    print(
        f"Epoch {epoch+1}/{epochs}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )

    print(
        f"Learning Rate: "
        f"{current_lr:.7f}"
    )


    # ---------------------
    # SAVE BEST MODEL
    # ---------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print(
            "\n*** BEST MODEL SAVED ***"
        )

        print(
            f"Best validation accuracy: "
            f"{best_val_accuracy * 100:.2f}%"
        )

    print("=" * 60)


# =========================
# FINAL
# =========================

print("\nTraining complete.")

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "\nBest model saved at:"
)

print(best_model_path)