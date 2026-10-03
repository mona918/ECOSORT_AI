import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from transfer_model import EcoSortResNet
from pathlib import Path


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# Paths
# --------------------------------------------------

base_dir = Path(__file__).resolve().parent.parent

test_dir = base_dir / "data" / "test"

model_path = (
    Path(__file__).resolve().parent
    / "transfer_resnet18.pth"
)


# --------------------------------------------------
# Preprocessing
# SAME as new training: 160x160
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((160, 160)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Dataset
# --------------------------------------------------

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

print("Classes:", test_dataset.classes)
print("Test images:", len(test_dataset))


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = EcoSortResNet()

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# --------------------------------------------------
# Predictions
# --------------------------------------------------

all_predictions = []
all_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# --------------------------------------------------
# Classification report
# --------------------------------------------------

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=test_dataset.classes,
        digits=4
    )
)


# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("Confusion Matrix:\n")
print(cm)

