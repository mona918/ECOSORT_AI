import torch
import matplotlib.pyplot as plt

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from transfer_model import EcoSortResNet


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# PATHS
# ==========================================

base_dir = Path(__file__).resolve().parent
project_dir = base_dir.parent

model_path = base_dir / "transfer_resnet18.pth"
test_dir = project_dir / "data" / "test"

print("Model path:", model_path)
print("Model exists:", model_path.exists())


# ==========================================
# CLASSES
# ==========================================

classes = [
    "dry",
    "e_waste",
    "recyclable",
    "wet"
]


# ==========================================
# TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# DATASET
# ==========================================

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Test images:", len(test_dataset))
print("Classes:", test_dataset.classes)


# ==========================================
# MODEL
# ==========================================

model = EcoSortResNet()

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ==========================================
# FIND MISCLASSIFIED IMAGES
# ==========================================

wrong_images = []

image_index = 0


with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        predictions = outputs.argmax(
            dim=1
        )

        for i in range(len(images)):

            actual = labels[i].item()
            predicted = predictions[i].item()

            if actual != predicted:

                wrong_images.append({
                    "index": image_index,
                    "path": test_dataset.samples[
                        image_index
                    ][0],
                    "actual": classes[actual],
                    "predicted": classes[predicted]
                })

            image_index += 1


# ==========================================
# RESULTS
# ==========================================

print()
print("==========================================")
print("MISCLASSIFICATION RESULTS")
print("==========================================")

print(
    "Total test images:",
    len(test_dataset)
)

print(
    "Total misclassified images:",
    len(wrong_images)
)

print()


# ==========================================
# PRINT ALL MISCLASSIFICATIONS
# ==========================================

for item in wrong_images:

    print(
        f"{item['index']} | "
        f"Actual: {item['actual']} | "
        f"Predicted: {item['predicted']}"
    )


# ==========================================
# SHOW FIRST 20
# ==========================================

number_to_show = min(
    20,
    len(wrong_images)
)

if number_to_show == 0:

    print("No misclassified images found.")

else:

    fig, axes = plt.subplots(
        4,
        5,
        figsize=(15, 12)
    )

    axes = axes.flatten()

    for i in range(20):

        if i < number_to_show:

            item = wrong_images[i]

            image = plt.imread(
                item["path"]
            )

            axes[i].imshow(image)

            axes[i].set_title(
                f"Actual: {item['actual']}\n"
                f"Pred: {item['predicted']}",
                fontsize=9
            )

            axes[i].axis("off")

        else:

            axes[i].axis("off")


    plt.tight_layout()


    # ======================================
    # SAVE IMAGE
    # ======================================

    output_path = (
        base_dir /
        "20_misclassified_images.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print()
    print("Misclassified image grid saved to:")
    print(output_path)
    