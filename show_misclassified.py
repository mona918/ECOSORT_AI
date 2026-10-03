import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from pathlib import Path

from transfer_model import EcoSortResNet


# ==============================
# DEVICE
# ==============================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==============================
# PATHS
# ==============================

base_dir = Path(__file__).resolve().parent.parent

model_path = Path(__file__).resolve().parent / "transfer_resnet18.pth"
test_dir = base_dir / "data" / "test"


# ==============================
# TRANSFORM
# ==============================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==============================
# DATASET
# ==============================

test_dataset = datasets.ImageFolder(
    test_dir,
    transform=transform
)


# ==============================
# MODEL
# ==============================

model = EcoSortResNet()

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# ==============================
# FIND ERRORS
# ==============================

wrong_images = []

with torch.no_grad():

    for index in range(len(test_dataset)):

        image, label = test_dataset[index]

        image_input = image.unsqueeze(0).to(device)

        output = model(image_input)

        predicted = torch.argmax(
            output,
            dim=1
        ).item()

        if predicted != label:

            wrong_images.append(
                (
                    index,
                    label,
                    predicted
                )
            )


# ==============================
# DISPLAY
# ==============================

classes = test_dataset.classes

print("Total misclassified:", len(wrong_images))

number_to_show = min(35, len(wrong_images))

fig, axes = plt.subplots(
    7,
    5,
    figsize=(15, 21)
)

for ax, item in zip(
    axes.flatten(),
    wrong_images[:number_to_show]
):

    index, actual, predicted = item

    image_path = test_dataset.samples[index][0]

    from PIL import Image

    image = Image.open(image_path).convert("RGB")

    ax.imshow(image)

    ax.set_title(
        f"Actual: {classes[actual]}\n"
        f"Pred: {classes[predicted]}",
        fontsize=9
    )

    ax.axis("off")


plt.tight_layout()

output_path = (
    Path(__file__).resolve().parent
    / "35_misclassified_images.png"
)

plt.savefig(
    output_path,
    dpi=150
)

plt.show()

print()
print("Saved to:")
print(output_path)