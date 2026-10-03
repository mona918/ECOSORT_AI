import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance
from pathlib import Path

from transfer_model import EcoSortResNet


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================
# PATHS
# =========================

base_dir = Path(__file__).resolve().parent

model_path = base_dir / "transfer_resnet18_best.pth"

real_world_dir = base_dir / "real_world"


# =========================
# CLASSES
# =========================

classes = [
    "dry",
    "e_waste",
    "recyclable",
    "wet"
]


# =========================
# NORMAL TRANSFORM
# =========================

normal_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# CROP TRANSFORM
# =========================

crop_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================
# LOAD MODEL
# =========================

model = EcoSortResNet()

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model = model.to(device)
model.eval()


# =========================
# STRONG TTA
# =========================

def predict_tta(image):

    images = []

    # 1. Original
    images.append(
        normal_transform(image)
    )

    # 2. Horizontal flip
    flipped = image.transpose(
        Image.Transpose.FLIP_LEFT_RIGHT
    )

    images.append(
        normal_transform(flipped)
    )

    # 3. Center crop
    images.append(
        crop_transform(image)
    )

    # 4. Brightness slightly lower
    darker = ImageEnhance.Brightness(
        image
    ).enhance(0.85)

    images.append(
        normal_transform(darker)
    )

    # 5. Brightness slightly higher
    brighter = ImageEnhance.Brightness(
        image
    ).enhance(1.15)

    images.append(
        normal_transform(brighter)
    )

    # 6. Contrast slightly lower
    low_contrast = ImageEnhance.Contrast(
        image
    ).enhance(0.85)

    images.append(
        normal_transform(low_contrast)
    )

    # 7. Contrast slightly higher
    high_contrast = ImageEnhance.Contrast(
        image
    ).enhance(1.15)

    images.append(
        normal_transform(high_contrast)
    )

    images = torch.stack(images).to(device)

    with torch.no_grad():

        outputs = model(images)

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        # Average all TTA predictions
        avg_probabilities = probabilities.mean(
            dim=0
        )

    predicted_index = (
        avg_probabilities.argmax().item()
    )

    confidence = (
        avg_probabilities[predicted_index].item()
        * 100
    )

    return (
        classes[predicted_index],
        confidence
    )


# =========================
# TEST
# =========================

correct = 0
total = 0


for class_name in classes:

    class_dir = (
        real_world_dir / class_name
    )

    if not class_dir.exists():
        continue

    print("\n" + "=" * 60)
    print("CLASS:", class_name)
    print("=" * 60)

    for image_path in class_dir.iterdir():

        if image_path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp"
        ]:
            continue

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            predicted_class, confidence = (
                predict_tta(image)
            )

            total += 1

            if predicted_class == class_name:

                correct += 1
                result = "CORRECT"

            else:

                result = "WRONG"

            print(
                f"{image_path.name} | "
                f"Actual: {class_name} | "
                f"Predicted: {predicted_class} | "
                f"Confidence: {confidence:.2f}% | "
                f"{result}"
            )

        except Exception as e:

            print(
                f"Error processing "
                f"{image_path.name}: {e}"
            )


# =========================
# FINAL RESULT
# =========================

accuracy = (
    100 * correct / total
    if total > 0
    else 0
)

print("\n" + "=" * 60)
print("STRONG TTA REAL-WORLD RESULTS")
print("=" * 60)

print("Real-world images:", total)
print("Correct:", correct)
print("Wrong:", total - correct)

print(
    f"Strong TTA accuracy: "
    f"{accuracy:.2f}%"
)

print("=" * 60)