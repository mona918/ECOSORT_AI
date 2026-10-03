import matplotlib.pyplot as plt

from torchvision import datasets, transforms


# Original image transform
original_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# Augmented image transform
augmentation_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.RandomRotation(15),
    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3
    ),
    transforms.ToTensor()
])


# Load dataset twice
original_dataset = datasets.ImageFolder(
    "data/train",
    transform=original_transform
)

augmented_dataset = datasets.ImageFolder(
    "data/train",
    transform=augmentation_transform
)


# Pick one image
index = 0

original_image, label = original_dataset[index]
augmented_image, _ = augmented_dataset[index]


# Convert tensor to image format
original_image = original_image.permute(1, 2, 0)
augmented_image = augmented_image.permute(1, 2, 0)


# Display
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(augmented_image)
plt.title("Augmented")
plt.axis("off")

plt.tight_layout()

plt.savefig("augmentation_comparison.png")

plt.show()