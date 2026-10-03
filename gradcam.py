import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image

from torchvision import transforms, datasets
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

model_path = base_dir / "transfer_resnet18_best.pth"
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
# LOAD MODEL
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
# LOAD TEST DATASET
# ==========================================

dataset = datasets.ImageFolder(
    test_dir,
    transform=transform
)

print("Test images:", len(dataset))


# ==========================================
# SELECT IMAGE
# ==========================================

image_path = Path(dataset.samples[0][0])
actual_class = dataset.classes[
    dataset.samples[0][1]
]

print("Image:", image_path)
print("Actual class:", actual_class)


# ==========================================
# ORIGINAL IMAGE
# ==========================================

original_image = Image.open(
    image_path
).convert("RGB")

input_tensor = transform(
    original_image
).unsqueeze(0).to(device)


# ==========================================
# GRAD-CAM
# ==========================================

activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# Last convolutional block
target_layer = model.model.layer4[-1].conv2

forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


# ==========================================
# FORWARD PASS
# ==========================================

output = model(input_tensor)

predicted_index = output.argmax(
    dim=1
).item()

predicted_class = classes[predicted_index]

print("Predicted class:", predicted_class)


# ==========================================
# BACKWARD PASS
# ==========================================

model.zero_grad()

score = output[0, predicted_index]

score.backward()


# ==========================================
# CREATE CAM
# ==========================================

weights = gradients.mean(
    dim=(2, 3),
    keepdim=True
)

cam = (
    weights * activations
).sum(dim=1).squeeze()

cam = F.relu(cam)

cam = cam.detach().cpu().numpy()

if cam.max() != 0:
    cam = cam / cam.max()


# ==========================================
# RESIZE CAM
# ==========================================

image_width, image_height = original_image.size

cam_image = Image.fromarray(
    np.uint8(cam * 255)
)

cam_image = cam_image.resize(
    (image_width, image_height)
)

cam = np.array(cam_image) / 255.0


# ==========================================
# DISPLAY
# ==========================================

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

plt.imshow(original_image)

plt.title(
    f"Actual: {actual_class}"
)

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(original_image)

plt.imshow(
    cam,
    cmap="jet",
    alpha=0.45
)

plt.title(
    f"Predicted: {predicted_class}"
)

plt.axis("off")

plt.tight_layout()


# ==========================================
# SAVE
# ==========================================

output_path = base_dir / "gradcam_result.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print()
print("Grad-CAM saved to:")
print(output_path)


# ==========================================
# REMOVE HOOKS
# ==========================================

forward_handle.remove()
backward_handle.remove()
