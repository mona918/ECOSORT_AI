import torch
import time
from PIL import Image
from torchvision import transforms
from transfer_model import EcoSortResNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the same model architecture used during training
model = EcoSortResNet()

model.load_state_dict(
    torch.load(
        "transfer_resnet18.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

image = Image.open("bottle.webp").convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# Warm-up
with torch.no_grad():
    for _ in range(10):
        model(image)

# Benchmark
num_runs = 100

start = time.time()

with torch.no_grad():
    for _ in range(num_runs):
        model(image)

end = time.time()

total_time = end - start
avg_time = total_time / num_runs
images_per_second = 1 / avg_time

print("Device:", device)
print("Total time:", round(total_time, 4), "seconds")
print("Average inference time:", round(avg_time * 1000, 2), "ms/image")
print("Images per second:", round(images_per_second, 2))