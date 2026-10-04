import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

from transfer_model import EcoSortResNet


# ==========================================
# PAGE SETUP
# ==========================================

st.set_page_config(
    page_title="EcoSort AI",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ EcoSort AI")
st.write("AI-powered waste classification")


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
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    # Find the folder containing app.py
    base_dir = Path(__file__).resolve().parent

    # Model file is in the same folder as app.py
    model_path = base_dir / "transfer_resnet18_best.pth"
    MODEL_URL="sha256:38a765184b82718a2fa00a589233ce9463aac0fd4b642d4f8f0816dac2bb2e88 "
    # Create model
    model = EcoSortResNet()

    # Load trained weights
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model


model = load_model()


# ==========================================
# IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# IMAGE UPLOAD
# ==========================================

st.subheader("Upload a waste image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


# ==========================================
# CAMERA
# ==========================================

st.subheader("Or take a photo")

camera_file = st.camera_input(
    "Take a picture"
)


# Use uploaded image OR camera image
file = uploaded_file if uploaded_file else camera_file


# ==========================================
# PREDICTION
# ==========================================

if file is not None:

    # Open image
    image = Image.open(file).convert("RGB")

    # Display image
    st.image(
        image,
        caption="Input Image",
        use_container_width=True
    )

    # Transform image
    image_tensor = transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to CPU/GPU
    image_tensor = image_tensor.to(device)


    # ======================================
    # MODEL PREDICTION
    # ======================================

    with torch.no_grad():

        output = model(image_tensor)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]


    # ======================================
    # TOP PREDICTION
    # ======================================

    predicted_index = probabilities.argmax().item()

    predicted_class = classes[predicted_index]

    confidence = probabilities[
        predicted_index
    ].item()


    # ======================================
    # DISPLAY RESULT
    # ======================================

    st.subheader("Prediction")

    st.success(
        f"{predicted_class.upper()} "
        f"({confidence * 100:.2f}% confidence)"
    )


    # ======================================
    # TOP 3 PREDICTIONS
    # ======================================

    st.subheader("Top 3 Predictions")

    top_values, top_indices = torch.topk(
        probabilities,
        k=3
    )

    for value, index in zip(
        top_values,
        top_indices
    ):

        class_name = classes[index.item()]

        confidence_value = value.item()

        st.write(
            f"**{class_name}** — "
            f"{confidence_value * 100:.2f}%"
        )

        st.progress(
            confidence_value
        )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "EcoSort AI — Waste Classification using "
    "Transfer Learning with ResNet18"
)
