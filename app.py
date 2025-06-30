import streamlit as st
import torch
import clip
from PIL import Image
import pandas as pd
import altair as alt
import torchvision
import torch.nn as nn
from torchvision import models

# Set page
st.set_page_config(page_title="E-Waste Classifier", layout="centered")
st.title("♻️ E-Waste Classifier")
st.markdown("Upload an image of an electronic waste item. The app classifies it using a fine-tuned MobileNetV3 or CLIP zero-shot model.")

# -------------------------------
# Extended e-waste classes (for CLIP)
class_names = [
    "Battery", "Keyboard", "Microwave", "Mouse", "Mobile phone", "PCB", "Music player",
    "Printer", "Television", "Washing machine", "Laptop", "Remote control", "Headphone",
    "Earphone", "Router", "Scanner", "Camera", "Smartwatch", "Tablet", "Charger",
    "Game console", "Hard disk", "Power bank", "DVD player", "Electric iron",
    "Projector", "Set-top box", "VR headset", "USB drive", "Flashlight",
    "Graphics card", "RAM stick", "CPU processor", "Motherboard", "Smart speaker",
    "Electronic toy", "Joystick", "Monitor", "Speaker", "Fan", "Landline phone", 
    "Modem", "Electric shaver", "Ink cartridge", "Unknown e-waste"
]

# Trained class list (MobileNetV3)
trained_class_names = [
    "Battery", "Keyboard", "Microwave", "Mobile", "Mouse", 
    "PCB", "Player", "Printer", "Television", "Washing Machine"
]

CONFIDENCE_THRESHOLD = 0.25

# -------------------------------
# Load CLIP model
@st.cache_resource
def load_clip_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

clip_model, clip_preprocess, device = load_clip_model()

# Load fine-tuned MobileNetV3 model
@st.cache_resource
def load_mobilenet():
    model = models.mobilenet_v3_small(pretrained=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(trained_class_names))
    model.load_state_dict(torch.load("best_mobilenetv3.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

mobilenet_model = load_mobilenet()

# -------------------------------
# Upload UI
uploaded_file = st.file_uploader("📁 Upload an e-waste image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess for both models
    image_input_clip = clip_preprocess(image).unsqueeze(0).to(device)
    image_input_mobilenet = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], 
                                         [0.229, 0.224, 0.225])
    ])(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # MobileNet prediction
        mobilenet_output = mobilenet_model(image_input_mobilenet)
        mobilenet_probs = torch.softmax(mobilenet_output, dim=1)
        m_prob, m_idx = mobilenet_probs[0].max(0)
        mobilenet_class = trained_class_names[m_idx]

        # CLIP zero-shot prediction
        text_inputs = torch.cat([
            clip.tokenize(f"electronic waste item: {c.lower()}") for c in class_names
        ]).to(device)

        image_features = clip_model.encode_image(image_input_clip)
        text_features = clip_model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        c_prob, c_idx = similarity[0].max(0)
        clip_class = class_names[c_idx]

        # Decide best prediction
        if m_prob.item() >= 0.7:
            predicted_class = mobilenet_class
            confidence = m_prob.item()
            method_used = "Fine-Tuned MobileNetV3"
        elif c_prob.item() >= CONFIDENCE_THRESHOLD:
            predicted_class = clip_class
            confidence = c_prob.item()
            method_used = "Zero-Shot CLIP"
        else:
            predicted_class = "Unknown e-waste"
            confidence = c_prob.item()
            method_used = "Zero-Shot CLIP (Low Confidence)"

    # -------------------------------
    # Display result
    st.subheader("🔍 Prediction Result")
    st.success(f"Predicted: **{predicted_class}**")
    st.write(f"Confidence: `{confidence:.2%}` via `{method_used}`")

    # CLIP class probabilities chart
    st.subheader("📊  Class Probabilities")
    probs_df = pd.DataFrame({
        "Class": class_names,
        "Probability": similarity[0].cpu().numpy()
    })
    chart = alt.Chart(probs_df).mark_bar().encode(
        x=alt.X("Probability:Q", scale=alt.Scale(domain=[0, 1])),
        y=alt.Y("Class:N", sort="-x"),
        color=alt.condition(
            alt.datum.Class == clip_class,
            alt.value("orange"),
            alt.value("steelblue")
        )
    ).properties(height=500)
    st.altair_chart(chart, use_container_width=True)
