import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from src.models.robust_model import RobustCNN
from src.attacks.fgsm import fgsm_attack
from src.utils.helpers import predict

# Setup device and load model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = RobustCNN().to(device)
# Load your trained model
model.load_state_dict(torch.load("src/models/robust_model.pth", map_location=device))
model.eval()

# ---------------------------
# Streamlit App

st.title("RobustML – Adversarially Resilient AI Demo")

st.write(
    """
    Upload an image and see how the model predicts it. 
    You can also apply a simple FGSM adversarial attack to see the model's robustness.
    """
)

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension

    # Predict on clean image
    if st.button("Predict Clean"):
        pred = predict(model, tensor)
        st.success(f"Predicted Class: {pred}")

    # Apply FGSM adversarial attack
    if st.button("Apply Adversarial Attack (FGSM)"):
        # test 0 replace with real label if available
        label = torch.tensor([0]).to(device)
        adv_tensor = fgsm_attack(model, tensor, label, epsilon=0.1)
        adv_pred = predict(model, adv_tensor)
        adv_img = transforms.ToPILImage()(adv_tensor.squeeze(0).cpu())
        st.image(adv_img, caption="Adversarial Image", use_column_width=True)
        st.warning(f"Predicted Class on Adversarial: {adv_pred}")
